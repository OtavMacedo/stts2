import torch
import torchaudio
import yaml
import phonemizer
import warnings
import librosa

from models import load_ASR_models, load_F0_models, build_model
from utils import recursive_munch
from symbols.BrPt_symbols import BRPT_list
from phonemizer.phonemize import _phonemize
from Utils.MLPLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from collections import OrderedDict

from phonemizer.punctuation import Punctuation
from phonemizer.logger import get_logger
from phonemizer.backend import BACKENDS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TTSInferenceEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean = -4
        self.std = 4

        self.config = yaml.safe_load(open("./config_base_cerebrium.yml"))
        self.rate = 24000

        self.backend = self._load_phonemizer_backend()
        self.model = self._load_all_models()
        self.sampler = self._create_sampler()

        # self.warmup()

    def _load_phonemizer_backend(self):
        return BACKENDS["espeak"](
            "pt-br",
            punctuation_marks=Punctuation.default_marks(),
            preserve_punctuation=True,
            with_stress=True,
            tie=False,
            language_switch="keep-flags",
            words_mismatch="ignore",
            logger=get_logger(),
        )

    def _load_all_models(self):
        ASR_config = self.config.get("ASR_config", False)
        ASR_path = self.config.get("ASR_path", False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        F0_path = self.config.get("F0_path", False)
        pitch_extractor = load_F0_models(F0_path)

        BERT_path = self.config.get("PLBERT_dir", False)
        plbert = load_plbert(BERT_path)

        model = build_model(
            recursive_munch(self.config["model_params"]),
            text_aligner,
            pitch_extractor,
            plbert,
        )

        params_whole = torch.load("epoch_base_2nd_00149.pth", map_location=self.device)
        params = params_whole["net"]

        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key])
                except Exception:
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    model[key].load_state_dict(new_state_dict, strict=False)

        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        return model

    def _create_sampler(self):
        """Cria o sampler de difusão."""
        return DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # parâmetros empíricos
            clamp=False,
        )

    def _text2phoneme(self, text):
        """Converte texto em fonemas."""
        seperator = phonemizer.separator.Separator(
            " |",
            "",
            "|",
        )
        ph_br = _phonemize(self.backend, text, seperator, False, 1, False, False)
        return ph_br

    def _length_to_mask(self, lengths):
        """Função auxiliar para criar máscaras."""
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def _preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self._preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    # def warmup(self):
    #     """Executa uma inferência inicial para aquecer a GPU."""
    #     print("[INFO] Aquecendo GPU...")
    #     try:
    #         # self.inference("A", diffusion_steps=2, embedding_scale=1)
    #         ref_s = torch.randn(1, 256).to(self.device)
    #         self.inference("A", ref_s=ref_s, diffusion_steps=2, embedding_scale=1)
    #         print("[INFO] Warm-up concluído.")
    #     except Exception as e:
    #         # Não é crítico se o warmup falhar, pode ser um pequeno erro de shape
    #         print(f"[WARN] Warmup ignorado devido a erro: {e}")

    # A função principal de inferência (MUITO IMPORTANTE manter em no_grad!)
    def inference(
        self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1
    ):
        """Executa a inferência TTS e retorna o array numpy de amostras."""

        # Conversão para tokens (mantendo o tratamento de erros)
        phs = self._text2phoneme(text)
        phs = phs.replace("|", "")
        phs = f"*{phs}&"
        tokens = []
        for i in list(phs):
            if i in BRPT_list:
                tokens.append(BRPT_list.index(i))
            else:
                print(f"ERRO: Caractere '{i}' não encontrado na lista BRPT_list.")
                raise ValueError("Caractere inválido na frase.")

        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = self._length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur[0].unsqueeze(0),
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()
