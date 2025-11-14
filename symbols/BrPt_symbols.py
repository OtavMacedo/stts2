_pad = "_"
_bos_eos = "*&"
_punctuation = ';:,.!?¡¿—…"«»“” ()'
_letters = "abcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ'̃'"
_stress = "123456789"

BRPT_list = (
    [_pad]
    + list(_bos_eos)
    + list(_punctuation)
    + list(_letters)
    + list(_letters_ipa)
    + list(_stress)
)

SPACE_ID = BRPT_list.index(" ")
