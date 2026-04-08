# Analyse d'erreurs du modèle final

## Résumé
- n_test_samples: `4957`
- n_errors: `432`
- error_rate: `0.0871494855759532`
- false_negative_hate_speech: `10` | false_positive_hate_speech: `10`

## Faux négatifs hate_speech (true=hate_speech, pred!=hate_speech)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| hate_speech | offensive_language | these new nigga hipster hypebeasts love all these faggot ass bright colors and nike keeps giving them what they want smh |
| hate_speech | offensive_language | faggot sack |
| hate_speech | offensive_language | rt i don t fuck with bitch niggas i only fuck niggas bitches |
| hate_speech | offensive_language | stupid teabagger restaurant making customers pay minimum wage fee |
| hate_speech | offensive_language | lmao let these hoes be hoes ain t no savin nem |
| hate_speech | offensive_language | rt man some lame niggah when i find him i m unfollowing tf out his ass |
| hate_speech | offensive_language | harm this pussy instead rt missing yr old usc medical student last seen tuesday may harm himself |
| hate_speech | offensive_language | rt amp alla my niggas hot boys so don t bring no ice in this bitch |
| hate_speech | offensive_language | niggas ain t playin a gram of defense man george need to tell these bitch ass niggas man up |

## Faux positifs hate_speech (true!=hate_speech, pred=hate_speech)
| True | Pred | Text |
|---|---|---|
| offensive_language | hate_speech | let s hang out nigger |
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| offensive_language | hate_speech | granted cracker isn t that bad because when you say that to a white person your saying their one of the slave |
| offensive_language | hate_speech | seeing elder queer couples marry really is emotional they get to live their final years with equality and with dignity |
| offensive_language | hate_speech | every half breed cop in america is trying to rape white women |
| offensive_language | hate_speech | faggot |
| neither | hate_speech | i openly admit to being the level of white trash that will drive across town to the gas station with free hot dogs amp half price drinks |
| offensive_language | hate_speech | thanksgiving with some ignore the chink eyes |
| neither | hate_speech | that wasn t as bad though drawing conclusions about white america based on a person poll is a slippery slope |

## Erreurs générales (échantillon)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| neither | offensive_language | this store is so redneck |
| offensive_language | hate_speech | let s hang out nigger |
| offensive_language | hate_speech | ok faggot |
| hate_speech | offensive_language | these new nigga hipster hypebeasts love all these faggot ass bright colors and nike keeps giving them what they want smh |
| neither | offensive_language | noap not taking them rt maca root pills making my vagina do the wop |
| hate_speech | offensive_language | faggot sack |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| neither | offensive_language | my fault it sound like it fucc nicca |
| offensive_language | hate_speech | granted cracker isn t that bad because when you say that to a white person your saying their one of the slave |
