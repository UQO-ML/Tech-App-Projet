# Analyse d'erreurs du modèle final

## Résumé
- n_test_samples: `4957`
- n_errors: `426`
- error_rate: `0.08593907605406496`
- false_negative_hate_speech: `10` | false_positive_hate_speech: `10`

## Faux négatifs hate_speech (true=hate_speech, pred!=hate_speech)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| hate_speech | offensive_language | these new nigga hipster hypebeasts love all these faggot ass bright colors and nike keeps giving them what they want smh |
| hate_speech | offensive_language | rt i don t fuck with bitch niggas i only fuck niggas bitches |
| hate_speech | offensive_language | stupid teabagger restaurant making customers pay minimum wage fee |
| hate_speech | offensive_language | lmao let these hoes be hoes ain t no savin nem |
| hate_speech | offensive_language | rt man some lame niggah when i find him i m unfollowing tf out his ass |
| hate_speech | offensive_language | harm this pussy instead rt missing yr old usc medical student last seen tuesday may harm himself |
| hate_speech | offensive_language | rt amp alla my niggas hot boys so don t bring no ice in this bitch |
| hate_speech | offensive_language | niggas ain t playin a gram of defense man george need to tell these bitch ass niggas man up |
| hate_speech | offensive_language | let s kill cracker babies wtf did i just hear wow |

## Faux positifs hate_speech (true!=hate_speech, pred=hate_speech)
| True | Pred | Text |
|---|---|---|
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| neither | hate_speech | she colored not lebanese |
| offensive_language | hate_speech | granted cracker isn t that bad because when you say that to a white person your saying their one of the slave |
| offensive_language | hate_speech | seeing elder queer couples marry really is emotional they get to live their final years with equality and with dignity |
| offensive_language | hate_speech | i should probs shave my spic stash soon its kinda getting out of hand |
| offensive_language | hate_speech | every half breed cop in america is trying to rape white women |
| offensive_language | hate_speech | we bout to be the next chik fil a lmao couple of fags said we told them if they didn t like their food to get out never happened |
| offensive_language | hate_speech | faggot |
| neither | hate_speech | i openly admit to being the level of white trash that will drive across town to the gas station with free hot dogs amp half price drinks |

## Erreurs générales (échantillon)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| neither | offensive_language | this store is so redneck |
| offensive_language | hate_speech | ok faggot |
| hate_speech | offensive_language | these new nigga hipster hypebeasts love all these faggot ass bright colors and nike keeps giving them what they want smh |
| neither | offensive_language | noap not taking them rt maca root pills making my vagina do the wop |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| neither | hate_speech | she colored not lebanese |
| neither | offensive_language | my fault it sound like it fucc nicca |
| offensive_language | hate_speech | granted cracker isn t that bad because when you say that to a white person your saying their one of the slave |
| offensive_language | hate_speech | seeing elder queer couples marry really is emotional they get to live their final years with equality and with dignity |
