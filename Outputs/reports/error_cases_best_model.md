# Analyse d'erreurs du modèle final

## Résumé
- n_test_samples: `4957`
- n_errors: `429`
- error_rate: `0.08654428081500908`
- false_negative_hate_speech: `10` | false_positive_hate_speech: `10`

## Faux négatifs hate_speech (true=hate_speech, pred!=hate_speech)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| hate_speech | offensive_language | rt i don t fuck with bitch niggas i only fuck niggas bitches |
| hate_speech | neither | stupid teabagger restaurant making customers pay minimum wage fee |
| hate_speech | offensive_language | lmao let these hoes be hoes ain t no savin nem |
| hate_speech | offensive_language | rt man some lame niggah when i find him i m unfollowing tf out his ass |
| hate_speech | offensive_language | harm this pussy instead rt missing yr old usc medical student last seen tuesday may harm himself |
| hate_speech | offensive_language | rt amp alla my niggas hot boys so don t bring no ice in this bitch |
| hate_speech | offensive_language | niggas ain t playin a gram of defense man george need to tell these bitch ass niggas man up |
| hate_speech | offensive_language | let s kill cracker babies wtf did i just hear wow |
| hate_speech | neither | they should have never gave a cracker a transmitter will flip when he sees this |

## Faux positifs hate_speech (true!=hate_speech, pred=hate_speech)
| True | Pred | Text |
|---|---|---|
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| offensive_language | hate_speech | granted cracker isn t that bad because when you say that to a white person your saying their one of the slave |
| offensive_language | hate_speech | seeing elder queer couples marry really is emotional they get to live their final years with equality and with dignity |
| offensive_language | hate_speech | every half breed cop in america is trying to rape white women |
| offensive_language | hate_speech | we bout to be the next chik fil a lmao couple of fags said we told them if they didn t like their food to get out never happened |
| offensive_language | hate_speech | faggot |
| neither | hate_speech | i openly admit to being the level of white trash that will drive across town to the gas station with free hot dogs amp half price drinks |
| offensive_language | hate_speech | chill i m not a coon rt sick of you fake bougie ass coons lol |
| neither | hate_speech | photo giving you that trailer park trash |

## Erreurs générales (échantillon)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| offensive_language | hate_speech | ok faggot |
| neither | offensive_language | noap not taking them rt maca root pills making my vagina do the wop |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| offensive_language | neither |  |
| neither | offensive_language | my fault it sound like it fucc nicca |
| offensive_language | hate_speech | granted cracker isn t that bad because when you say that to a white person your saying their one of the slave |
| offensive_language | hate_speech | seeing elder queer couples marry really is emotional they get to live their final years with equality and with dignity |
| hate_speech | offensive_language | rt i don t fuck with bitch niggas i only fuck niggas bitches |
| neither | offensive_language | munching on ww crackers amp studying nycs trans syst since nyc amp i haven t had a thing in over yrs |
