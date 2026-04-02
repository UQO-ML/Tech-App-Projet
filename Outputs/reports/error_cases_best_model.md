# Analyse d'erreurs du modèle final

## Résumé
- n_test_samples: `4957`
- n_errors: `626`
- error_rate: `0.12628606011700624`
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
| hate_speech | offensive_language | rt rt tight pants and loud colors you look like a fag |
| hate_speech | offensive_language | rt most teachers abuse their position there s teaching with discipline and there s being a cunt most are cunts |

## Faux positifs hate_speech (true!=hate_speech, pred=hate_speech)
| True | Pred | Text |
|---|---|---|
| offensive_language | hate_speech | rt there is a gift shop at the museum and they charge admission it s a mass grave you fucking cunts what a disgrace |
| offensive_language | hate_speech | twitter is not a source of news broadcast you fucking retards it is when nbc abc cnn fox etc are all keeping quiet |
| offensive_language | hate_speech | let s hang out nigger |
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | any man even considering plastic surgery is gay gay gay imagine ya boy say i d get more hoes then u after i get these new lips |
| neither | hate_speech | noap not taking them rt maca root pills making my vagina do the wop |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| offensive_language | hate_speech | can we plz go to starbucks and do that ebola sounds like a really ghetto hood name tbh |
| offensive_language | hate_speech | granted cracker isn t that bad because when you say that to a white person your saying their one of the slave |
| offensive_language | hate_speech | seeing elder queer couples marry really is emotional they get to live their final years with equality and with dignity |

## Erreurs générales (échantillon)
| True | Pred | Text |
|---|---|---|
| offensive_language | hate_speech | rt there is a gift shop at the museum and they charge admission it s a mass grave you fucking cunts what a disgrace |
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| offensive_language | neither | we block us then and see what happens you think we on ya high yellow ass now just watch |
| offensive_language | hate_speech | twitter is not a source of news broadcast you fucking retards it is when nbc abc cnn fox etc are all keeping quiet |
| offensive_language | hate_speech | let s hang out nigger |
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | any man even considering plastic surgery is gay gay gay imagine ya boy say i d get more hoes then u after i get these new lips |
| neither | hate_speech | noap not taking them rt maca root pills making my vagina do the wop |
| offensive_language | hate_speech | people who never use any hair product are unkept white trash douche bags |
| offensive_language | neither |  |
