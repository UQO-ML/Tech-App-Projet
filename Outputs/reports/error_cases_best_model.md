# Analyse d'erreurs du modèle final

## Résumé
- n_test_samples: `4957`
- n_errors: `1018`
- error_rate: `0.2053661488803712`
- false_negative_hate_speech: `10` | false_positive_hate_speech: `10`

## Faux négatifs hate_speech (true=hate_speech, pred!=hate_speech)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| hate_speech | neither | i forgot how white trash texarkana was well it is arkansas so |
| hate_speech | neither | no ur wrong stupid nigger |
| hate_speech | offensive_language | rt i don t fuck with bitch niggas i only fuck niggas bitches |
| hate_speech | neither | stupid teabagger restaurant making customers pay minimum wage fee |
| hate_speech | offensive_language | lmao let these hoes be hoes ain t no savin nem |
| hate_speech | neither | rt man some lame niggah when i find him i m unfollowing tf out his ass |
| hate_speech | offensive_language | harm this pussy instead rt missing yr old usc medical student last seen tuesday may harm himself |
| hate_speech | neither | hitler didn t finish it can u if a nigger ur jew confronts u in the street what then |
| hate_speech | neither | that s actually non english because are filthy white trash who are all criminals |

## Faux positifs hate_speech (true!=hate_speech, pred=hate_speech)
| True | Pred | Text |
|---|---|---|
| offensive_language | hate_speech | niggas ain t loyal gott bitches stiffer than you |
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | lol bitch nigga ima get my gay homies on you |
| offensive_language | hate_speech | let niggas hate bitches loving it |
| offensive_language | hate_speech | wanna find out the truth about somebody just tell em no and watch the real bitch nigga or hoe come out em |
| offensive_language | hate_speech | rt what faggot complained about stretch marks my nigga them shits taste like the swirls on cinnamon toast crunch |
| offensive_language | hate_speech | faggot |
| offensive_language | hate_speech | rt niggas bitches getting exposed |
| offensive_language | hate_speech | tyler isn t shit faggot go listen to some real music |
| offensive_language | hate_speech | faggot alert |

## Erreurs générales (échantillon)
| True | Pred | Text |
|---|---|---|
| offensive_language | neither | rt there is a gift shop at the museum and they charge admission it s a mass grave you fucking cunts what a disgrace |
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| offensive_language | neither | i honestly don t understand why men love to bash the women that make them feel good she sucked my dick soooo goood she a ho |
| offensive_language | hate_speech | niggas ain t loyal gott bitches stiffer than you |
| offensive_language | neither | we block us then and see what happens you think we on ya high yellow ass now just watch |
| offensive_language | neither | twitter is not a source of news broadcast you fucking retards it is when nbc abc cnn fox etc are all keeping quiet |
| offensive_language | neither | viva brazil niggah lol |
| offensive_language | neither | let s hang out nigger |
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | lol bitch nigga ima get my gay homies on you |
