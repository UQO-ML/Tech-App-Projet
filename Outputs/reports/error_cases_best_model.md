# Analyse d'erreurs du modèle final

## Résumé
- n_test_samples: `4957`
- n_errors: `686`
- error_rate: `0.13839015533588864`
- false_negative_hate_speech: `10` | false_positive_hate_speech: `10`

## Faux négatifs hate_speech (true=hate_speech, pred!=hate_speech)
| True | Pred | Text |
|---|---|---|
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| hate_speech | offensive_language | these new nigga hipster hypebeasts love all these faggot ass bright colors and nike keeps giving them what they want smh |
| hate_speech | neither | i forgot how white trash texarkana was well it is arkansas so |
| hate_speech | offensive_language | no ur wrong stupid nigger |
| hate_speech | offensive_language | faggot sack |
| hate_speech | offensive_language | rt i don t fuck with bitch niggas i only fuck niggas bitches |
| hate_speech | offensive_language | stupid teabagger restaurant making customers pay minimum wage fee |
| hate_speech | offensive_language | i dont have any problems with zak but you seem like a faggot |
| hate_speech | offensive_language | lmao let these hoes be hoes ain t no savin nem |
| hate_speech | offensive_language | rt man some lame niggah when i find him i m unfollowing tf out his ass |

## Faux positifs hate_speech (true!=hate_speech, pred=hate_speech)
| True | Pred | Text |
|---|---|---|
| offensive_language | hate_speech | ok faggot |
| offensive_language | hate_speech | rt what faggot complained about stretch marks my nigga them shits taste like the swirls on cinnamon toast crunch |
| offensive_language | hate_speech | faggot |
| offensive_language | hate_speech | rt this faggot is a lil attention whore fuck him |
| offensive_language | hate_speech | really u use in another nigga name and u follow a nigga who doesn t fucking like you your a faggot kok eyed mother fucker |
| offensive_language | hate_speech | fuck her pussy fuck her mouth fuck her ass cause she nasty |
| offensive_language | hate_speech | fuck no faggot |
| offensive_language | hate_speech | their insult even if it s self deprecating the first thing straights go to is faggot queer tranny gay even supposed allies |
| offensive_language | hate_speech | plot twist he is a faggot |
| offensive_language | hate_speech | you cant call someone a fag if you re snuggling up with guys |

## Erreurs générales (échantillon)
| True | Pred | Text |
|---|---|---|
| offensive_language | neither | rt there is a gift shop at the museum and they charge admission it s a mass grave you fucking cunts what a disgrace |
| hate_speech | offensive_language | rt stfu hoe rt i m proud to be african american proud african american i m proud to be white racist |
| neither | offensive_language | this store is so redneck |
| offensive_language | neither | let s hang out nigger |
| offensive_language | hate_speech | ok faggot |
| offensive_language | neither | mum agrees with me ur being a wee moody cunt and you cani deal with the fact i m right |
| hate_speech | offensive_language | these new nigga hipster hypebeasts love all these faggot ass bright colors and nike keeps giving them what they want smh |
| hate_speech | neither | i forgot how white trash texarkana was well it is arkansas so |
| hate_speech | offensive_language | no ur wrong stupid nigger |
| hate_speech | offensive_language | faggot sack |
