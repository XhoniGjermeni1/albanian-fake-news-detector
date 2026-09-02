# Global Linear Feature Coefficients

Model hash: `52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5`

Koeficient pozitiv e shtyn decision score drejt klasës `fake` (1), ndërsa
koeficient negativ drejt `real` (0). Calibration-i sigmoid transformon score-n
në probabilitet, por nuk ndryshon renditjen e këtyre koeficientëve.

## Word n-grams drejt fake

| Rank | N-gram | Koeficienti |
|---:|---|---:|
| 1 | `poshte` | 1.1393 |
| 2 | `me␠poshte` | 1.0775 |
| 3 | `marketingun` | 1.0302 |
| 4 | `ja` | 0.9966 |
| 5 | `dhe␠shiko` | 0.9763 |
| 6 | `pamjet` | 0.9727 |
| 7 | `shiko` | 0.9488 |
| 8 | `Ja` | 0.8363 |
| 9 | `posht` | 0.8084 |
| 10 | `me␠posht` | 0.7543 |

## Word n-grams drejt real

| Rank | N-gram | Koeficienti |
|---:|---|---:|
| 1 | `në` | -1.1555 |
| 2 | `Telegrafi` | -0.9847 |
| 3 | `Online` | -0.9686 |
| 4 | `Kosova␠Sot` | -0.9571 |
| 5 | `Sot␠Online` | -0.9510 |
| 6 | `të` | -0.9398 |
| 7 | `për` | -0.8608 |
| 8 | `Shiko` | -0.7361 |
| 9 | `InfoKosova` | -0.7287 |
| 10 | `Sot` | -0.6991 |

## Character n-grams drejt fake

| Rank | N-gram | Koeficienti |
|---:|---|---:|
| 1 | `␠E␠` | 0.7746 |
| 2 | `pe␠` | 0.5817 |
| 3 | `oshte` | 0.5618 |
| 4 | `!.␠` | 0.5514 |
| 5 | `posht` | 0.5461 |
| 6 | `posh` | 0.5451 |
| 7 | `osht` | 0.5141 |
| 8 | `keti` | 0.5098 |
| 9 | `␠sh` | 0.5058 |
| 10 | `␠posh` | 0.4965 |

## Character n-grams drejt real

| Rank | N-gram | Koeficienti |
|---:|---|---:|
| 1 | `␠e␠` | -0.5855 |
| 2 | `”,␠` | -0.5388 |
| 3 | `leg` | -0.4378 |
| 4 | `at␠` | -0.4241 |
| 5 | `na␠` | -0.4087 |
| 6 | `eleg` | -0.4022 |
| 7 | `␠On` | -0.3911 |
| 8 | `Tele` | -0.3880 |
| 9 | `Tel` | -0.3872 |
| 10 | `ele` | -0.3838 |

## Interpretimi i saktë

Këto janë asociime globale të mësuara nga corpus-i. Ato nuk provojnë se një
lajm është faktikisht real ose fake dhe nuk duhen lexuar si marrëdhënie
shkakësore. Character n-grams janë fragmente ortografike dhe shpesh kanë më pak
kuptim të drejtpërdrejtë se word n-grams. Linguistic features të UI-së mbeten
descriptive-only dhe nuk janë pjesë e këtij classifier-i.
