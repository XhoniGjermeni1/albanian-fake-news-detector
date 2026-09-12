# Dita 8 - Përmirësimi i aplikacionit Streamlit

## Qëllimi

Qëllimi i Ditës 8 ishte përmirësimi i qartësisë, qëndrueshmërisë dhe prezantimit të aplikacionit të Ditës 7. Modeli nuk u ndryshua dhe nuk u trajnua përsëri.

## Kontrolli i Ditës 7

Aplikacioni i Ditës 7 ishte funksional:

- hapej pa gabime;
- `predict_news_for_app(title, content)` kthente të tria vendimet;
- shfaqte probabilitetet Real/Fake;
- shfaqte karakteristikat gjuhësore;
- bllokonte input-in bosh dhe paralajmëronte për tekst të shkurtër;
- përdorte modelin e kalibruar dhe pragjet 30%/70%;
- shfaqte kufizimin për fact-checking.

Testet ekzistuese të app-it kaluan `7/7` para ndryshimeve.

## Përmirësimet e ndërfaqes

Rezultati tani shfaqet me një formulim të kuptueshëm:

- `likely_real`: lajmi duket më shumë si i vërtetë sipas modelit;
- `uncertain`: modeli nuk ka siguri të mjaftueshme;
- `likely_fake`: lajmi duket më shumë si i pavërtetë sipas modelit.

Për `likely_real` dhe `likely_fake` shfaqet lidhja mes probabilitetit Fake dhe pragut përkatës. Emri teknik ruhet vetëm si informacion plotësues.

## Probabilitetet

Probabilitetet Real dhe Fake shfaqen:

- si përqindje në dy metrics të ndara;
- me dy progress bars;
- me shënimin se janë probabilitete të modelit dhe jo prova faktike.

## Zona `uncertain`

Kur probabiliteti Fake është mes 30% dhe 70%, app-i shpjegon se:

- rezultati ndodhet në zonën e pasigurt;
- modeli nuk ka siguri të mjaftueshme;
- kjo nuk do të thotë “gjysmë real, gjysmë fake”;
- lajmi duhet kontrolluar me burime të jashtme.

## Shpjegimi gjuhësor

Vlerat numerike vazhdojnë të shfaqen, por tani shoqërohen me fjali të kuptueshme për:

- gjatësinë dhe numrin e fjalëve;
- përdorimin e pikëçuditëseve;
- përdorimin e shkronjave të mëdha;
- përdorimin e shkronjave shqipe `ë/ç`;
- fjalët sensacionale;
- treguesit e burimit;
- shprehjet e pasigurisë.

Këto përshkruhen si vëzhgime mbi tekstin dhe jo si prova që lajmi është real ose fake.

## Validimi i input-it

- Input bosh: bllokohet pa bërë prediction.
- Vetëm titull: lejohet me paralajmërim.
- Vetëm përmbajtje: lejohet.
- Më pak se 20 fjalë: lejohet me paralajmërim për besueshmëri më të ulët.
- Mbi 20,000 karaktere: lejohet me paralajmërim për kohën e analizës.
- Mbi 100,000 karaktere: bllokohet për të mbrojtur qëndrueshmërinë e app-it.
- Gabimet e modelit trajtohen me mesazh të qartë dhe regjistrohen në log.

## Ngarkimi i modelit

App-i ngarkon `models/calibrated_tfidf_logreg.joblib`; nuk kryen trajnim. `st.cache_resource` e ruan modelin në memorie dhe shmang leximin nga disku pas çdo klikimi. Mungesa ose dështimi i modelit trajtohet pa rrëzuar ndërfaqen.

## Provat manuale

| Rasti | Sjellja |
|---|---|
| Input bosh | U bllokua me mesazh të qartë |
| Vetëm titull | Prediction u krye me paralajmërim |
| Vetëm përmbajtje | Prediction u krye pa crash |
| Tekst shumë i shkurtër | Prediction u krye me paralajmërim |
| Raport institucional | `likely_real`, P(Fake) 12.01% |
| Njoftim i paqartë | `uncertain`, P(Fake) 64.92% |
| Tekst clickbait | `likely_fake`, P(Fake) 98.87% |
| Shumë pikëçuditëse dhe source markers | U gjetën 6 pikëçuditëse dhe marker-at përkatës |
| Tekst me 31,500 karaktere | U analizua me paralajmërim dhe pa crash |

Shembujt provojnë rrjedhën teknike të app-it; rezultatet e tyre nuk janë fact-checking.

## Testet

Testet e Streamlit u zgjeruan nga 7 në 13 dhe mbulojnë validimin, shpjegimet njerëzore, zonën `uncertain`, paralajmërimin për fact-checking dhe input-et e ndara.

App-i u nis lokalisht dhe endpoint-i i kontrollit ktheu `HTTP 200 / ok`. Pamja me rezultat `uncertain` u kontrollua në desktop dhe në viewport mobile 390 px. Në mobile, dokumenti dhe zona kryesore kishin gjerësi 390 px, pa tejkalim horizontal.

Pamjet e kontrollit ruhen te:

```text
reports/figures/day8_app_desktop.png
reports/figures/day8_app_mobile.png
```

Rezultati i gjithë projektit:

```text
39 passed
```

## Kufizimet

- Modeli nuk kontrollon fakte, burime ose kontekst jashtë tekstit.
- Teksti i shkurtër mund të marrë probabilitet shumë të fortë.
- Probabiliteti varet nga ngjashmëria me datasetin e trajnimit.
- Pragjet 30%/70% janë pragje fillestare të zgjedhura nga analiza e Ditës 6.
- Shpjegimi gjuhësor tregon çfarë u gjet në tekst, jo pse një pretendim është i vërtetë ose i pavërtetë.

## Dita 9

Rekomandohet një testim i strukturuar me lajme të reja jashtë datasetit, krijimi i një tabele me prediction-in dhe etiketën e verifikuar manualisht, si dhe analizimi i rasteve ku modeli gabon. Ky hap do të ndihmojë vlerësimin përfundimtar dhe diskutimin e kufizimeve në diplomë.
