# Dita 18 - Integrimi i modelit final në Streamlit

## Përfundimi

Modeli klasik final u integrua me sukses në aplikacionin Streamlit. Integrimi
nuk ritrajnoi modelin dhe nuk ndryshoi TF-IDF, classifier-in, calibration-in ose
pragjet. Hash-i SHA-256 i artefaktit final mbeti
`52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5`, identik
me kandidatin e ngrirë të Ditës 16.

## Kontrata runtime

| Elementi | Vlera |
|---|---|
| Aplikacioni | `app/streamlit_app.py` |
| Artefakti | `models/final_word_char_linear_svm_calibrated_v1.joblib` |
| Manifesti | `models/final_model_v1_manifest.json` |
| Funksioni i prediction | `predict_final_news()` |
| Model ID | `albanian_fake_news_word_char_svm_sigmoid_v1` |
| Versioni | `1.0.0` |
| Përfaqësimi | Word + Character TF-IDF |
| Classifier | Linear SVM, `C=1.0` |
| Calibration | sigmoid |
| Pragjet | `<0.30` real, `0.30-0.70` uncertain, `>0.70` fake |
| Cache | `st.cache_resource` |

Manifesti dhe artefakti kontrollohen përpara prediction-it. Kur njëri mungon
ose manifesti nuk përputhet me kontratën finale, UI shfaq një mesazh të qartë
dhe çaktivizon butonin e analizës.

Karakteristikat gjuhësore vazhdojnë të shfaqen për shpjegim, por nuk i jepen
pipeline-it të modelit dhe nuk ndryshojnë probabilitetet ose vendimin.

## Regression checks

Të gjashtë rastet finale të Ditës 17 dhanë të njëjtin vendim dhe të njëjtat
probabilitete në `predict_final_news()` dhe në rrjedhën e aplikacionit.

| Rasti | ID | Label | P(fake) | Vendimi | Përputhja |
|---|---|---:|---:|---|---|
| likely real korrekt | `true_1594` | real | 0.0100% | `likely_real` | po |
| likely fake korrekt | `fake_531` | fake | 99.99997% | `likely_fake` | po |
| uncertain | `true_585` | real | 49.7187% | `uncertain` | po |
| false positive | `true_586` | real | 89.6365% | `likely_fake` | po |
| false negative | `fake_1104` | fake | 10.5938% | `likely_real` | po |
| gabim me confidence të lartë | `fake_1932` | fake | 0.0828% | `likely_real` | po |

Gabimet e njohura ruhen qëllimisht në këtë kontroll: integrimi duhet të
riprodhojë modelin e ngrirë, jo të ndryshojë rezultatet e tij.

## Input-et joideale

| Input-i | Sjellja e verifikuar |
|---|---|
| bosh | bllokohet me mesazh të qartë |
| vetëm titull | pranohet me warning |
| vetëm përmbajtje | pranohet |
| shumë i shkurtër | pranohet me warning |
| Unicode NFC/NFD | jep prediction identik |
| shumë i gjatë, brenda kufirit | pranohet me warning dhe pa crash |
| mbi 100,000 karaktere | bllokohet |
| vetëm emoji ose pikësim | bllokohet |

Warning-u që modeli nuk bën fact-checking shfaqet gjithmonë në faqen kryesore
dhe përsëritet pranë rezultatit.

## Testimi

- `python -m pytest -q`: **109 passed**, 0 failed.
- Health check i Streamlit: HTTP 200, përgjigjja `ok`.
- Kontrollet vizuale në desktop dhe në viewport mobile 390 px kaluan pa
  mbivendosje ose elemente të prera.
- Në mobile, `innerWidth` dhe `document.scrollWidth` ishin të dyja 390 px; nuk
  u gjet overflow horizontal.
- Screenshot-et ruhen te `reports/figures/day18_streamlit_final.png` dhe
  `reports/figures/day18_streamlit_final_mobile.png`.

## Probleme të rregulluara

- Runtime-i i Streamlit u kalua nga modeli i vjetër Logistic Regression te
  artefakti final i ngrirë.
- U hoqën nga app-i importet dhe path-et e modelit të vjetër.
- U shtua trajtimi i qartë për model ose manifest që mungon ose nuk përputhet.
- U dokumentuan model ID, versioni dhe roli vetëm shpjegues i linguistic
  features.

Modeli i vjetër `calibrated_tfidf_logreg.joblib` ruhet për krahasim dhe
riprodhimin e analizave historike, por nuk përdoret më nga Streamlit.

## Kufizimet

Integrimi nuk ndryshon kufizimet e modelit: bias-i sipas gjatësisë, domain
shift-i, paqëndrueshmëria e teksteve shumë të shkurtra dhe fakti që rezultati
është klasifikim gjuhësor, jo verifikim faktik.

## Rekomandim për Ditën 19

Dita 19 duhet të fokusohet te kontrolli final i përdorshmërisë dhe demonstrimit:
testim në mobile/desktop, rishikim i teksteve të UI-së, përgatitje e skenarit të
demonstrimit dhe udhëzimeve të ekzekutimit. Modeli, calibration-i dhe pragjet
duhet të mbeten të ngrira.
