# Testimi i kontratës së aplikacionit

## 1. Qëllimi

Të verifikohej që modeli i kalibruar, pragjet, preprocessing-u dhe ndërfaqja japin vendime të qëndrueshme dhe të kuptueshme për përdoruesin.

## 2. Çfarë u provua

U kontrolluan probabilitetet, kufijtë e pragjeve, përputhja e vendimit me probabilitetin, Unicode NFC/NFD, input-et bosh ose jo-informuese, paralajmërimet për tekst të shkurtër dhe paraqitja desktop/mobile e aplikacionit.

## 3. Rezultatet kryesore

Nuk u gjetën probabilitete të pavlefshme ose vendime jashtë rregullave të pragjeve. Për modelin e kësaj faze, vendimet e forta mbuluan 86.11% të test set-it dhe kishin accuracy 93.99%. CSV/JSON i veçantë historik nuk ekziston aktualisht; shifrat verifikohen në raportin final.

## 4. Përfundimi

Kontrata me tre nivele ishte e përshtatshme për UI, me kusht që paralajmërimi se sistemi nuk është fact-checker të mbetet i dukshëm.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Pasi kontrata funksionoi në test set-in e brendshëm, u validua dhe u përdor një benchmark i jashtëm për të matur përgjithësimin.

## 6. Artefaktet dhe script-et përkatëse

- Script vlerësimi: `archive/experiments/models/evaluate_app_system.py`.
- Helper historik: `archive/experiments/models/predict.py`.
- Teste: `tests/test_app_system.py`, `tests/test_streamlit_app.py`.
- Model: `../06_initial_probability_calibration/artifacts/calibrated_tfidf_logreg.joblib`.
- Figura: [app_desktop_initial.png](figures/app_desktop_initial.png), [app_mobile_initial.png](figures/app_mobile_initial.png), [app_desktop_refined.png](figures/app_desktop_refined.png), [app_mobile_refined.png](figures/app_mobile_refined.png).
- CSV/JSON historike: të padisponueshme.
