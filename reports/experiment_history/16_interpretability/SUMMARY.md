# Interpretimi global i modelit

## 1. Qëllimi

Të identifikoheshin word dhe character n-grams me koeficientët më të fortë drejt klasës real ose fake, pa ndryshuar prediction-et.

## 2. Çfarë u provua

Script-i hap modelin final, merr Linear SVM brenda objektit të kalibruar, lidh koeficientët me emrat e features dhe zgjedh termat me peshën më pozitive ose më negative për secilën degë Word/Character.

## 3. Rezultatet kryesore

Raporti final verifikon se koeficientët pozitivë shtyjnë score-in drejt fake dhe koeficientët negativë drejt real. Output-i i pritshëm `top_linear_features.csv` nuk ekziston aktualisht dhe nuk ka figurë të ruajtur; për këtë arsye termat individualë shënohen si të padisponueshëm dhe nuk rikrijohen.

## 4. Përfundimi

Koeficientët përshkruajnë lidhjet globale që modeli ka mësuar nga corpus-i. Ata nuk provojnë shkakun e një prediction-i dhe nuk janë dëshmi se një lajm është faktikisht real ose fake.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Nuk u bë ndryshim i mëtejshëm i modelit. Interpretability u mbajt si analizë shpjeguese pas ngrirjes së versionit final.

## 6. Artefaktet dhe script-et përkatëse

- Script: `archive/experiments/model_interpretability/linear_feature_coefficients.py`.
- Model i lexuar: `final_model/final_word_char_linear_svm_calibrated_v1.joblib`.
- Burim përfundimi: `reports/final/FINAL_REPORT.md`.
- CSV i koeficientëve: i padisponueshëm në repository-n aktual.
- Figura: të padisponueshme.
