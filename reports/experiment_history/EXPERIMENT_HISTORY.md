# Historia kronologjike e eksperimenteve

Ky dokument rindërton rrjedhën e eksperimenteve të diplomës nga corpus-i raw deri te modeli final. Rendi bazohet në varësitë midis script-eve, dataset-eve, modeleve dhe output-eve që ekzistojnë në repository. Emrat historikë `dayX` ruhen vetëm te path-et origjinale; këtu fazat emërtohen sipas pyetjes shkencore.

Asnjë model nuk është ritrajnuar për këtë histori. CSV, JSON dhe PNG janë kopje të artefakteve ekzistuese. Modelet `.joblib` nuk janë kopjuar dhe referohen vetëm në vendndodhjet e tyre origjinale. Kur një output nuk ekziston, ai shënohet si i padisponueshëm.

## Rrjedha e vendimeve

```text
Dataset i vlefshëm dhe split pa leakage
→ kontroll minimal Dummy
→ Word TF-IDF + Logistic Regression
→ linguistic features kanë sinjal, por hybrid nuk përmirëson TF-IDF
→ probabilitetet kërkojnë calibration dhe zonë uncertain
→ external evaluation zbulon domain shift
→ gjatësia shpjegon vetëm një pjesë të problemit
→ Word+Character TF-IDF përfaqëson më mirë tekstin
→ Linear SVM është classifier-i më i mirë
→ C=1.0 është zgjedhja më e qëndrueshme
→ sigmoid është calibration-i më i sigurt
→ 0.30/0.70 japin vendime më të kujdesshme
→ kandidati ngrihet byte-for-byte si modeli final
→ koeficientët inspektohen vetëm për interpretim
```

## Përmbledhja e fazave

| Nr. | Faza | Rezultati ose vendimi kryesor | Përmbledhja |
|---:|---|---|---|
| 1 | Dataset dhe leakage | 3,994 artikuj; 3,195 train; 792 test zyrtar | [SUMMARY](01_dataset_validation/SUMMARY.md) |
| 2 | Dummy baseline | Accuracy 0.5038; F1 fake 0.0000 | [SUMMARY](02_dummy_baseline/SUMMARY.md) |
| 3 | Word TF-IDF + Logistic Regression | Accuracy 0.8838; F1 fake 0.8808 | [SUMMARY](03_word_tfidf_logreg/SUMMARY.md) |
| 4 | Linguistic features | Dallime të matshme, me rrezik bias-i nga gjatësia | [SUMMARY](04_linguistic_features/SUMMARY.md) |
| 5 | Linguistic/hybrid | Hybrid nuk e kaloi TF-IDF-in | [SUMMARY](05_linguistic_hybrid_models/SUMMARY.md) |
| 6 | Probabilitetet fillestare | Nevojitet calibration dhe zonë uncertain | [SUMMARY](06_initial_probability_calibration/SUMMARY.md) |
| 7 | Kontrata e aplikacionit | Pragjet dhe vendimet funksionuan në mënyrë konsistente | [SUMMARY](07_application_contract/SUMMARY.md) |
| 8 | External evaluation | Baseline accuracy 0.4750; domain shift i fortë | [SUMMARY](08_external_evaluation/SUMMARY.md) |
| 9 | Gjatësia/domain shift | Gjatësia ndikon, por nuk shpjegon gjithçka | [SUMMARY](09_length_domain_shift/SUMMARY.md) |
| 10 | Përfaqësimet TF-IDF | U zgjodh Word+Character | [SUMMARY](10_tfidf_representations/SUMMARY.md) |
| 11 | Classifier-at | U zgjodh Linear SVM | [SUMMARY](11_classifier_comparison/SUMMARY.md) |
| 12 | SVM tuning | U zgjodh C=1.0 | [SUMMARY](12_svm_tuning/SUMMARY.md) |
| 13 | Calibration | U zgjodh sigmoid | [SUMMARY](13_calibration_comparison/SUMMARY.md) |
| 14 | Thresholds | U zgjodhën 0.30/0.70 | [SUMMARY](14_threshold_selection/SUMMARY.md) |
| 15 | Modeli final | Accuracy 0.9116; artefakti u ngri pa ritrajnim | [SUMMARY](15_final_model/SUMMARY.md) |
| 16 | Interpretability | Koeficientët përdoren vetëm si sinjale globale | [SUMMARY](16_interpretability/SUMMARY.md) |

## 1. Validimi i dataset-it dhe leakage-it

**Pyetja/Problemi →** A mund të përdoret corpus-i për ML pa përzier artikuj të lidhur ose kopje ekzakte midis train-it dhe test-it?

**Eksperimenti →** `load_dataset.py`, `validate_dataset.py` dhe `build_dataset.py` strukturuan 3,994 artikuj. `clean_text.py` krijoi tekstin zyrtar të modelit dhe split-i u ndërtua sipas `pair_id`. `data_utils.py` kontrolloi edhe dublikatat ekzakte të tekstit.

**Rezultati →** U krijuan 3,195 raste train dhe 799 test. Shtatë raste test ishin kopje ekzakte të train-it dhe u përjashtuan vetëm nga evaluation, duke lënë 792 raste zyrtare.

**Përfundimi →** Çdo eksperiment duhet të përdorë split-et e ngrira dhe grupim sipas `pair_id` plus tekstit identik.

**Vendimi tjetër →** Mbi këtë bazë të përbashkët u mat fillimisht një classifier trivial. [Detaje](01_dataset_validation/SUMMARY.md)

## 2. Dummy baseline

**Pyetja/Problemi →** Sa do të arrinte një model që nuk lexon tekstin?

**Eksperimenti →** `DummyClassifier(strategy="most_frequent")` zgjodhi klasën real, sepse train-i përmban 1,598 real dhe 1,597 fake.

**Rezultati →** Accuracy ishte 0.5038, F1 weighted 0.3376 dhe F1 fake 0.0000.

**Përfundimi →** Balanca e klasave nuk jep klasifikim të dobishëm; një model duhet të mësojë nga teksti.

**Vendimi tjetër →** U përdor Word TF-IDF për të kthyer fjalët në vektorë dhe Logistic Regression për klasifikim. Output-i i veçantë Dummy nuk ekziston; vlerat vijnë nga raporti final. [Detaje](02_dummy_baseline/SUMMARY.md)

## 3. Word TF-IDF dhe Logistic Regression

**Pyetja/Problemi →** A mjaftojnë fjalët dhe çiftet e fjalëve për të dalluar lajmet real/fake?

**Eksperimenti →** Word TF-IDF `(1,2)`, `min_df=2`, maksimumi 30,000 features dhe Logistic Regression me class weights të balancuar.

**Rezultati →** Në 792 raste zyrtare modeli dha accuracy 0.8838 dhe F1 fake 0.8808. Krahasimi i verifikueshëm ruhet te [model_comparison.csv](03_word_tfidf_logreg/model_comparison.csv).

**Përfundimi →** Teksti përmban sinjal shumë më të fortë se Dummy, por word tokens nuk kapin çdo variant ortografik dhe modeli humb më shumë raste fake.

**Vendimi tjetër →** U studiuan karakteristikat e stilit për të parë nëse mund të plotësonin TF-IDF-in. [Detaje](03_word_tfidf_logreg/SUMMARY.md)

## 4. Analiza e linguistic features

**Pyetja/Problemi →** A ndryshojnë klasat në gjatësi, pikësim, kapitalizim, fraza sensacionale, burime dhe diakritika?

**Eksperimenti →** U nxorën features numerike dhe u krahasuan mesataret, Mann–Whitney U dhe Cohen's d.

**Rezultati →** U gjetën dallime të matshme. [feature_summary.csv](04_linguistic_features/feature_summary.csv) ruan mesataret e verifikueshme; output-i i plotë i testeve statistikore nuk ekziston aktualisht.

**Përfundimi →** Linguistic features kanë sinjal, por gjatësia dhe disa marker-a mund të jenë bias i corpus-it.

**Vendimi tjetër →** Features u testuan si model më vete dhe në kombinim me TF-IDF. [Detaje](04_linguistic_features/SUMMARY.md)

## 5. Modelet linguistic-only dhe hybrid

**Pyetja/Problemi →** A e rrit performancën bashkimi i sinjaleve gjuhësore me TF-IDF?

**Eksperimenti →** U krahasuan TF-IDF, linguistic-only, hybrid dhe hybrid pa features direkte të gjatësisë.

**Rezultati →** TF-IDF dha accuracy 0.8977; linguistic-only 0.8270; hybrid 0.8902; hybrid pa gjatësi 0.8914. Figura ekzistuese është [model_comparison.png](05_linguistic_hybrid_models/figures/model_comparison.png).

**Përfundimi →** Linguistic dhe hybrid nuk e përmirësuan modelin tekstual.

**Vendimi tjetër →** Linguistic features u mbajtën vetëm për analizë/UI; modeli kandidat mbeti tekstual dhe u analizuan probabilitetet e tij. [Detaje](05_linguistic_hybrid_models/SUMMARY.md)

## 6. Analiza fillestare e probabiliteteve

**Pyetja/Problemi →** A janë probabilitetet e baseline-it të përshtatshme për vendime të kujdesshme në UI?

**Eksperimenti →** U aplikua calibration group-safe dhe u matën Brier, log-loss, ECE, confidence dhe disa zona uncertain.

**Rezultati →** Calibration-i uli accuracy në këtë fazë, por prodhoi probabilitete të përdorshme. Zona 0.30–0.70 ishte varianti më konservator. Output-et tabelore historike nuk janë të disponueshme; figura ekzistuese ruhet [këtu](06_initial_probability_calibration/figures/probability_calibration.png).

**Përfundimi →** Një vendim binar në 0.50 fsheh pasigurinë dhe nuk mjafton për aplikacionin.

**Vendimi tjetër →** U testua kontrata e plotë probabilitet → prag → vendim. [Detaje](06_initial_probability_calibration/SUMMARY.md)

## 7. Kontrata e aplikacionit

**Pyetja/Problemi →** A japin preprocessing-u, probabilitetet, pragjet dhe UI-ja të njëjtin vendim në çdo rast?

**Eksperimenti →** U kontrolluan kufijtë, Unicode NFC/NFD, input-et problematike, paralajmërimet dhe paraqitja desktop/mobile.

**Rezultati →** Nuk u gjetën probabilitete ose vendime të pavlefshme. Vendimet e forta mbuluan 86.11% të test-it dhe kishin accuracy 93.99% për modelin e asaj faze.

**Përfundimi →** Kontrata me tre nivele ishte e përshtatshme, por paralajmërimi kundër interpretimit si fact-checker duhej të mbetej.

**Vendimi tjetër →** Sistemi u provua jashtë corpus-it. [Detaje](07_application_contract/SUMMARY.md)

## 8. External evaluation

**Pyetja/Problemi →** A përgjithësohet modeli në lajme me burime, periudha dhe stil tjetër?

**Eksperimenti →** U validua një benchmark me 40 raste të balancuara dhe u aplikua kontrata e modelit historik.

**Rezultati →** Baseline-i dha accuracy 0.4750, recall real 0.0500 dhe recall fake 0.9000. Tabela [external_evaluation.csv](08_external_evaluation/external_evaluation.csv) përmban edhe rezultatin e modelit final për krahasim.

**Përfundimi →** U identifikua domain shift i fortë dhe prirje e baseline-it drejt fake.

**Vendimi tjetër →** U analizua gjatësia dhe stabiliteti i prediction-it për të kuptuar një pjesë të kësaj rënieje. [Detaje](08_external_evaluation/SUMMARY.md)

## 9. Gjatësia dhe domain shift-i

**Pyetja/Problemi →** A është gjatësia shkaku kryesor i rënies external?

**Eksperimenti →** U analizuan grupet e gjatësisë, korrelacionet, shkurtimi/zgjerimi i kontrolluar dhe dallimet e domain-it.

**Rezultati →** Probabiliteti fake kishte lidhje negative me gjatësinë. Në modelin final, fake mbi 250 fjalë kishin recall 0.4483; detajet janë te [length_metrics.csv](09_length_domain_shift/length_metrics.csv).

**Përfundimi →** Gjatësia ndikon indirekt përmes TF-IDF, por nuk shpjegon e vetme domain shift-in.

**Vendimi tjetër →** U shtua Character TF-IDF për të kapur sinjale nën nivelin e fjalës. [Detaje](09_length_domain_shift/SUMMARY.md)

## 10. Word, Character dhe Word+Character TF-IDF

**Pyetja/Problemi →** Cila përfaqësim tekstual kap më mirë informacionin e corpus-it?

**Eksperimenti →** U krahasuan tre përfaqësime me classifier dhe evaluation të njëjtë; external nuk u përdor për zgjedhje.

**Rezultati →** Word+Character dha F1 weighted 0.9027 dhe lidhjen më të ulët me gjatësinë ndër kandidatët kryesorë. Zgjedhja verifikohet te [selection.json](10_tfidf_representations/selection.json).

**Përfundimi →** Kombinimi ruan kuptimin e fjalëve dhe shton sinjale ortografike/strukturore.

**Vendimi tjetër →** Përfaqësimi u ngri dhe u përdor për krahasimin e classifier-ave. [Detaje](10_tfidf_representations/SUMMARY.md)

## 11. Krahasimi i classifier-ave

**Pyetja/Problemi →** Cili algoritëm klasifikon më mirë të njëjtët vektorë Word+Character?

**Eksperimenti →** Logistic Regression, Linear SVM dhe ComplementNB u krahasuan në pesë folds group-safe vetëm mbi train.

**Rezultati →** Mean F1 weighted ishte 0.8952 për Logistic Regression, 0.9110 për Linear SVM dhe 0.8810 për ComplementNB. [selection.json](11_classifier_comparison/selection.json) dokumenton rregullin dhe auditimin e folds.

**Përfundimi →** Linear SVM ishte më i mirë dhe më i qëndrueshëm.

**Vendimi tjetër →** U krye tuning vetëm i parametrit `C` të Linear SVM. [Detaje](11_classifier_comparison/SUMMARY.md)

## 12. Tuning i Linear SVM

**Pyetja/Problemi →** Cila vlerë `C` jep kompromisin më të mirë performancë–regularizim?

**Eksperimenti →** U provuan `0.25, 0.5, 1.0, 2.0, 4.0` në të njëjtat folds dhe u matën F1, stabiliteti, recall gap dhe generalization gap.

**Rezultati →** `C=1.0` dha mean F1 0.9110 dhe devijim standard 0.0021. `C=4.0` ishte vetëm rreth 0.0013 më lart, por më pak stabil.

**Përfundimi →** Fitimi minimal nuk justifikoi kompleksitet dhe regularizim më të dobët.

**Vendimi tjetër →** `C=1.0` u ngri dhe decision scores e tij u kalibruan. [Detaje](12_svm_tuning/SUMMARY.md)

## 13. Sigmoid kundrejt isotonic

**Pyetja/Problemi →** Cila metodë i kthen scores e SVM në probabilitete më të besueshme?

**Eksperimenti →** U përdor nested group-safe OOF calibration për sigmoid dhe isotonic.

**Rezultati →** Sigmoid: Brier 0.06534, log-loss 0.21752, ECE 0.01486. Isotonic: Brier 0.06588, log-loss 0.24753, ECE 0.01379. Fold-et ruhen te [calibration_fold_metrics.csv](13_calibration_comparison/calibration_fold_metrics.csv).

**Përfundimi →** Sigmoid kishte cilësi të krahasueshme, log-loss më të mirë dhe rrezik më të ulët overfitting-u.

**Vendimi tjetër →** Probabilitetet sigmoid OOF u përdorën për zgjedhjen e pragjeve. [Detaje](13_calibration_comparison/SUMMARY.md)

## 14. Zgjedhja e thresholds

**Pyetja/Problemi →** Sa e gjerë duhet të jetë zona uncertain?

**Eksperimenti →** U krahasuan 0.30/0.70, 0.35/0.65 dhe 0.40/0.60 vetëm me OOF train predictions.

**Rezultati →** 0.30/0.70 dha strong accuracy 0.9488 me coverage 0.8858 dhe vendosi 132 gabime binare në zonën uncertain. [selection.json](14_threshold_selection/selection.json) ruan vendimin.

**Përfundimi →** Pragjet më konservatore u preferuan për cilësi më të lartë të vendimeve të forta.

**Vendimi tjetër →** Konfigurimi i plotë u ngri për vlerësim final. [Detaje](14_threshold_selection/SUMMARY.md)

## 15. Vlerësimi dhe ngrirja e modelit final

**Pyetja/Problemi →** A është kandidati i zgjedhur i saktë, determinist dhe identik me artefaktin që do të përdorë aplikacioni?

**Eksperimenti →** U kontrolluan konfigurimi, hash-et, preprocessing-u, probabilities, thresholds, prediction anchors, reload-i dhe group leakage. Kandidati u kopjua pa ritrajnim.

**Rezultati →** Në test-in e brendshëm: accuracy 0.9116, F1 fake 0.9098, Brier 0.0658 dhe log-loss 0.2192. Në external: accuracy 0.6000. Modeli ka SHA-256 `52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5`.

**Përfundimi →** Versioni `v1.0.0` është i ngrirë dhe i riprodhueshëm, por mbetet classifier gjuhësor me kufizime domain shift-i.

**Vendimi tjetër →** Modeli u lidh me `predict_final.py` dhe Streamlit; ndryshime të tjera nuk u bënë. [Detaje](15_final_model/SUMMARY.md)

![Krahasimi final](15_final_model/figures/model_comparison.png)

## 16. Interpretability

**Pyetja/Problemi →** Cilat n-grame kanë peshat globale më të forta drejt real ose fake?

**Eksperimenti →** Script-i i interpretimit lexon modelin final dhe lidh koeficientët e Linear SVM me feature names Word/Character.

**Rezultati →** Drejtimi i koeficientëve verifikohet në raportin final, por `top_linear_features.csv` dhe një figurë përkatëse nuk ekzistojnë aktualisht. Termat individualë mbeten të padisponueshëm.

**Përfundimi →** Koeficientët janë lidhje të mësuara nga corpus-i, jo shpjegim shkakësor ose provë faktike.

**Vendimi tjetër →** Analiza u mbajt vetëm për interpretim; modeli final nuk u ndryshua. [Detaje](16_interpretability/SUMMARY.md)

## Konfigurimi që doli nga historia

```text
Preprocessing: Unicode NFC + normalizim hapësirash
Përfaqësimi: Word TF-IDF (1,2) + Character char_wb TF-IDF (3,5)
Classifier: Linear SVM
C: 1.0
Calibration: sigmoid, 5 folds group-safe
Pragjet: 0.30 / 0.70
Artefakti: final_model/final_word_char_linear_svm_calibrated_v1.joblib
```
