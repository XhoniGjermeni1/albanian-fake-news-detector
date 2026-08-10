# Dita 20 - Mbyllja teknike e projektit klasik

## Vendimi final

Projekti klasik mund të konsiderohet **100% i përfunduar** për dorëzim,
demonstrim dhe mësim. Modeli nuk u ndryshua. Versioni final është `v1.0.0`.

## Çfarë u pastrua

- Runtime-i final nuk importon më helper-a nga moduli historik `predict.py`.
  Pragjet dhe shpjegimi gjuhësor u vendosën te `prediction_utils.py`; API-të e
  vjetra vazhdojnë të funksionojnë për raportet dhe testet historike.
- U korrigjuan docstring-et që e quanin preprocessing-un vetëm “baseline” dhe
  prediction-in e vjetër “future app”.
- U kontrolluan importet e skedarëve kryesorë; nuk u gjetën importe të
  papërdorura.
- `plotly` dhe `statsmodels` u hoqën nga requirements sepse nuk importoheshin
  nga asnjë modul, notebook ose test.
- Notebook-et morën cell IDs të vlefshme për versionet moderne të Jupyter-it.
- Cache-t dhe output-et e përkohshme janë të mbuluara nga `.gitignore`.

Asnjë funksion trajnimi, konfigurim TF-IDF, classifier, calibration ose prag nuk
u ndryshua.

## Çfarë u organizua

Path-et e skripteve historike nuk u lëvizën, sepse përdoren nga testet dhe
raportet e diplomës. Në vend të një lëvizjeje me rrezik, u shtuan:

- `src/models/README.md`: ndarja runtime/finalization/historical analysis;
- `models/README.md`: artefaktet finale dhe modelet lokale eksperimentale;
- `reports/README.md`: roli i raporteve dhe pikat kryesore finale;
- `.gitmodules`: mapping-u që mungonte për Albanian Fake News Corpus;
- `CHANGELOG.md`: përmbledhja e release-it klasik.

Corpus-i ishte Git gitlink pa `.gitmodules`, gjë që do të thyente clone-in e
pastër. Tani `git clone --recurse-submodules` merr commit-in e ngrirë
`e6ee2695f7bba964a1d23abc515791bbb18e65d0` pa ndryshuar skedarët raw.

## Modelet

Artefakti final:

```text
models/final_word_char_linear_svm_calibrated_v1.joblib
```

SHA-256:

```text
52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5
```

Hash-i përputhet me kandidatin e Ditës 16 dhe me manifestin. Manifesti
konfirmon Word + Character TF-IDF, Linear SVM `C=1.0`, sigmoid calibration dhe
pragjet `0.30/0.70`.

Modelet eksperimentale `*.joblib` mbeten lokalisht për analiza historike, por
injorohen nga Git. Përjashtimi i vetëm është artefakti final 1.35 MB. Modeli i
vjetër Logistic Regression nuk përdoret nga Streamlit ose `predict_final.py`.

## README dhe dependencies

README u reduktua nga një ditar i gjatë në një udhëzues teknik prej 227
rreshtash. Ai përmban qëllimin, dataset-in, pipeline-in, modelin, metrikat,
kufizimet, strukturën, instalimin, Streamlit, notebook-un, testet dhe warning-un
e fact-checking-ut.

`requirements.txt` përdor versione të fiksuara dhe përmban vetëm stack-un e
nevojshëm për kodin, analizat e ruajtura, Streamlit, testet dhe notebook-et.

## Environment-i i pastër

U krijua `.venv` e re me Python 3.11.9. Instalimi u bë vetëm nga
`requirements.txt` dhe `pip check` raportoi `No broken requirements found`.

U verifikuan nga kjo `.venv`:

- importet dhe versionet e paketave;
- ngarkimi i modelit final dhe klasat `[0, 1]`;
- `predict_final_news()`;
- 6 rastet e ngrira të Ditës 17;
- notebook-u final me Jupyter/nbconvert;
- Streamlit në `http://127.0.0.1:8503` me HTTP 200;
- `compileall` dhe gjithë test suite-i.

Notebook-u ekzekutoi 10/10 qeliza kodi me 0 error dhe pa warning për cell IDs.

## Testet dhe regression checks

Rezultati final nga `.venv`:

```text
115 passed, 0 failed
```

Testet u ndryshuan vetëm për të shtuar kontrata sigurie:

- runtime-i final nuk importon modulin historik të modelit;
- submodule-i i dataset-it është deklaruar;
- `.gitignore` përfshin vetëm modelin final;
- README dhe requirements ruajnë komandat/versionet finale.

Gjashtë rastet e Ditës 17 dhanë 6/6 vendime identike. Diferenca maksimale e
probabilitetit ishte `1.11e-16`, vetëm precision i serializimit numerik të CSV,
jo ndryshim i modelit. Tabela ruhet te `reports/day20_regression_checks.csv`.

## Skedarët për mbrojtje

Për rrjedhën kryesore mjafton të mësohen këta skedarë:

1. `src/data/load_dataset.py`: leximi dhe label-et e corpus-it.
2. `src/preprocessing/clean_text.py`: NFC, hapësirat dhe bashkimi title/content.
3. `src/features/linguistic_features.py`: sinjalet e shpjegimit.
4. `src/models/prediction_utils.py`: pragjet dhe vendimi me tri nivele.
5. `src/models/predict_final.py`: modeli final, probabilitetet dhe output-i.
6. `app/streamlit_app.py`: validimi, cache, UI dhe shfaqja e rezultatit.
7. `tests/test_final_model.py` dhe `tests/test_streamlit_app.py`: regresioni.
8. `notebooks/02_final_walkthrough.ipynb`: demonstrimi end-to-end.

Skriptet e tjera të modeleve tregojnë historinë eksperimentale dhe nuk janë
prioritet për mësimin 2–3 ditor të mbrojtjes.

## Zgjerime opsionale

Pas `v1.0.0` mund të studiohen XLM-RoBERTa, SHAP, benchmark më i madh i jashtëm
dhe deploy online. Ato janë projekte pasuese dhe jo punë e pambyllur e versionit
klasik.
