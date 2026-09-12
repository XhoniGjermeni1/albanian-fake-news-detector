# Dita 19 - Përdorshmëria, demonstrimi dhe walkthrough-u

## Përfundimi

Projekti është gati për demonstrim. Modeli final nuk u ritrajnua dhe nuk u
ndryshua. Hash-i SHA-256 mbeti
`52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5`, identik
me kandidatin e ngrirë.

## Kontrolli i UI-së

U verifikuan forma e input-it, probabilitetet Real/Fake, të tri vendimet,
shpjegimi `uncertain`, linguistic explanation, warning-u i fact-checking-ut,
seksioni “Rreth modelit” dhe input-et e shkurtra/të gjata.

- Desktop: 1440 x 1100 px, pa elemente të prera.
- Mobile: viewport real 390 x 844 px; `document.scrollWidth=390`, pa overflow.
- Streamlit health check: HTTP 200.
- Teksti i “Rreth modelit” u bë më profesional: “korpus shqiptar me etiketa”
  dhe “Kalibrim sigmoid”.
- Nuk u gjet nevojë për ndryshim layout-i ose logjike.

Pamjet ruhen te:

```text
reports/figures/day19_streamlit_desktop.png
reports/figures/day19_streamlit_mobile.png
```

## Demonstrimi

U përgatitën pesë raste me input të plotë:

| Roli | Article ID | Label | Vendimi | P(fake) |
|---|---|---|---|---:|
| likely real korrekt | `true_1594` | real | `likely_real` | 0.0100% |
| likely fake korrekt | `fake_531` | fake | `likely_fake` | 99.99997% |
| uncertain | `true_585` | real | `uncertain` | 49.7187% |
| false positive | `true_586` | real | `likely_fake` | 89.6365% |
| false negative | `fake_1104` | fake | `likely_real` | 10.5938% |

Tre rastet e para formojnë demonstrimin kryesor. Dy rastet e fundit shpjegojnë
bias-in e gjatësisë, stilin formal, citimin e pretendimeve dhe kufirin mes
klasifikimit gjuhësor dhe fact-checking-ut.

Materialet ruhen te:

```text
reports/day19_demo_cases.csv
reports/day19_demo_guide.md
```

## Ekzekutimi nga kompjuter i pastër

README tani dokumenton Python 3.11, krijimin e `.venv`, instalimin e versioneve
të fiksuara, nisjen e Streamlit, hapjen e JupyterLab dhe ekzekutimin e testeve.
`pip --dry-run` zgjidhi të gjitha dependency-t pa konflikt.

Komandat kryesore janë:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app\streamlit_app.py
python -m jupyter lab notebooks\02_final_walkthrough.ipynb
python -m pytest -q
```

`.gitignore` vazhdon të injorojë modelet eksperimentale, por lejon artefaktin
final 1.35 MB që ai të mund të përfshihet në repository bashkë me manifestin.

## Notebook-u final

`notebooks/02_final_walkthrough.ipynb` ka 21 qeliza, 10 prej tyre me kod. Ai:

- ngarkon `articles.parquet` të prodhuar nga pipeline-i i dataset-it;
- kontrollon klasat dhe tregon preprocessing-un NFC;
- inspekton Word + Character TF-IDF pa bërë `fit`;
- nxjerr linguistic features për shpjegim;
- ngarkon modelin final dhe riprodhon rastet e Ditës 17;
- shpjegon probabilitetet dhe zonën `uncertain`;
- lexon metrikat finale dhe vizaton confusion matrix;
- krahason testin e brendshëm me vlerësimin e jashtëm pilot;
- interpreton një false negative.

Të 10 qelizat u ekzekutuan me sukses në rreth 4 sekonda. Notebook-u nuk trajnon
ose ruan model të ri.

## Testet

- Testet e fokusuara për modelin, app-in dhe walkthrough-un: **38 passed**.
- Test suite-i i plotë: **112 passed**, 0 failed.
- Streamlit nuk ka referenca runtime te modeli i vjetër.

## Dita 20

Dita 20 duhet të kufizohet te pastrimi teknik: rishikimi i skedarëve të
gjeneruar, organizimi i output-eve që do të futen në Git, README-ja finale,
kontrolli nga një `.venv` i ri dhe përgatitja e commit/release-it përfundimtar.
Modeli dhe rezultatet zyrtare duhet të mbeten të ngrira.
