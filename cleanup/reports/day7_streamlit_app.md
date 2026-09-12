# Dita 7 - Aplikacioni minimal Streamlit

## Qëllimi

Qëllimi i Ditës 7 ishte lidhja e modelit të kalibruar të Ditës 6 me një ndërfaqe të thjeshtë Streamlit. Aplikacioni lejon vendosjen e titullit dhe përmbajtjes së një lajmi në shqip dhe shfaq një vlerësim të modelit, probabilitetet dhe karakteristikat gjuhësore të vëzhguara.

## Kontrolli i funksionit të parashikimit

Funksioni `predict_news_for_app(title, content)` ekzistonte dhe punonte me modelin `models/calibrated_tfidf_logreg.joblib`. Ai kthen:

- `decision`: `likely_real`, `uncertain` ose `likely_fake`;
- probabilitetet për real dhe fake;
- pragjet 0.30 dhe 0.70;
- linguistic explanation;
- paralajmërimin se rezultati nuk është fact-checking.

Funksionit iu shtua mundësia për të marrë një model tashmë të ngarkuar. Kjo lejon që Streamlit ta ruajë modelin në cache dhe të mos e lexojë përsëri nga disku pas çdo klikimi. Në shpjegim u përfshinë edhe shprehjet e pasigurisë të gjetura.

## Aplikacioni

U krijua `app/streamlit_app.py` me:

- input për titullin;
- input për përmbajtjen;
- butonin `Analizo lajmin`;
- mesazh të kuptueshëm për vendimin dhe emrin teknik të tij;
- probabilitetet real/fake në përqindje;
- numrin e fjalëve, gjatësinë, pikëçuditëset, uppercase ratio dhe diacritic ratio;
- fjalët sensacionale, treguesit e burimit dhe shprehjet e pasigurisë;
- seksionin `Rreth modelit` dhe tri shembuj testimi;
- paralajmërim të dukshëm se modeli nuk verifikon fakte.

Modeli ngarkohet me `st.cache_resource`, prandaj ruhet në memorie gjatë përdorimit të aplikacionit.

## Validimi i input-it

- Titull dhe përmbajtje bosh: analiza bllokohet me mesazh gabimi.
- Vetëm titull: analiza lejohet, por shfaqet paralajmërim për mungesën e përmbajtjes.
- Përmbajtje me më pak se 20 fjalë: analiza lejohet me paralajmërim.
- Vetëm përmbajtje: analiza lejohet.
- Gabimet gjatë ngarkimit të modelit ose parashikimit trajtohen pa rrëzuar aplikacionin.

## Provat e parashikimit

Funksioni u provua me tekst normal, tekst shumë të shkurtër, tekst sensacional dhe tekst me tregues burimi. Të gjitha rastet kthyen strukturën e plotë pa gabim. Në provat e shkurtra të shkruara pa `ë/ç`, modeli dha probabilitet të lartë për fake; kjo tregon ndjeshmëri ndaj stilit, gjatësisë dhe ngjashmërisë me corpus-in.

Tri shembujt e aplikacionit mbulojnë të gjitha zonat e vendimit:

| Shembulli | Vendimi | P(real) | P(fake) |
|---|---:|---:|---:|
| Raport institucional | `likely_real` | 87.99% | 12.01% |
| Njoftim i shkurtër | `uncertain` | 35.08% | 64.92% |
| Titull sensacional | `likely_fake` | 1.13% | 98.87% |

Këto rezultate shërbejnë vetëm për testimin e rrjedhës së aplikacionit dhe nuk janë shembuj fact-checking.

## Testet

U shtuan teste për:

- input bosh;
- vetëm titull;
- përmbajtje të shkurtër;
- vetëm përmbajtje;
- hapjen e ndërfaqes;
- shfaqjen e paralajmërimeve;
- rezultatin, probabilitetet dhe shpjegimin;
- të tria vendimet e modelit.

Aplikacioni u nis lokalisht dhe endpoint-i i kontrollit ktheu `HTTP 200 / ok`. Ndërfaqja u kontrollua në pamje desktop dhe mobile; sidebar-i përdor gjendjen `auto`, kështu që në ekran të vogël formulari mbetet i dukshëm dhe pa tejkalim horizontal.

Pamjet e kontrollit ruhen te:

```text
reports/figures/day7_app_desktop.png
reports/figures/day7_app_mobile.png
```

Rezultati përfundimtar:

```text
33 passed
```

## Ekzekutimi

Nga rrënja e projektit:

```powershell
streamlit run app\streamlit_app.py
```

## Kufizimet

- Modeli analizon modele statistikore të tekstit dhe nuk verifikon pretendime ose burime.
- Teksti shumë i shkurtër mund të japë probabilitete të forta, edhe kur informacioni është i pamjaftueshëm.
- Dataseti është relativisht i vogël dhe mund të përmbajë stile ose burime karakteristike.
- Probabiliteti i kalibruar nuk është garanci që një lajm është real ose fake.
- Linguistic features paraqiten për shpjegim dhe nuk duhet të lexohen si prova absolute.

## Dita 8

Hapi i rekomanduar është testimi me një grup më të gjerë lajmesh të reja dhe shembujsh jashtë datasetit, dokumentimi i rasteve ku aplikacioni gabon dhe përmirësimi i paraqitjes vetëm aty ku testimi tregon nevojë. Para çdo publikimi duhet të ruhet qartë ndarja mes klasifikimit gjuhësor dhe verifikimit faktik.
