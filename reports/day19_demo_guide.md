# Udhëzuesi i demonstrimit final

## Përgatitja

1. Aktivizo environment-in dhe nis aplikacionin:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m streamlit run app\streamlit_app.py
   ```

2. Hap `reports/day19_demo_cases.csv`. Për çdo rast kopjo kolonat `title` dhe
   `content` në input-et përkatëse të Streamlit.
3. Mbaj hapur `notebooks/02_final_walkthrough.ipynb` për të treguar rrjedhën
   dataset → preprocessing → TF-IDF → model → probabilitete.

CSV-ja përmban input-in e plotë. Ky dokument përdor titujt dhe ID-të që rastet
të gjenden pa paqartësi.

## Hyrja e prezantimit

Shpjego shkurt:

- modeli analizon ngjashmërinë gjuhësore me corpus-in real/fake;
- rezultati është probabilitet sipas modelit, jo e vërtetë absolute;
- modeli nuk kontrollon burimin dhe nuk bën fact-checking;
- pragjet janë `<30% likely_real`, `30%-70% uncertain`, `>70% likely_fake`.

## Demonstrimi kryesor

### 1. Likely real korrekt

- **Rreshti CSV:** `demo_order=1`, `article_id=true_1594`
- **Titulli:** Shqipëri, një i vdekur dhe 53 raste të reja me COVID-19
- **Label-i:** real
- **Rezultati i pritur:** `likely_real`
- **P(real):** 99.98999%
- **P(fake):** 0.01001%
- **Sinjale për t'u komentuar:** 169 fjalë, 0 pikëçuditëse, treguesit e burimit
  `sipas` dhe `ministria`, uppercase ratio 7.07%, diacritic ratio 7.34%.
- **Çfarë shpjegohet:** teksti ka stil raportues dhe referon institucione, por
  këto janë vetëm karakteristika të vëzhguara. Ato nuk provojnë faktet.

### 2. Likely fake korrekt

- **Rreshti CSV:** `demo_order=2`, `article_id=fake_531`
- **Titulli:** EKSKLUZIVE: Albin Kurti President i Kosoves?
- **Label-i:** fake
- **Rezultati i pritur:** `likely_fake`
- **P(real):** 0.00003%
- **P(fake):** 99.99997%
- **Sinjale për t'u komentuar:** 86 fjalë, markeri sensacional `ekskluzive`, pa
  tregues burimi nga lista, uppercase ratio 8.39%, diacritic ratio 0.38%.
- **Çfarë shpjegohet:** modeli gjeti sinjale që ngjajnë me shembujt fake të
  trajnimit. Edhe confidence i lartë nuk është verifikim faktik.

### 3. Uncertain

- **Rreshti CSV:** `demo_order=3`, `article_id=true_585`
- **Titulli:** Sejdiu i gatshëm të kandidojë për president, nëse dorëhiqet Thaçi
- **Label-i:** real
- **Rezultati i pritur:** `uncertain`
- **P(real):** 50.28133%
- **P(fake):** 49.71867%
- **Sinjale për t'u komentuar:** 152 fjalë, 0 pikëçuditëse, pa marker
  sensacional ose tregues burimi nga listat, uppercase ratio 7.04%, diacritic
  ratio 6.20%.
- **Çfarë shpjegohet:** probabiliteti fake është brenda zonës 30%-70%, prandaj
  modeli nuk ka siguri të mjaftueshme. `Uncertain` nuk do të thotë se lajmi
  është gjysmë i vërtetë dhe gjysmë fake.

## Shembujt opsionalë të kufizimeve

### False positive

- **Rreshti CSV:** `demo_order=4`, `article_id=true_586`
- **Titulli:** Menjëherë fshijeni nësë ju vjen ky mesazh në telefon
- **Label-i real:** real
- **Rezultati:** `likely_fake`, P(fake) 89.63651%
- **Interpretimi:** titulli paralajmërues dhe citimi i një pretendimi të rremë
  brenda një artikulli korrigjues mund ta ngatërrojnë modelin. Modeli nuk kupton
  automatikisht se artikulli po e përgënjeshtron pretendimin.

### False negative

- **Rreshti CSV:** `demo_order=5`, `article_id=fake_1104`
- **Titulli:** Nuk hapen xhamitë as në ditët e fundit të Ramazanit, edhe ne
  Hoxhallarët duam pushim
- **Label-i real:** fake
- **Rezultati:** `likely_real`, P(real) 89.40620%
- **Interpretimi:** teksti ka 378 fjalë, stil formal, citime dhe emra
  institucionesh. Ky rast ilustron bias-in e gjatësisë dhe faktin se një lajm
  fake mund të imitojë formën e lajmeve real.

## Mbyllja e demonstrimit

Përfundo me këto tri pika:

1. Modeli final është i dobishëm si sinjal gjuhësor dhe jo si arbitër i së
   vërtetës.
2. Zona `uncertain` ndalon një pjesë të vendimeve të forta kur probabiliteti
   nuk është bindës.
3. False positive dhe false negative tregojnë pse verifikimi me burime të
   jashtme mbetet i domosdoshëm.

Formulime që duhen përdorur: “sipas modelit”, “ngjashmëri gjuhësore”, “sinjale
të vëzhguara”. Shmang formulime si “modeli e verifikoi” ose “lajmi është me
siguri i vërtetë/fake”.
