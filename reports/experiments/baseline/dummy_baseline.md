# Trivial Dummy Baseline

Ky eksperiment përdor `DummyClassifier(strategy="most_frequent")` mbi të
njëjtin train set dhe test set të brendshëm prej 792 artikujsh. Shtatë kopjet
ekzakte train-test përjashtohen me të njëjtën logjikë si vlerësimi final.

| Modeli | Accuracy | F1 weighted | F1 fake |
|---|---:|---:|---:|
| Dummy most-frequent | 0.5038 | 0.3376 | 0.0000 |
| Word TF-IDF + Logistic Regression | 0.8838 | 0.8838 | 0.8808 |
| Word + Character TF-IDF + Linear SVM | 0.9116 | 0.9116 | 0.9098 |

Dummy baseline parashikon vetëm klasën shumicë dhe nuk është kandidat për
aplikacionin. Ai tregon se modelet reale mësojnë sinjal përtej shpërndarjes së
klasave.
