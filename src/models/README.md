# Active Model Code

Kjo dosje përmban vetëm kontratën e modelit final:

- `builders.py`: konfigurimi i ngrirë Word TF-IDF + Character TF-IDF +
  `LinearSVC(C=1.0)`;
- `model_contract.py`: kontrolli i konfigurimit të modelit, preprocessing-ut
  dhe SHA-256;
- `prediction_utils.py`: thresholds `0.30/0.70` dhe shpjegimet gjuhësore;
- `predict_final.py`: ngarkimi i artefaktit dhe API-ja finale e prediction-it.

Calibration-i i përzgjedhur është i ruajtur brenda artefaktit final. Kodi që
e krahasoi, trajnoi dhe ngriu historikisht modelin ndodhet te
`archive/experiments/models/` dhe nuk importohet nga runtime-i.
