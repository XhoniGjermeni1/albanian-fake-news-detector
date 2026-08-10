# Changelog

## [1.0.0] - 2026-08-10

Versioni klasik final i projektit.

### Përfshin

- Albanian Fake News Corpus si Git submodule;
- pipeline modular për dataset, preprocessing dhe linguistic features;
- model final Word + Character TF-IDF + Linear SVM `C=1.0`;
- sigmoid probability calibration dhe pragjet `0.30/0.70`;
- API-në `predict_final_news()` dhe aplikacionin Streamlit;
- modelin final, manifestin dhe regression checks;
- walkthrough notebook dhe materialet e demonstrimit;
- vlerësim të brendshëm, benchmark të jashtëm pilot dhe analizë kufizimesh;
- 115 teste dhe verifikim në environment të ri Python 3.11.

### Kufizime të njohura

- sistemi klasifikon sinjale gjuhësore dhe nuk bën fact-checking;
- mbeten bias-i i gjatësisë, domain shift-i dhe paqëndrueshmëria e teksteve
  shumë të shkurtra.

XLM-RoBERTa, SHAP dhe deploy online mbeten jashtë versionit klasik `v1.0.0`.
