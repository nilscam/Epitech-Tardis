⏺ Tous les livrables sont en place :

  ┌─────────────────────┬────────┬──────────────────────────────────────────────────────────────┐
  │       Fichier       │ Taille │                           Contenu                            │
  ├─────────────────────┼────────┼──────────────────────────────────────────────────────────────┤
  │ requirements.txt    │ 143 B  │ Dépendances (pandas, sklearn, streamlit, joblib…)            │
  ├─────────────────────┼────────┼──────────────────────────────────────────────────────────────┤
  │ tardis_eda.ipynb    │ 974 KB │ Nettoyage + feature engineering + EDA (exécuté)              │
  ├─────────────────────┼────────┼──────────────────────────────────────────────────────────────┤
  │ cleaned_dataset.csv │ 3.5 MB │ 11 076 lignes × 35 colonnes                                  │
  ├─────────────────────┼────────┼──────────────────────────────────────────────────────────────┤
  │ tardis_model.ipynb  │ 218 KB │ 4 modèles comparés, tuning GridSearchCV, résidus, importance │
  ├─────────────────────┼────────┼──────────────────────────────────────────────────────────────┤
  │ model.joblib        │ 1.4 MB │ Gradient Boosting tuné (RMSE=3.58, MAE=1.99, R²=0.29)        │
  ├─────────────────────┼────────┼──────────────────────────────────────────────────────────────┤
  │ tardis_dashboard.py │ 17 KB  │ App Streamlit 5 onglets + filtres + prédiction               │
  ├─────────────────────┼────────┼──────────────────────────────────────────────────────────────┤
  │ README.md           │ 5 KB   │ Install, usage, décisions techniques                         │
  └─────────────────────┴────────┴──────────────────────────────────────────────────────────────┘

  Points clés :
  - Dataset brut contenait 2 formats de date, séparateurs décimaux mixtes (, et .), unités parasites — parser tolérant dans l'EDA.
  - Cible winsorisée à [-30 ; 120] min, 238 lignes sans cible supprimées, doublons retirés.
  - Features du modèle restreintes à celles connues avant le départ (gares, service, mois, année, temps de parcours, nombre de trains planifiés)
   → pas de fuite de cible. Gain de ~18 % sur RMSE vs baseline "moyenne".
  - Dashboard vérifié en local (HTTP 200, pas d'erreurs). Lance streamlit run tardis_dashboard.py après les 2 notebooks.

  Pour lancer :
  source .venv/bin/activate
  jupyter nbconvert --to notebook --execute --inplace tardis_eda.ipynb
  jupyter nbconvert --to notebook --execute --inplace tardis_model.ipynb
  streamlit run tardis_dashboard.py
