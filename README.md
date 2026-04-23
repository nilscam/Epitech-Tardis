# TARDIS — *Predicting the Unpredictable*

Projet Epitech d'analyse et de prédiction des retards ferroviaires SNCF.
Pipeline complet : exploration de données → modélisation prédictive → dashboard interactif.

---

## Aperçu

TARDIS analyse un dataset d'agrégations **mensuelles** des trajets SNCF
(gare de départ → gare d'arrivée) entre 2018 et 2025 et fournit :

1. Un **notebook d'exploration** (`tardis_eda.ipynb`) — nettoyage, feature engineering,
   statistiques et visualisations.
2. Un **notebook de modélisation** (`tardis_model.ipynb`) — comparaison de 3 modèles
   de régression, tuning d'hyperparamètres et sauvegarde d'un modèle réutilisable.
3. Un **dashboard Streamlit** (`tardis_dashboard.py`) — KPIs, visualisations
   interactives avec filtres et interface de prédiction.

**Cible du modèle :** `Average delay of all trains at arrival` (retard moyen
à l'arrivée en minutes, valeur continue).

---

## Prérequis

- Python **3.10+** (testé sur 3.14)
- `dataset.csv` à la racine du projet

---

## Installation

```bash
# 1. Créer et activer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

Le pipeline doit être exécuté **dans l'ordre** : l'EDA produit `cleaned_dataset.csv`,
le modèle produit `model.joblib`, puis le dashboard consomme les deux.

### 1. Nettoyage & exploration

```bash
jupyter nbconvert --to notebook --execute --inplace tardis_eda.ipynb
```

Ou ouvrir le notebook dans Jupyter Lab pour l'exécuter cellule par cellule :

```bash
jupyter lab tardis_eda.ipynb
```

**Sortie :** `cleaned_dataset.csv` (~11 000 lignes, 35 colonnes).

### 2. Entraînement du modèle

```bash
jupyter nbconvert --to notebook --execute --inplace tardis_model.ipynb
```

**Sortie :** `model.joblib` contenant le pipeline scikit-learn complet,
les noms de features, les métriques et les meilleurs hyperparamètres.

### 3. Dashboard

```bash
streamlit run tardis_dashboard.py
```

Le dashboard s'ouvre sur <http://localhost:8501>.

---

## Contenu du dashboard

| Onglet | Fonctionnalités |
|---|---|
| **📈 Distribution** | Histogramme + KDE des retards, répartition par catégorie, boxplot par service |
| **🚉 Par gare** | Top gares les plus en retard / plus ponctuelles, tableau détaillé, export CSV |
| **📅 Tendances** | Série temporelle mensuelle, comparaison mois / année, heatmap année × mois |
| **🔗 Corrélations** | Matrice de corrélation + contribution moyenne des causes de retard |
| **🔮 Prédiction** | Formulaire de saisie, prédiction + intervalle de confiance, scénario alternatif |

**Filtres sidebar** : type de service, plage de dates, gares de départ et d'arrivée.
Tous les graphes et KPIs se mettent à jour dynamiquement.

---

## Structure des livrables

```
tardis/
├── requirements.txt              # Dépendances Python
├── dataset.csv                   # Dataset brut (entrée)
├── tardis_eda.ipynb              # Étape 1-2 : cleaning + EDA
├── cleaned_dataset.csv           # Sortie de l'étape 1
├── tardis_model.ipynb            # Étape 3 : modélisation
├── model.joblib                  # Modèle entraîné (sortie étape 3)
├── tardis_dashboard.py           # Étape 4 : dashboard Streamlit
└── README.md
```

---

## Décisions techniques

- **Parsing tolérant** du CSV : le dataset source mélange séparateurs décimaux
  (`.` et `,`), unités parasites (`6.9 min`), espaces dans les chiffres et
  deux formats de date (`YYYY-MM`, `YYYY/MM`). Un parser dédié normalise tout
  cela dans l'EDA.
- **Winsorisation** de la cible à `[-30 ; 120]` minutes pour neutraliser les
  outliers aberrants sans perdre d'information.
- **Anti-fuite de cible** : le modèle utilise uniquement des features disponibles
  *avant le départ* (gares, service, mois, année, temps de parcours, nombre de
  trains planifiés). Les variables fortement corrélées à la cible (retards au
  départ, nombre de trains en retard, causes) sont **exclues** — sinon le R²
  serait artificiellement ~1.
- **Modèle retenu** : **Gradient Boosting Regressor** tuné par `GridSearchCV`.
  Il bat la baseline (prédire la moyenne) d'environ 18 % sur le RMSE de test,
  ce qui constitue un signal utile compte tenu de l'absence d'informations
  granulaires (heure, jour de la semaine) dans le dataset source.
- **OneHotEncoder(handle_unknown="ignore")** : le pipeline encode à zéro toute
  gare jamais vue à l'entraînement — robustesse à l'inférence.

---

## Limites connues

- **Granularité mensuelle** : impossible de prédire le retard d'un trajet précis
  (pas d'heure ni de jour de la semaine dans le dataset).
- **Chocs exogènes** (grèves, canicules, pannes majeures) non modélisables
  sans features externes.
- Les retards sont **très hétéroscédastiques** (forte variance conditionnelle) ;
  l'intervalle de confiance à ± 1,96 × RMSE est une approximation.

---

## Formatage du code

Le code suit le style **ruff formatter** :

```bash
pip install ruff
ruff format .
```
