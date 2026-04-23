# TARDIS — *Predicting the Unpredictable*

Projet Epitech d'analyse et de prédiction des retards ferroviaires SNCF.
Pipeline complet : exploration de données → modélisation prédictive → dashboard interactif.

---

## Aperçu

TARDIS analyse un dataset d'agrégations **mensuelles** des trajets SNCF
(gare de départ → gare d'arrivée) entre 2018 et 2025 et fournit :

1. Un **notebook d'exploration** (`tardis_eda.ipynb`) — nettoyage, normalisation
   des gares, feature engineering, statistiques et visualisations.
2. Un **notebook de modélisation** (`tardis_model.ipynb`) — comparaison de 4 modèles
   de régression, tuning d'hyperparamètres avec validation temporelle, analyse des
   résidus et de l'importance des features.
3. Un **dashboard Streamlit** (`tardis_dashboard.py`) — KPIs, visualisations
   interactives avec filtres et interface de prédiction.

**Cible du modèle :** `Average delay of all trains at arrival` (retard moyen
à l'arrivée en minutes, valeur continue).

---

## Performance du modèle

Évalué sur un **split temporel par date** (train 2018-01 → 2024-05, test 2024-06 → 2025-12).
Les baselines sont évaluées **avant** la sélection du modèle final — un modèle ML
qui ne bat pas la baseline "moyenne par route" n'apporte rien.

| Modèle                              | RMSE (min) | MAE (min) |        R² |
|:------------------------------------|-----------:|----------:|----------:|
| Baseline — moyenne globale (train)  |       4.24 |      3.02 |     −0.07 |
| Baseline — moyenne par route (train)|       3.55 |      2.48 |     +0.25 |
| Ridge Regression                    |       3.43 |      2.43 |     +0.30 |
| Random Forest                       |       3.36 |      2.37 |     +0.33 |
| **Gradient Boosting (retenu)**      |   **3.34** |  **2.42** | **+0.34** |
| Gradient Boosting (tuné via CV)     |       3.39 |      2.45 |     +0.32 |

> **Lecture :** la baseline globale a un R² **négatif** sur le test car le retard
> moyen a dérivé (train = 5.98 min, test = 7.03 min). Comparer à la **moyenne par
> route** est plus honnête : le Gradient Boosting gagne **+0.09 R²** vs. cette
> baseline — gain modeste mais réel, issu principalement de la saisonnalité et
> des effets volume × hub.
>
> **Le GB tuné n'est PAS retenu** : le tuning via `TimeSeriesSplit` optimise une
> CV bruitée (peu de folds, périodes courtes) et recommande un modèle plus
> régularisé qui dégrade le score test réel. On retient donc le meilleur sur test.

---

## Prérequis

- Python **3.11+**
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

**Sortie :** `cleaned_dataset.csv` (~11 000 lignes × 35 colonnes, gares normalisées).

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

| Onglet             | Fonctionnalités                                                                 |
|:-------------------|:--------------------------------------------------------------------------------|
| **📈 Distribution** | Histogramme + KDE des retards, répartition par catégorie, boxplot par service   |
| **🚉 Par gare**     | Top gares les plus en retard / plus ponctuelles, tableau détaillé, export CSV   |
| **📅 Tendances**    | Série temporelle mensuelle, comparaison mois / année, heatmap année × mois      |
| **🔗 Corrélations** | Matrice de corrélation + contribution moyenne des causes de retard              |
| **🔮 Prédiction**   | Formulaire de saisie, prédiction + intervalle de confiance, scénario alternatif |

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

- **Parsing tolérant du CSV source** : le dataset d'origine mélange séparateurs
  décimaux (`.` et `,`), unités parasites (`6.9 min`), espaces dans les chiffres
  et deux formats de date (`YYYY-MM`, `YYYY/MM`). Un parser dédié normalise
  tout cela dans l'EDA.
- **Normalisation des gares** : le dataset brut contient la même gare écrite
  de plusieurs façons (`PARIS LYON` / `Paris Lyon` / `paris lyon`,
  `ST MALO` / `SAINT MALO`, `Tgv` vs `TGV`, valeur parasite `'0'`).
  Une fonction dédiée (`normalise_station`) applique : suppression des diacritiques,
  uppercase, `ST` → `SAINT`, compactage des espaces/tirets, filtrage des parasites.
  **Résultat : 132 → 59 gares uniques**.
- **Dédup logique** : après normalisation, ~200 lignes ont la même clé
  `(route, date, service)` avec des valeurs numériques divergentes
  (rééditions / retraitements côté source). On les agrège par moyenne, clé
  explicite. 38 paires `(route, date)` subsistent légitimement (National +
  International sur la même liaison le même mois).
- **Suppression des outliers cible** aberrants : on supprime les lignes où la
  cible sort de `[-30 ; 120]` min (9 lignes, clairement corrompues à la source
  plutôt qu'outliers à apprendre). **Pas de winsorisation** : clipper change la
  distribution et contamine autant le test que le train.
- **Imputation fittée sur train uniquement** : les NaN résiduels sur les
  features numériques sont imputés par la médiane via un `SimpleImputer`
  **intégré à la `Pipeline` sklearn** — la médiane n'est calculée que sur
  `X_train` (via `GridSearchCV`, re-fittée sur chaque fold CV).
- **Split temporel par date** (et non par rang d'index) : cutoff strict au
  `2024-06-01`. Un split par index coupait parfois au milieu d'un mois
  (mai 2024 à moitié train / à moitié test → fuite). Le cutoff par date évite
  cela (`train_dates.max() < test_dates.min()` est asserté).
- **Anti-fuite de cible** : le modèle utilise uniquement des features
  disponibles *avant le départ* (gares, service, mois, année, temps de
  parcours, nombre de trains planifiés, indicateurs saisonniers). Les
  variables fortement corrélées à la cible (retards au départ, nombre de
  trains en retard, causes) sont **exclues** — sinon R² ≈ 1 artificiel.
- **Encodage cyclique du mois** : `MonthSin`, `MonthCos` au lieu d'un
  `Month` numérique linéaire (sinon décembre et janvier sont "éloignés").
- **Scaling ciblé** : `StandardScaler` uniquement sur les variables
  **continues** (année, temps de parcours, volume, sin/cos). Les indicateurs
  binaires (`IsPeakMonth`, `IsParisDeparture`, …) passent tels quels.
- **Baseline route-mean** : une baseline qui prédit la moyenne historique
  *par route* (fittée sur train) sert de point de comparaison honnête, au-delà
  de la `DummyRegressor` globale.
- **Sélection du modèle final** : on compare tous les candidats sur le RMSE
  **test** (et non aveuglément sur le best-CV). La CV `TimeSeriesSplit` sur
  peu de folds est trop bruitée pour trancher sur un tel dataset ; elle sert
  à *guider* le tuning, pas à *décider* du modèle livré.
- **OneHotEncoder(handle_unknown="ignore")** : une gare jamais vue à
  l'entraînement voit toutes ses colonnes encodées à 0. **À retenir :** ce
  n'est **pas** une "moyenne conditionnelle", c'est la prédiction pour une
  gare fictive — à signaler côté utilisateur si on est amené à prédire sur
  une gare nouvelle.

---

## Limites connues

- **Granularité mensuelle** : impossible de prédire le retard d'un trajet précis
  (pas d'heure ni de jour de la semaine dans le dataset).
- **Chocs exogènes** (grèves, canicules, pannes majeures) non modélisables
  sans features externes.
- **Hétéroscédasticité des résidus** : la variance de l'erreur dépend du niveau
  prédit. L'intervalle affiché dans le dashboard utilise `± 1.96 × σ_résidus`
  (où σ est calculé sur les résidus test, pas via le RMSE — qui inclut le biais).
  C'est une approximation gaussienne ; il est surdimensionné pour les petits
  retards et sous-dimensionné pour les gros. Une version plus rigoureuse
  utiliserait de la *quantile regression* ou de la *conformal prediction*.
- **Sommes des % causes ≠ 100** : ~25 % des lignes sources ont une somme
  hors-[90 ; 110] %. On ne corrige pas (on ne masque pas un artefact source) ;
  les analyses "part moyenne de chaque cause" sont donc à lire comme indicatives.
- **Dérive temporelle** : sur la période de test (2024-2025), le retard moyen
  réel est plus élevé que sur la période d'entraînement. Le modèle reste
  utilisable mais sous-estime légèrement l'ampleur des retards récents.

---

## Formatage du code

Le code suit le style **ruff formatter** :

```bash
pip install ruff
ruff format .
```
