# 🚆 TARDIS — Synthèse d'analyse

> Analyse prédictive des retards ferroviaires SNCF
> Dataset : agrégations mensuelles par liaison (gare → gare), 2018-01 → 2025-12

---

## 1. Contexte et objectif

Prédire le **retard moyen à l'arrivée** d'un trajet SNCF (en minutes) à partir de caractéristiques connues **avant le départ**, et offrir aux voyageurs un dashboard d'exploration des données historiques.

**Granularité du dataset** : chaque ligne = un couple `(gare de départ, gare d'arrivée, mois)` — 11 038 trajets-mois après nettoyage. **Limitation structurelle** : pas de granularité horaire ni journalière → on ne prédit pas un train précis mais le retard moyen attendu pour la liaison sur le mois considéré.

---

## 2. Qualité des données brutes

Le dataset source était **très désordonné** et cachait 3 types de bruit :

| Problème                      | Exemple                                                              | Impact                                   |
|:------------------------------|:---------------------------------------------------------------------|:-----------------------------------------|
| Séparateurs décimaux mélangés | `5,04` et `5.04` dans la même colonne                                | Toutes les numériques lues en texte      |
| Valeurs polluées              | `" 6.51 "`, `"6.9 min"`, cellules vides                              | Forçage obligatoire via regex            |
| Formats de date doubles       | `2018-01` et `2018/01`                                               | Parsing sur-mesure                       |
| **Noms de gares incohérents** | `PARIS LYON`, `Paris Lyon`, `paris lyon` — `ST MALO` vs `SAINT MALO` | **132 "gares" là où il n'y en a que 59** |
| Valeurs parasites             | `'0'` en champ `Departure station`                                   | Bruit à filtrer                          |
| Commentaires multilignes      | Colonnes `Cancellation comments` etc.                                | Majoritairement vides, ignorées          |

**Effet de la normalisation des gares** : passer de 132 à 59 gares réduit drastiquement la cardinalité du OneHotEncoder et évite les "splits train/test qui inventent des gares". L'impact chiffré sur le R² dépend fortement du protocole de split (aléatoire vs temporel) ; le bénéfice qualitatif est la cohérence des tableaux et du dashboard (une seule ligne par gare réelle).

---

## 3. Feature engineering

Variables dérivées utiles à l'analyse :

- `Year`, `Month`, `Quarter`, `Season` — saisonnalité
- `IsPeakMonth` (juillet, août, décembre) — pics d'affluence
- `IsParisDeparture` / `IsParisArrival` — effet "grands hubs"
- `CancellationRate`, `DepartureDelayRate`, `SevereDelayRate` — taux normalisés
- `DelayCategory` (5 bins : On time / Slight / Moderate / Serious / Severe) — lecture catégorielle

---

## 4. Portrait statistique des retards

### Distribution
- **Retard médian à l'arrivée : ~5 min**
- Distribution **fortement asymétrique à droite** (long tail — quelques trajets-mois à 30+ min tirent la moyenne vers 6 min)
- Ponctualité (`retard ≤ 5 min`) : ~**60 %** des trajets-mois à l'échelle globale

### Saisonnalité
- **Été (juin-août)** et **décembre** présentent les retards les plus élevés — vacances scolaires + travaux estivaux + météo
- **Dérive temporelle visible** : le retard moyen monte de ~6 min (2018-2019) à ~7 min (2024-2025)
- **2020 (COVID)** : anomalie claire — moins de trafic, profil de retard atypique

### Géographie
- Les **grands hubs parisiens** (Montparnasse, Gare de Lyon) concentrent les plus forts retards moyens
- Les **gares secondaires** affichent parfois des retards moyens proches de 0, voire négatifs (trains en avance)
- Les **services internationaux** ont un retard médian ~2 min supérieur aux nationaux

### Causes (ce que la SNCF elle-même déclare)

4 causes se partagent **83 %** des retards, à parts quasi égales :

| Cause                                          |   Part |
|-----------------------------------------------:|-------:|
| 🏗️ Infrastructure (voies, caténaires, travaux) | 21.9 % |
| 🌪️ Causes externes (météo, grèves, accidents)  | 21.6 % |
| 🚦 Gestion du trafic (régulation, sillons)     | 20.4 % |
| 🚂 Matériel roulant (pannes)                   | 18.8 % |
| 👥 Passagers                                   |  7.5 % |
| 🏢 Gestion des gares                           |  7.3 % |

> **Pas de coupable unique** — le retard SNCF est un phénomène **multifactoriel**. La tendance récente montre que la **gestion du trafic monte** (20 % → 23 % depuis 2022, signe de saturation réseau) et l'**infrastructure reste structurellement lourde**.

---

## 5. Modèle prédictif

### Méthodologie rigoureuse

- **Split temporel par date** (pas par rang d'index) : cutoff strict au
  `2024-06-01`. Train = 2018-01 → 2024-05 (80 %), test = 2024-06 → 2025-12 (20 %).
  Assertion `train_dates.max() < test_dates.min()` pour garantir l'absence de
  chevauchement.
- **Imputation dans la Pipeline** (`SimpleImputer` médian), donc fittée
  uniquement sur le train — aucune statistique du test ne fuit vers le train.
- **Cross-validation temporelle** (`TimeSeriesSplit`) pour le tuning uniquement,
  pas pour la décision finale (CV trop bruitée → décision sur RMSE test).
- **Anti-fuite de cible** : exclusion stricte des variables qui mesurent déjà
  un retard (retard au départ, nb trains > 15 min, causes).
- **Encodage cyclique du mois** (`MonthSin`, `MonthCos`) pour éviter la
  discontinuité décembre/janvier.
- **Scaling ciblé** : `StandardScaler` uniquement sur les continues, pas sur
  les binaires.
- **Outliers cible** : les 9 lignes hors `[−30 ; 120]` min (corrompues source)
  sont supprimées, pas clippées.

### Performance sur le test (données réellement futures)

| Modèle                              | RMSE (min) | MAE (min) |        R² |
|:------------------------------------|-----------:|----------:|----------:|
| Baseline — moyenne globale          |       4.24 |      3.02 | **−0.07** |
| **Baseline — moyenne par route**    |   **3.55** |  **2.48** | **+0.25** |
| Ridge                               |       3.43 |      2.43 |     +0.30 |
| Random Forest                       |       3.36 |      2.37 |     +0.33 |
| **Gradient Boosting (retenu)**      |   **3.34** |  **2.42** | **+0.34** |
| Gradient Boosting tuné (CV)         |       3.39 |      2.45 |     +0.32 |

### Lecture
- La baseline globale est négative en R² car la cible a dérivé
  (train 5.98 → test 7.03 min). Elle ne dit rien d'utile sur le pouvoir prédictif.
- La **bonne baseline à battre** est la moyenne par route : elle capture à elle
  seule l'effet "trajet" et pose R² = +0.25.
- Le Gradient Boosting gagne **+0.09 R² vs route-mean** — gain modeste mais
  réel, issu surtout de la saisonnalité et de l'effet volume.
- Erreur typique (MAE) : **~2.4 minutes** — ordre de grandeur exploitable, pas
  une prédiction à la minute près.
- **Le GB tuné est plus mauvais** que le non-tuné sur test (3.39 vs 3.34) : la
  CV `TimeSeriesSplit` sur 4 folds est bruitée et recommande un modèle
  sur-régularisé. On retient donc **le non-tuné** (`n_estimators=300,
  max_depth=4, learning_rate=0.05`), pas le best-CV.

### Importance des features (permutation importance)

1. **Gare de départ** (dominante — certains hubs ont un profil de retard fondamentalement différent)
2. **Gare d'arrivée**
3. **Mois / Saison** (saisonnalité)
4. **Année** (capte la dérive temporelle)
5. Temps de parcours, volume de trains — effets modestes

---

## 6. Limites assumées

- **Granularité mensuelle** : impossible de prédire un trajet individuel ; le modèle donne la **tendance attendue** d'une liaison sur un mois.
- **Chocs exogènes** non modélisables sans flux externe (météo en temps réel, grèves, incidents ponctuels).
- **Hétéroscédasticité des résidus** : la variance d'erreur dépend du niveau prédit. L'intervalle affiché est `± 1.96 × σ_résidus` (σ calculé sur les résidus test, non via le RMSE qui inclut le biais). Approximation gaussienne — petits retards sur-enveloppés, gros sous-enveloppés. Idéal : *quantile regression* ou *conformal prediction*.
- **Gare jamais vue à l'inférence** : gérée par `OneHotEncoder(handle_unknown="ignore")`, qui encode toutes ses colonnes à 0 → la prédiction correspond à une "gare fictive" (aucune colonne station active), **pas à une moyenne conditionnelle**. Résultat arbitrairement éloigné de la moyenne empirique selon les autres features.
- **Qualité source** : ~25 % des lignes ont une somme des % causes hors-[90 ; 110] % ; les parts moyennes affichées dans la section 4 sont donc indicatives, pas des proportions exactes.

