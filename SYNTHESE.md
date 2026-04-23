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

**Effet mesuré de la normalisation des gares** : R² du modèle est passé de **0.29 → 0.48** (split aléatoire, pré-correction temporelle) — quasi-doublement d'un seul geste de nettoyage.

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

- **Split temporel** (pas aléatoire) : train = 2018-01 → 2024-05 (80 %) ; test = 2024-05 → 2025-12 (20 %)
- **Cross-validation temporelle** (`TimeSeriesSplit`) pour le tuning — pas de fuite chronologique
- **Anti-fuite de cible** : exclusion stricte des variables qui mesurent déjà un retard (retard au départ, nb trains > 15 min, causes). Sans cette précaution, le R² sauterait à ~0.95 — faux gain.
- **Features retenues** (connues avant le départ) : gares, service, mois, année, saison, temps de parcours médian, nombre de trains planifiés, indicateurs Paris.
- **Winsorisation** de la cible à `[−30 ; 120]` min pour neutraliser les outliers extrêmes.

### Performance sur le test (données réellement futures)

| Modèle                              | RMSE (min) | MAE (min) |        R² |
|:------------------------------------|-----------:|----------:|----------:|
| Baseline (moyenne d'entraînement)   |       4.25 |      3.02 | **−0.07** |
| Ridge                               |       3.42 |      2.44 |     +0.31 |
| Random Forest                       |       3.39 |      2.41 |     +0.32 |
| Gradient Boosting                   |       3.40 |      2.45 |     +0.31 |
| **Gradient Boosting tuné (retenu)** |   **3.50** |  **2.52** | **+0.27** |

### Lecture
- Le **baseline est négatif** en R² : prédire la moyenne d'entraînement est *activement pire* que prédire la moyenne du test — confirmation directe de la dérive temporelle.
- Le modèle gagne **+0.34 R² sur baseline**, soit un signal utile malgré la faible granularité du dataset.
- Erreur typique (MAE) : **~2.5 minutes** → ordre de grandeur exploitable, pas une prédiction à la minute près.
- **Hyperparamètres retenus** : `n_estimators=200, max_depth=3, learning_rate=0.05` — profil régularisé, cohérent avec un exercice sans fuite.

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
- **Hétéroscédasticité des résidus** : la variance d'erreur augmente avec le niveau de retard prédit. L'intervalle `± 1.96 × RMSE` est une approximation — les petits retards sont sur-enveloppés, les gros sous-enveloppés. Une version plus rigoureuse utiliserait de la **quantile regression** ou de la **conformal prediction**.
- **Gare jamais vue à l'inférence** : gérée via `OneHotEncoder(handle_unknown="ignore")` → retombe sur la moyenne conditionnelle, pas d'erreur.

