# TARDIS — *Predicting the Unpredictable*

> Projet Epitech — Analyse de données ferroviaires SNCF

---

## Contexte

Vous faites partie d'un service d'analyse de données SNCF nouvellement créé, dont la mission est d'améliorer l'efficacité des trajets ferroviaires à travers le pays. Votre objectif : analyser des données historiques de retards, découvrir des patterns cachés et développer un modèle prédictif capable d'anticiper les retards avant qu'ils ne surviennent.

Le dashboard que vous produirez pourra être utilisé par des milliers de voyageurs pour mieux planifier leurs trajets.

---

## Préliminaires

| Élément | Détail |
|---|---|
| **Fichiers à rendre** | `tardis_eda.ipynb`, `tardis_model.ipynb`, `tardis_dashboard.py` |
| **Langage** | Python — pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit |
| **Style de code** | Formaté avec [ruff formatter](https://docs.astral.sh/ruff/) |

---

## Objectifs

- **Data Cleaning & Preprocessing** — Gérer les valeurs manquantes, les incohérences et préparer le dataset pour l'analyse.
- **Exploratory Data Analysis (EDA)** — Générer des visualisations pertinentes pour comprendre les tendances et corrélations.
- **Predictive Modeling** — Implémenter un modèle de machine learning pour prédire les retards.
- **Dashboard Development** — Créer une application web interactive avec Streamlit pour présenter les insights.

---

## Étape 1 : Exploration & Nettoyage des données

Avant toute analyse, s'assurer que le dataset est propre et structuré. Inspecter sa structure, identifier les valeurs manquantes ou les doublons, convertir les types de données et créer de nouvelles variables (jour de la semaine, heures de pointe, etc.) pour enrichir les prédictions.

### Tâches

- [ ] Charger et inspecter le dataset (nommé `dataset.csv` lors des tests automatisés)
- [ ] Gérer les valeurs manquantes et supprimer les doublons
- [ ] Convertir les colonnes vers les types de données appropriés
- [ ] Réaliser du feature engineering pour créer de nouvelles variables utiles

### Livrable attendu

Un notebook Jupyter **`tardis_eda.ipynb`** contenant :

- Chargement et inspection initiale des données
- Traitement des valeurs manquantes et des doublons
- Conversions de types de données
- Feature engineering (variables temporelles, catégories de retard, etc.)
- **Output :** fichier `cleaned_dataset.csv`

---

## Étape 2 : Visualisation & Analyse des données

Une fois le dataset nettoyé, utiliser des visualisations pour comprendre les tendances et patterns des retards ferroviaires.

### Tâches

- [ ] Générer des statistiques descriptives pour mieux comprendre le dataset
- [ ] Tracer les distributions des retards et identifier les durées les plus fréquentes
- [ ] Comparer les retards par gare et par heure de la journée

### Livrable attendu

Continuer dans **`tardis_eda.ipynb`** avec :

- Statistiques descriptives pour les variables clés
- Visualisations multiples (distributions, comparaisons, corrélations)
- Interprétations et insights rédigés

---

## Étape 3 : Construction du modèle prédictif

Construire un **modèle de régression** pour prédire la durée du retard en minutes. Le modèle prend en entrée les caractéristiques d'un trajet et retourne un retard estimé.

### Cible de prédiction

| Élément | Détail |
|---|---|
| **Variable cible** | Durée du retard (valeur continue en minutes) |
| **Exemples de features** | Gare de départ, gare d'arrivée, heure de départ, jour de la semaine, type de train |
| **Exigence minimale** | Le modèle doit surpasser une baseline (ex. : prédire le retard moyen) |

### Tâches

- [ ] **Feature Engineering** : Encoder les variables catégorielles et créer des variables temporelles (heure, jour de la semaine, heures de pointe)
- [ ] **Entraînement** : Entraîner au moins 2-3 modèles de régression (algorithmes linéaires et à base d'arbres)
- [ ] **Évaluation** : Utiliser les métriques de régression appropriées (RMSE, MAE, R²) pour comparer les modèles
- [ ] **Sélection** : Choisir le meilleur modèle et appliquer un tuning des hyperparamètres
- [ ] **Justification** : Documenter les choix effectués et les améliorations obtenues

### Livrable attendu

Un notebook Jupyter **`tardis_model.ipynb`** contenant :

- Processus de feature engineering et de sélection
- Comparaison de plusieurs modèles avec métriques
- Résultats du tuning des hyperparamètres
- Justification du modèle final retenu
- Fichier modèle sauvegardé pour intégration dans le dashboard

---

## Étape 4 : Développement du dashboard Streamlit

Rendre l'analyse accessible via un dashboard interactif permettant aux utilisateurs d'explorer les insights et d'interagir avec le modèle prédictif.

### Must Have *(obligatoire)*

- [ ] **Visualisation de la distribution des retards** : Au moins un graphique montrant les patterns de retard
- [ ] **Panneau de statistiques** : Afficher les métriques clés (retard moyen, nombre total de trajets, taux de ponctualité)
- [ ] **Interface de prédiction** : Inputs pour les paramètres du trajet + action de prédiction + affichage clair du retard prédit

### Should Have

- [ ] Analyse comparative par gare : graphiques comparant les retards selon les gares
- [ ] Filtres interactifs : contrôles pour filtrer par gare, route ou période
- [ ] Visualisations dynamiques : graphiques qui se mettent à jour selon les sélections
- [ ] Heatmap de corrélation : représentation visuelle des relations entre facteurs de retard
- [ ] Explication du modèle : importance des features ou variables ayant influencé la prédiction

### Could Have

- [ ] Plusieurs types de visualisation (histogrammes, courbes, scatter plots, heatmaps)
- [ ] Interactivité avancée (sélecteurs de plage de dates, filtres multi-sélection, drill-down)
- [ ] Indicateurs de confiance des prédictions (intervalles de prédiction)
- [ ] Prédictions comparatives (comparer des scénarios)
- [ ] Indicateurs de qualité des données (complétude, fiabilité)
- [ ] Export des données filtrées ou des graphiques

> 💡 **Conseil :** Privilégier la qualité à la quantité — mieux vaut peu de fonctionnalités bien réalisées que beaucoup de fonctionnalités bâclées.

### Livrable attendu

Un script Python **`tardis_dashboard.py`** avec :

- Application Streamlit fonctionnelle
- Toutes les fonctionnalités "Must Have" implémentées
- Intégration du modèle pour les prédictions
- **`README.md`** avec instructions d'installation et d'utilisation

---

## Récapitulatif des livrables

| Fichier | Description |
|---|---|
| `requirements.txt` | Toutes les dépendances du projet |
| `tardis_eda.ipynb` | Nettoyage, exploration et feature engineering |
| `cleaned_dataset.csv` | Dataset traité issu du notebook EDA |
| `tardis_model.ipynb` | Entraînement, évaluation et sélection du modèle |
| `model.pkl` ou `model.joblib` | Fichier du modèle entraîné |
| `tardis_dashboard.py` | Dashboard Streamlit interactif |
| `README.md` | Documentation d'installation, d'usage et du projet |

---

## Bonus

- [ ] Implémenter une technique de sélection de features pour optimiser le modèle
- [ ] Ajouter des mises à jour de données en temps réel via l'open data SNCF
- [ ] Utiliser des techniques de visualisation avancées (graphiques animés, cartes géospatiales)
- [ ] Expérimenter des modèles de deep learning pour une meilleure précision
- [ ] Inclure un composant d'explicabilité du modèle (reasons for predictions)