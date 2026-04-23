"""TARDIS — Dashboard Streamlit pour l'analyse et la prédiction des retards SNCF."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

DATA_PATH = Path("cleaned_dataset.csv")
MODEL_PATH = Path("model.joblib")
TARGET = "Average delay of all trains at arrival"

st.set_page_config(
    page_title="TARDIS — SNCF Delay Intelligence",
    page_icon="🚆",
    layout="wide",
)

sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner="Chargement du dataset nettoyé…")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(path: Path) -> dict:
    return joblib.load(path)


def kpi_card(col, label: str, value: str, help_text: str | None = None) -> None:
    col.metric(label, value, help=help_text)


def main() -> None:
    st.title("🚆 TARDIS — *Predicting the Unpredictable*")
    st.caption(
        "Dashboard d'analyse des retards SNCF — agrégations mensuelles par trajet (gare → gare)."
    )

    if not DATA_PATH.exists():
        st.error(
            f"Fichier `{DATA_PATH}` introuvable. Exécute d'abord le notebook `tardis_eda.ipynb`."
        )
        st.stop()
    if not MODEL_PATH.exists():
        st.error(
            f"Fichier `{MODEL_PATH}` introuvable. Exécute d'abord le notebook `tardis_model.ipynb`."
        )
        st.stop()

    df = load_data(DATA_PATH)
    artifact = load_model(MODEL_PATH)
    pipeline = artifact["pipeline"]
    features = artifact["features"]

    # ---------- Sidebar filters ----------
    st.sidebar.header("🎛️ Filtres")

    if st.sidebar.button("🔄 Réinitialiser", use_container_width=True):
        for k in ("flt_services", "flt_deps", "flt_arrs", "flt_period"):
            st.session_state.pop(k, None)
        st.rerun()

    services = sorted(df["Service"].dropna().unique().tolist())
    selected_services = st.sidebar.multiselect(
        "Type de service",
        services,
        default=services,
        key="flt_services",
        placeholder="Tous les services",
    )

    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()
    date_range = st.sidebar.slider(
        "Période",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM",
        key="flt_period",
    )

    all_dep = sorted(df["Departure station"].dropna().unique().tolist())

    with st.sidebar.expander("🚉 Gares", expanded=False):
        selected_deps = st.multiselect(
            "Gares de départ",
            all_dep,
            default=[],
            key="flt_deps",
            placeholder="Toutes les gares",
            help="Laisser vide pour inclure toutes les gares de départ.",
        )

        # Cascade : si une/des gares de départ sont choisies, on restreint
        # les arrivées aux destinations réellement desservies.
        if selected_deps:
            reachable = (
                df.loc[df["Departure station"].isin(selected_deps), "Arrival station"]
                .dropna()
                .unique()
            )
            arr_options = sorted(reachable.tolist())
            arr_help = (
                f"{len(arr_options)} destination(s) desservie(s) depuis "
                f"{len(selected_deps)} gare(s) sélectionnée(s)."
            )
        else:
            arr_options = sorted(df["Arrival station"].dropna().unique().tolist())
            arr_help = "Laisser vide pour inclure toutes les gares d'arrivée."

        # Purge les sélections obsolètes si la gare de départ a changé
        if "flt_arrs" in st.session_state:
            st.session_state["flt_arrs"] = [
                a for a in st.session_state["flt_arrs"] if a in arr_options
            ]

        selected_arrs = st.multiselect(
            "Gares d'arrivée",
            arr_options,
            default=[],
            key="flt_arrs",
            placeholder="Toutes les destinations",
            help=arr_help,
        )

    mask = (
        df["Service"].isin(selected_services)
        & (df["Date"] >= pd.Timestamp(date_range[0]))
        & (df["Date"] <= pd.Timestamp(date_range[1]))
    )
    if selected_deps:
        mask &= df["Departure station"].isin(selected_deps)
    if selected_arrs:
        mask &= df["Arrival station"].isin(selected_arrs)

    filtered = df.loc[mask].copy()

    # Feedback de volume en bas de sidebar
    n_total = len(df)
    n_filtered = len(filtered)
    ratio = n_filtered / n_total if n_total else 0
    st.sidebar.caption(
        f"**{n_filtered:,}** / {n_total:,} lignes retenues ({ratio:.0%})"
    )

    if filtered.empty:
        st.sidebar.warning("Aucune donnée ne correspond aux filtres.")
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        st.stop()

    # ---------- KPIs ----------
    st.subheader("📊 Indicateurs clés")

    # Seuils de ponctualité SNCF (standard communiqué par type de service) :
    # - National courte distance / TER : 5 min
    # - National grandes lignes / TGV : 15 min ; l'International cale sur ce même seuil ici.
    PUNCT_THRESHOLDS = {"National": 5.0, "International": 15.0}

    total_trips = int(filtered["Number of scheduled trains"].sum())
    total_cancelled = int(filtered["Number of cancelled trains"].sum())
    avg_delay = float(filtered[TARGET].mean())

    # Ponctualité pondérée par service (au lieu d'un seuil unique à 5 min)
    thresholds = filtered["Service"].map(PUNCT_THRESHOLDS).fillna(5.0)
    pct_on_time = float((filtered[TARGET] <= thresholds).mean() * 100)

    cancel_rate = (total_cancelled / total_trips * 100) if total_trips else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Trajets programmés", f"{total_trips:,}")
    kpi_card(c2, "Retard moyen à l'arrivée", f"{avg_delay:.1f} min")
    kpi_card(
        c3,
        "% à l'heure",
        f"{pct_on_time:.1f} %",
        help_text="Seuil par service : National ≤ 5 min, International ≤ 15 min",
    )
    kpi_card(c4, "Trains annulés", f"{total_cancelled:,}")
    kpi_card(c5, "Taux d'annulation", f"{cancel_rate:.2f} %")

    st.divider()

    # ---------- Tabs ----------
    tabs = st.tabs(
        [
            "📈 Distribution",
            "🚉 Par gare",
            "📅 Évolution",
            "🔗 Corrélations",
            "🔮 Prédiction",
            "📝 Synthèse",
        ]
    )

    # ---- Tab 1 : Distribution ----
    with tabs[0]:
        st.subheader("Distribution du retard moyen à l'arrivée")
        col_a, col_b = st.columns([2, 1])

        with col_a:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.histplot(filtered[TARGET], bins=50, kde=True, color="#1f77b4", ax=ax)
            ax.axvline(
                filtered[TARGET].median(),
                color="red",
                linestyle="--",
                label=f"Médiane = {filtered[TARGET].median():.1f} min",
            )
            ax.set_xlabel("Retard moyen (min)")
            ax.set_ylabel("Nombre de trajets-mois")
            ax.legend()
            st.pyplot(fig, clear_figure=True)

        with col_b:
            cat_order = [
                "On time / Early",
                "Slight (0-5m)",
                "Moderate (5-15m)",
                "Serious (15-30m)",
                "Severe (30m+)",
            ]
            cat_counts = (
                filtered["DelayCategory"].value_counts().reindex(cat_order).fillna(0)
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x=cat_counts.values,
                y=cat_counts.index,
                palette="YlOrRd",
                ax=ax,
            )
            ax.set_xlabel("Trajets-mois")
            ax.set_ylabel("")
            ax.set_title("Catégories de retard")
            st.pyplot(fig, clear_figure=True)

        st.subheader("Retard par type de service")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        sns.boxplot(
            data=filtered,
            x="Service",
            y=TARGET,
            palette="Set2",
            ax=ax,
            showfliers=False,
        )
        ax.set_ylabel("Retard moyen (min)")
        st.pyplot(fig, clear_figure=True)

    # ---- Tab 2 : Par gare ----
    with tabs[1]:
        st.subheader("Comparatif des gares de départ")

        station_stats = (
            filtered.groupby("Departure station")
            .agg(
                AvgDelay=(TARGET, "mean"),
                MedianDelay=(TARGET, "median"),
                Trips=("Number of scheduled trains", "sum"),
                Routes=("Route", "nunique"),
            )
            .query("Trips > 100")
            .sort_values("AvgDelay", ascending=False)
        )

        top_n = st.slider("Nombre de gares à afficher", 5, 30, 15, key="top_n_stations")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top gares les plus en retard**")
            top = station_stats.head(top_n).reset_index()
            fig, ax = plt.subplots(figsize=(7, max(3.5, top_n * 0.28)))
            sns.barplot(
                data=top, x="AvgDelay", y="Departure station", palette="Reds_r", ax=ax
            )
            ax.set_xlabel("Retard moyen (min)")
            ax.set_ylabel("")
            st.pyplot(fig, clear_figure=True)

        with col_b:
            st.markdown("**Top gares les plus ponctuelles**")
            bottom = station_stats.tail(top_n).iloc[::-1].reset_index()
            fig, ax = plt.subplots(figsize=(7, max(3.5, top_n * 0.28)))
            sns.barplot(
                data=bottom,
                x="AvgDelay",
                y="Departure station",
                palette="Greens",
                ax=ax,
            )
            ax.set_xlabel("Retard moyen (min)")
            ax.set_ylabel("")
            st.pyplot(fig, clear_figure=True)

        st.markdown("**Tableau détaillé (filtré)**")
        st.dataframe(
            station_stats.round(2),
            use_container_width=True,
            height=320,
        )

        st.download_button(
            "⬇️ Exporter ce tableau (CSV)",
            data=station_stats.round(3).to_csv().encode(),
            file_name="tardis_stations_filtered.csv",
            mime="text/csv",
        )

    # ---- Tab 3 : Tendances ----
    with tabs[2]:
        st.subheader("Évolution mensuelle du retard moyen")
        monthly = (
            filtered.groupby(filtered["Date"].dt.to_period("M"))[TARGET]
            .mean()
            .reset_index()
        )
        monthly["Date"] = monthly["Date"].dt.to_timestamp()
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=monthly, x="Date", y=TARGET, marker="o", ax=ax)
        ax.set_ylabel("Retard moyen (min)")
        ax.set_xlabel("")
        st.pyplot(fig, clear_figure=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Par mois de l'année**")
            m = filtered.groupby("Month")[TARGET].mean()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.barplot(x=m.index, y=m.values, palette="viridis", ax=ax)
            ax.set_xlabel("Mois")
            ax.set_ylabel("Retard moyen (min)")
            st.pyplot(fig, clear_figure=True)

        with col_b:
            st.markdown("**Par année**")
            y = filtered.groupby("Year")[TARGET].mean()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.barplot(x=y.index.astype(int), y=y.values, palette="mako", ax=ax)
            ax.set_xlabel("Année")
            ax.set_ylabel("Retard moyen (min)")
            st.pyplot(fig, clear_figure=True)

        st.markdown("**Heatmap : retard moyen par année × mois**")
        pivot = filtered.pivot_table(
            index="Year", columns="Month", values=TARGET, aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.heatmap(
            pivot,
            cmap="RdYlGn_r",
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "min"},
            ax=ax,
        )
        st.pyplot(fig, clear_figure=True)

    # ---- Tab 4 : Corrélations ----
    with tabs[3]:
        st.subheader("Matrice de corrélation")
        num_cols = [
            "Average journey time",
            "Number of scheduled trains",
            "Average delay of all trains at departure",
            TARGET,
            "Number of trains delayed > 15min",
            "Number of trains delayed > 60min",
            "CancellationRate",
            "DepartureDelayRate",
            "SevereDelayRate",
            "Pct delay due to external causes",
            "Pct delay due to infrastructure",
            "Pct delay due to traffic management",
            "Pct delay due to rolling stock",
        ]
        corr = filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True, ax=ax
        )
        st.pyplot(fig, clear_figure=True)

        st.subheader("Contribution moyenne des causes de retard")
        cause_cols = [
            "Pct delay due to external causes",
            "Pct delay due to infrastructure",
            "Pct delay due to traffic management",
            "Pct delay due to rolling stock",
            "Pct delay due to station management and equipment reuse",
            "Pct delay due to passenger handling (crowding, disabled persons, connections)",
        ]
        cause_means = filtered[cause_cols].mean().sort_values()
        pretty = {c: c.replace("Pct delay due to ", "") for c in cause_cols}
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            x=cause_means.values,
            y=[pretty[c] for c in cause_means.index],
            palette="flare",
            ax=ax,
        )
        ax.set_xlabel("% moyen du retard total")
        st.pyplot(fig, clear_figure=True)

    # ---- Tab 5 : Prédiction ----
    with tabs[4]:
        st.subheader("🔮 Prédire le retard d'un trajet")
        st.caption(
            "Choisis une gare de départ et d'arrivée existante. "
            "Les caractéristiques du trajet (service, temps de parcours, volume) "
            "sont déterminées automatiquement à partir de l'historique **antérieur "
            "à la date de prédiction**."
        )

        # Cutoff train stocké dans l'artifact ; fallback = max des dates train utilisées
        # au moment du fit. On s'en sert pour ne PAS utiliser de lignes futures comme
        # meta d'une route à l'inférence (sinon : fuite test → prédiction).
        train_cutoff = pd.Timestamp(artifact.get("train_cutoff", df["Date"].max()))
        df_train_slice = df[df["Date"] < train_cutoff]

        route_groups = df_train_slice.groupby(["Departure station", "Arrival station"])
        route_meta = route_groups.agg(
            service=("Service", lambda s: s.mode().iloc[0]),
            journey_time=("Average journey time", "median"),
            scheduled=("Number of scheduled trains", "median"),
            n_months=(TARGET, "count"),
            avg_delay=(TARGET, "mean"),
            median_delay=(TARGET, "median"),
            punctuality=(TARGET, lambda s: (s <= 5).mean() * 100),
        ).reset_index()

        MONTH_LABELS = {
            1: "Janvier",
            2: "Février",
            3: "Mars",
            4: "Avril",
            5: "Mai",
            6: "Juin",
            7: "Juillet",
            8: "Août",
            9: "Septembre",
            10: "Octobre",
            11: "Novembre",
            12: "Décembre",
        }
        SEASON_MAP = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }

        # --- Sélecteurs interactifs (hors formulaire pour la cascade dep → arr) ---
        departures_with_routes = sorted(route_meta["Departure station"].unique())
        col_dep, col_arr = st.columns(2)
        dep = col_dep.selectbox(
            "Gare de départ",
            departures_with_routes,
            key="pred_dep",
        )

        valid_arrivals = sorted(
            route_meta.loc[
                route_meta["Departure station"] == dep, "Arrival station"
            ].unique()
        )
        arr = col_arr.selectbox(
            "Gare d'arrivée",
            valid_arrivals,
            key="pred_arr",
            help=f"{len(valid_arrivals)} destination(s) desservie(s) depuis {dep}.",
        )

        meta = route_meta[
            (route_meta["Departure station"] == dep)
            & (route_meta["Arrival station"] == arr)
        ].iloc[0]

        # --- Carte "informations du trajet" (dérivées, non modifiables) ---
        st.markdown(f"#### 🚆 Trajet **{dep} → {arr}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Type de service", meta["service"])
        m2.metric("Temps de parcours médian", f"{meta['journey_time']:.0f} min")
        m3.metric("Trains planifiés / mois", f"{meta['scheduled']:.0f}")
        m4.metric(
            "Ponctualité historique",
            f"{meta['punctuality']:.0f} %",
            help=(
                f"Part des {int(meta['n_months'])} mois observés sur ce trajet "
                "où le retard moyen est resté ≤ 5 min."
            ),
        )

        # --- Formulaire : seulement les paramètres temporels décisionnels ---
        with st.form("prediction_form"):
            col_y, col_m = st.columns(2)
            year = col_y.number_input(
                "Année",
                min_value=2018,
                max_value=2030,
                value=int(df["Year"].max()),
                step=1,
            )
            month = col_m.selectbox(
                "Mois",
                list(range(1, 13)),
                index=5,
                format_func=lambda m: f"{m:02d} — {MONTH_LABELS[m]}",
            )
            submit = st.form_submit_button("Prédire le retard", type="primary")

        if submit:

            def build_row(yr: int, mo: int) -> pd.DataFrame:
                row = {
                    "Departure station": dep,
                    "Arrival station": arr,
                    "Service": meta["service"],
                    "Season": SEASON_MAP[mo],
                    "Year": yr,
                    "Month": mo,
                    "Quarter": (mo - 1) // 3 + 1,
                    "MonthSin": float(np.sin(2 * np.pi * mo / 12)),
                    "MonthCos": float(np.cos(2 * np.pi * mo / 12)),
                    "Average journey time": float(meta["journey_time"]),
                    "Number of scheduled trains": float(meta["scheduled"]),
                    "IsPeakMonth": int(mo in (7, 8, 12)),
                    "IsWorksMonth": int(mo in (6, 7, 8, 9)),
                    "IsParisDeparture": int("PARIS" in dep.upper()),
                    "IsParisArrival": int("PARIS" in arr.upper()),
                }
                # Toutes les features attendues par le pipeline — on remplit
                # uniquement celles présentes dans `row` ; le reste serait
                # manquant (mais le pipeline a un imputer).
                return pd.DataFrame([{k: row.get(k) for k in features}])

            prediction = float(pipeline.predict(build_row(int(year), int(month)))[0])

            # IC correct : ± 1.96 × std(résidus) (le RMSE inclut le biais et n'est
            # pas σ). Fallback sur RMSE si residual_std absent (ancien artifact).
            sigma = artifact["metrics"].get("residual_std", artifact["metrics"]["rmse"])
            bias = artifact["metrics"].get("residual_bias", 0.0)
            low = prediction - 1.96 * sigma
            high = prediction + 1.96 * sigma

            st.success(f"**Retard prédit : {prediction:.1f} minutes**")
            st.caption(
                f"Intervalle approximatif (± 1.96 × σ_résidus) : "
                f"[{low:.1f} ; {high:.1f}] min — σ = {sigma:.2f} min, "
                f"biais moyen du modèle = {bias:+.2f} min. "
                "Interprétation prudente : résidus hétéroscédastiques, "
                "la variance réelle dépend du niveau prédit."
            )

            st.info(
                f"📚 **Historique** sur {dep} → {arr} ({int(meta['n_months'])} mois "
                f"avant {train_cutoff:%Y-%m}) : "
                f"retard moyen observé **{meta['avg_delay']:.1f} min**, "
                f"médiane **{meta['median_delay']:.1f} min**."
            )

    # ---- Tab 6 : Synthèse ----
    with tabs[5]:
        synthese_path = Path("SYNTHESE.md")
        if synthese_path.exists():
            st.markdown(synthese_path.read_text(encoding="utf-8"))
        else:
            st.warning(
                f"Fichier `{synthese_path}` introuvable. "
                "Place une synthèse Markdown à la racine du projet."
            )

    st.divider()
    st.caption(
        f"Dataset : {len(df):,} trajets-mois · Filtré : {len(filtered):,} · "
        f"Période : {df['Date'].min():%Y-%m} → {df['Date'].max():%Y-%m}"
    )


if __name__ == "__main__":
    main()
