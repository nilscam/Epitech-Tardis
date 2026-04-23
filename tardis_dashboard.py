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

    services = sorted(df["Service"].dropna().unique().tolist())
    selected_services = st.sidebar.multiselect(
        "Type de service", services, default=services
    )

    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()
    date_range = st.sidebar.slider(
        "Période",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM",
    )

    all_dep = sorted(df["Departure station"].dropna().unique().tolist())
    all_arr = sorted(df["Arrival station"].dropna().unique().tolist())

    selected_deps = st.sidebar.multiselect(
        "Gares de départ (vide = toutes)", all_dep, default=[]
    )
    selected_arrs = st.sidebar.multiselect(
        "Gares d'arrivée (vide = toutes)", all_arr, default=[]
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

    if filtered.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        st.stop()

    # ---------- KPIs ----------
    st.subheader("📊 Indicateurs clés")

    total_trips = int(filtered["Number of scheduled trains"].sum())
    total_cancelled = int(filtered["Number of cancelled trains"].sum())
    avg_delay = float(filtered[TARGET].mean())
    pct_on_time = float((filtered[TARGET] <= 5).mean() * 100)
    cancel_rate = (total_cancelled / total_trips * 100) if total_trips else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Trajets programmés", f"{total_trips:,}")
    kpi_card(c2, "Retard moyen à l'arrivée", f"{avg_delay:.1f} min")
    kpi_card(c3, "% à l'heure (≤ 5 min)", f"{pct_on_time:.1f} %")
    kpi_card(c4, "Trains annulés", f"{total_cancelled:,}")
    kpi_card(c5, "Taux d'annulation", f"{cancel_rate:.2f} %")

    st.divider()

    # ---------- Tabs ----------
    tabs = st.tabs(
        [
            "📈 Distribution",
            "🚉 Par gare",
            "📅 Tendances",
            "🔗 Corrélations",
            "🔮 Prédiction",
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
            "Renseigne les paramètres d'un trajet et le modèle estime le retard moyen à l'arrivée."
        )

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                dep = st.selectbox("Gare de départ", all_dep, index=0)
                service = st.selectbox("Type de service", services, index=0)
                year = st.number_input(
                    "Année", min_value=2018, max_value=2030, value=int(df["Year"].max())
                )
            with col2:
                arr = st.selectbox("Gare d'arrivée", all_arr, index=0)
                month = st.selectbox("Mois", list(range(1, 13)), index=5)
                journey_time = st.number_input(
                    "Temps de parcours moyen (min)",
                    min_value=10,
                    max_value=600,
                    value=int(df["Average journey time"].median()),
                )
            with col3:
                scheduled = st.number_input(
                    "Nombre de trains planifiés sur le mois",
                    min_value=1,
                    max_value=5000,
                    value=int(df["Number of scheduled trains"].median()),
                )
                compare_mode = st.checkbox(
                    "Comparer un scénario alternatif", value=False
                )

            submit = st.form_submit_button("Prédire le retard", type="primary")

        if submit:
            if dep == arr:
                st.warning(
                    "Gare de départ et d'arrivée identiques — vérifie ta saisie."
                )
                st.stop()

            season_map = {
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

            def build_row(dep_, arr_, svc, yr, mo, jt, sched) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "Departure station": dep_,
                            "Arrival station": arr_,
                            "Service": svc,
                            "Season": season_map[mo],
                            "Year": yr,
                            "Month": mo,
                            "Quarter": (mo - 1) // 3 + 1,
                            "Average journey time": jt,
                            "Number of scheduled trains": sched,
                            "IsPeakMonth": int(mo in (7, 8, 12)),
                            "IsParisDeparture": int("PARIS" in dep_.upper()),
                            "IsParisArrival": int("PARIS" in arr_.upper()),
                        }
                    ]
                )[features]

            row = build_row(dep, arr, service, year, month, journey_time, scheduled)
            prediction = float(pipeline.predict(row)[0])

            # Intervalle empirique : RMSE du modèle
            rmse = artifact["metrics"]["rmse"]
            low = max(0.0, prediction - 1.96 * rmse)
            high = prediction + 1.96 * rmse

            st.success(f"**Retard prédit : {prediction:.1f} minutes**")
            st.caption(
                f"Intervalle de confiance ~95 % (± 1.96 × RMSE) : "
                f"[{low:.1f} ; {high:.1f}] min — RMSE modèle = {rmse:.2f} min."
            )

            # Context — retards historiques observés sur ce trajet
            route_hist = df[
                (df["Departure station"] == dep) & (df["Arrival station"] == arr)
            ]
            if not route_hist.empty:
                st.info(
                    f"📚 Historique sur ce trajet ({len(route_hist)} mois) : "
                    f"moyenne **{route_hist[TARGET].mean():.1f} min**, "
                    f"médiane **{route_hist[TARGET].median():.1f} min**."
                )

            if compare_mode:
                st.markdown("---")
                st.markdown("#### 🔁 Scénario alternatif")
                colA, colB, colC = st.columns(3)
                alt_month = colA.selectbox(
                    "Mois alt.", list(range(1, 13)), index=0, key="alt_month"
                )
                alt_service = colB.selectbox(
                    "Service alt.",
                    services,
                    index=min(1, len(services) - 1),
                    key="alt_service",
                )
                alt_year = colC.number_input(
                    "Année alt.",
                    min_value=2018,
                    max_value=2030,
                    value=int(year),
                    key="alt_year",
                )
                alt_row = build_row(
                    dep, arr, alt_service, alt_year, alt_month, journey_time, scheduled
                )
                alt_pred = float(pipeline.predict(alt_row)[0])
                delta = alt_pred - prediction
                st.metric(
                    f"Retard prédit (scénario alt.)",
                    f"{alt_pred:.1f} min",
                    delta=f"{delta:+.1f} min vs scénario principal",
                    delta_color="inverse",
                )

        st.divider()
        st.markdown("#### 🧠 Qualité du modèle")
        m = artifact["metrics"]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("RMSE", f"{m['rmse']:.2f} min")
        mc2.metric("MAE", f"{m['mae']:.2f} min")
        mc3.metric("R²", f"{m['r2']:.3f}")
        st.caption(
            f"Meilleurs hyperparamètres : `{artifact['best_params']}` · "
            f"Modèle : Gradient Boosting tuné par GridSearchCV."
        )

    st.divider()
    st.caption(
        f"Dataset : {len(df):,} trajets-mois · Filtré : {len(filtered):,} · "
        f"Période : {df['Date'].min():%Y-%m} → {df['Date'].max():%Y-%m}"
    )


if __name__ == "__main__":
    main()
