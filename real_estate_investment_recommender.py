"""
Real Estate Investment Recommender — Streamlit App
A professional-grade tool for analyzing and recommending real estate investments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PropIQ — Real Estate Recommender",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8e4dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #16191f;
    border-right: 1px solid #2a2d35;
}
[data-testid="stSidebar"] * {
    color: #c9c5bc !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #8a8680 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1d26 0%, #12151d 60%, #0d1018 100%);
    border: 1px solid #2a2d35;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(234,179,8,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #f5f0e8;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #8a8680;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}
.accent { color: #e5b94e; }

/* Metric cards */
.metric-card {
    background: #16191f;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #e5b94e44; }
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #e5b94e;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8a8680;
    margin-top: 0.4rem;
}

/* Section headings */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #f5f0e8;
    border-left: 3px solid #e5b94e;
    padding-left: 0.8rem;
    margin: 2rem 0 1rem 0;
}

/* Score badge */
.score-high { color: #4ade80; font-weight: 600; }
.score-mid  { color: #facc15; font-weight: 600; }
.score-low  { color: #f87171; font-weight: 600; }

/* Divider */
hr { border-color: #2a2d35; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2a2d35;
    border-radius: 10px;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    background: #e5b94e;
    color: #0f1117;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #f0cb6a;
    transform: translateY(-1px);
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2a2d35;
    border-radius: 10px;
    padding: 1rem;
    background: #13161d;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.85rem;
    font-weight: 500;
    color: #8a8680 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e5b94e !important;
    border-bottom-color: #e5b94e !important;
}

/* Info/warning boxes */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_clean(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        return None

    # Column normalisation
    rename_map = {
        "beds": "bedrooms", "baths": "bathrooms",
        "type": "property_type", "city": "location",
        "neighbourhood": "location", "neighborhood": "location",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # square_footage
    if "size" in df.columns and "square_footage" not in df.columns:
        df["square_footage"] = pd.to_numeric(
            df["size"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
        )
        missing = df["square_footage"].isnull().sum()
        if missing:
            df.loc[df["square_footage"].isnull(), "square_footage"] = np.random.randint(800, 3000, missing)
    elif "square_footage" not in df.columns:
        df["square_footage"] = np.random.randint(800, 3000, len(df))
    df["square_footage"] = df["square_footage"].fillna(1200).astype(int)

    # Synthetic columns when missing
    n = len(df)
    defaults = {
        "property_id":          lambda: range(1, n + 1),
        "year_built":           lambda: np.random.randint(1980, 2023, n),
        "bedrooms":             lambda: np.random.randint(1, 5, n),
        "bathrooms":            lambda: np.random.randint(1, 4, n),
        "property_type":        lambda: np.random.choice(["Apartment","House","Condo","Townhouse"], n),
        "location":             lambda: np.random.choice(["Downtown","Suburban Area","Rural Area","Coastal"], n),
        "property_tax_rate":    lambda: np.random.uniform(0.005, 0.02, n).round(4),
        "vacancy_rate":         lambda: np.random.uniform(0.01, 0.10, n).round(3),
        "amenities_score":      lambda: np.random.randint(1, 10, n),
    }
    for col, fn in defaults.items():
        if col not in df.columns:
            df[col] = list(fn())

    # rental_income_potential
    if "rental_income_potential" not in df.columns:
        if "price" in df.columns:
            df["rental_income_potential"] = (
                df["price"] * np.random.uniform(0.001, 0.005, n) + np.random.randint(500, 2000, n)
            ).clip(lower=1000).astype(int)
        else:
            df["rental_income_potential"] = np.random.randint(1000, 5000, n)

    # price fallback
    if "price" not in df.columns:
        df["price"] = np.random.randint(150_000, 1_000_000, n)

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(300_000)
    return df


def generate_synthetic(n=50):
    locations = ["Downtown", "Suburban Area", "Rural Area", "Coastal"]
    prop_types = ["Apartment", "House", "Condo", "Townhouse"]
    data = {
        "property_id":          range(1, n + 1),
        "location":             np.random.choice(locations, n),
        "property_type":        np.random.choice(prop_types, n),
        "bedrooms":             np.random.randint(1, 5, n),
        "bathrooms":            np.random.randint(1, 4, n),
        "square_footage":       np.random.randint(800, 3000, n),
        "price":                np.random.randint(150_000, 1_000_000, n),
        "year_built":           np.random.randint(1980, 2023, n),
        "rental_income_potential": np.random.randint(1000, 5000, n),
        "property_tax_rate":    np.random.uniform(0.005, 0.02, n).round(4),
        "vacancy_rate":         np.random.uniform(0.01, 0.10, n).round(3),
        "cap_rate":             np.random.uniform(0.03, 0.08, n).round(3),
        "amenities_score":      np.random.randint(1, 10, n),
    }
    return pd.DataFrame(data)


def engineer_features(df):
    df = df.copy()
    df["gross_rental_yield"] = (df["rental_income_potential"] * 12) / df["price"]
    df["noi"] = (df["rental_income_potential"] * 12) - (df["price"] * df["property_tax_rate"])
    df["cash_on_cash_return"] = df["noi"] / df["price"]
    if "cap_rate" not in df.columns:
        df["cap_rate"] = ((df["rental_income_potential"] * 12) / df["price"]).clip(0.03, 0.08).round(3)
    df["effective_yield"] = df["gross_rental_yield"] * (1 - df["vacancy_rate"])
    df["price_per_sqft"] = df["price"] / df["square_footage"]
    df["property_age"] = 2024 - df["year_built"]
    df["investment_score"] = (
        (df["cap_rate"]             / df["cap_rate"].max())             * 30 +
        (df["gross_rental_yield"]   / df["gross_rental_yield"].max())   * 25 +
        (1 - df["vacancy_rate"]     / df["vacancy_rate"].max())         * 20 +
        (df["amenities_score"]      / 10)                               * 15 +
        (1 - df["property_age"]     / df["property_age"].max())         * 10
    ).round(2)
    return df


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def recommend(
    df,
    min_cap_rate, max_cap_rate,
    min_yield, max_price,
    min_beds, min_score,
    preferred_locations, preferred_types,
    sort_by,
):
    mask = (
        (df["cap_rate"]           >= min_cap_rate) &
        (df["cap_rate"]           <= max_cap_rate) &
        (df["gross_rental_yield"] >= min_yield) &
        (df["price"]              <= max_price) &
        (df["bedrooms"]           >= min_beds) &
        (df["investment_score"]   >= min_score)
    )
    result = df[mask].copy()
    if preferred_locations:
        result = result[result["location"].isin(preferred_locations)]
    if preferred_types:
        result = result[result["property_type"].isin(preferred_types)]
    return result.sort_values(sort_by, ascending=False)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
DARK_BG = "#13161d"
GRID    = "#2a2d35"
ACCENT  = "#e5b94e"
TEXT    = "#c9c5bc"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
    font=dict(family="DM Sans", color=TEXT, size=12),
    xaxis=dict(gridcolor=GRID, showline=False),
    yaxis=dict(gridcolor=GRID, showline=False),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

PALETTE = ["#e5b94e","#60a5fa","#4ade80","#f87171","#a78bfa","#fb923c"]


def fig_price_dist(df):
    fig = px.histogram(df, x="price", nbins=30, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, title="Price Distribution")
    return fig

def fig_scatter(df):
    fig = px.scatter(
        df, x="square_footage", y="price",
        color="property_type", size="investment_score",
        color_discrete_sequence=PALETTE,
        hover_data=["location","cap_rate","gross_rental_yield"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Price vs Square Footage")
    return fig

def fig_cap_vs_yield(df):
    fig = px.scatter(
        df, x="cap_rate", y="gross_rental_yield",
        color="location", size="price",
        color_discrete_sequence=PALETTE,
        hover_data=["property_type","investment_score"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Cap Rate vs Gross Rental Yield")
    return fig

def fig_location_avg(df):
    grp = df.groupby("location")[["price","cap_rate","gross_rental_yield"]].mean().reset_index()
    fig = go.Figure()
    for col, color in zip(["price","cap_rate","gross_rental_yield"], PALETTE):
        norm = grp[col] / grp[col].max()
        fig.add_trace(go.Bar(
            name=col.replace("_"," ").title(),
            x=grp["location"], y=norm,
            marker_color=color,
        ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", title="Normalised Metrics by Location")
    return fig

def fig_score_dist(df):
    fig = px.histogram(df, x="investment_score", nbins=20, color_discrete_sequence=[ACCENT])
    fig.update_layout(**PLOTLY_LAYOUT, title="Investment Score Distribution")
    return fig

def fig_correlation(df):
    cols = ["price","cap_rate","gross_rental_yield","cash_on_cash_return","vacancy_rate","investment_score","amenities_score"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text}",
        hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Correlation Heatmap", height=450)
    return fig

def fig_top_properties(recs, n=10):
    top = recs.head(n)
    fig = px.bar(
        top, x="investment_score", y=top.index.astype(str),
        orientation="h", color="location",
        color_discrete_sequence=PALETTE,
        hover_data=["price","cap_rate","property_type"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Top {n} Recommended Properties by Score", yaxis_title="Property Index")
    return fig

def fig_roi_comparison(recs):
    cols = ["cap_rate","gross_rental_yield","cash_on_cash_return","effective_yield"]
    available = [c for c in cols if c in recs.columns]
    melted = recs[available].head(15).reset_index().melt(id_vars="index", value_vars=available)
    fig = px.box(
        melted, x="variable", y="value",
        color="variable", color_discrete_sequence=PALETTE,
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Return Metric Distributions (Top Recommendations)", showlegend=False)
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():

    # ── HERO ──────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Prop<span class="accent">IQ</span></div>
        <div class="hero-sub">Intelligent Real Estate Investment Analysis & Recommendation Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Data Source")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        st.markdown("---")
        st.markdown("### 🎯 Filters")

        use_synthetic = uploaded is None

        if use_synthetic:
            st.info("No file uploaded — using synthetic demo data (50 properties).")
            df_raw = generate_synthetic(50)
        else:
            df_raw = load_and_clean(uploaded)
            if df_raw is None or df_raw.empty:
                st.error("Could not parse file.")
                return

        df = engineer_features(df_raw)

        max_possible_price = int(df["price"].max())
        locations = sorted(df["location"].dropna().unique().tolist())
        prop_types = sorted(df["property_type"].dropna().unique().tolist())

        min_cap = st.slider("Min Cap Rate", 0.0, 0.15, 0.04, 0.005, format="%.3f")
        max_cap = st.slider("Max Cap Rate", 0.0, 0.20, 0.15, 0.005, format="%.3f")
        min_yield = st.slider("Min Gross Rental Yield", 0.0, 0.20, 0.05, 0.005, format="%.3f")
        max_price = st.slider("Max Price ($)", 50_000, max_possible_price, min(700_000, max_possible_price), 10_000)
        min_beds = st.slider("Min Bedrooms", 1, 6, 1)
        min_score = st.slider("Min Investment Score", 0.0, 100.0, 0.0, 1.0)

        st.markdown("### 📍 Location & Type")
        sel_locations = st.multiselect("Preferred Locations", locations)
        sel_types = st.multiselect("Property Types", prop_types)

        st.markdown("### 🔢 Sort By")
        sort_by = st.selectbox(
            "Sort recommendations by",
            ["investment_score","cap_rate","gross_rental_yield","cash_on_cash_return","price"],
        )

        st.markdown("---")
        run = st.button("🔍 Find Investments")

    # ── SUMMARY METRICS ──────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Properties</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">${df['price'].median()/1000:.0f}K</div>
            <div class="metric-label">Median Price</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['cap_rate'].mean()*100:.1f}%</div>
            <div class="metric-label">Avg Cap Rate</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['gross_rental_yield'].mean()*100:.1f}%</div>
            <div class="metric-label">Avg Rental Yield</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['investment_score'].mean():.1f}</div>
            <div class="metric-label">Avg Invest. Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🏆 Recommendations", "📈 Analytics", "🗂️ Raw Data"])

    # ── TAB 1: EDA ────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        row1 = st.columns(2)
        with row1[0]:
            st.plotly_chart(fig_price_dist(df), use_container_width=True)
        with row1[1]:
            st.plotly_chart(fig_score_dist(df), use_container_width=True)

        row2 = st.columns(2)
        with row2[0]:
            st.plotly_chart(fig_scatter(df), use_container_width=True)
        with row2[1]:
            st.plotly_chart(fig_location_avg(df), use_container_width=True)

        st.plotly_chart(fig_correlation(df), use_container_width=True)

    # ── TAB 2: RECOMMENDATIONS ────────────────
    with tab2:
        st.markdown('<div class="section-title">Investment Recommendations</div>', unsafe_allow_html=True)

        recs = recommend(
            df,
            min_cap_rate=min_cap, max_cap_rate=max_cap,
            min_yield=min_yield, max_price=max_price,
            min_beds=min_beds, min_score=min_score,
            preferred_locations=sel_locations, preferred_types=sel_types,
            sort_by=sort_by,
        )

        if recs.empty:
            st.warning("No properties match the current filters. Try relaxing your criteria.")
        else:
            ra, rb, rc = st.columns(3)
            with ra:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{len(recs)}</div>
                    <div class="metric-label">Matching Properties</div>
                </div>""", unsafe_allow_html=True)
            with rb:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">${recs['price'].median()/1000:.0f}K</div>
                    <div class="metric-label">Median Price</div>
                </div>""", unsafe_allow_html=True)
            with rc:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{recs['investment_score'].max():.1f}</div>
                    <div class="metric-label">Best Score</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(fig_top_properties(recs), use_container_width=True)
            st.plotly_chart(fig_roi_comparison(recs), use_container_width=True)

            # Display table
            display_cols = [
                c for c in [
                    "property_id","location","property_type","bedrooms","bathrooms",
                    "price","cap_rate","gross_rental_yield","cash_on_cash_return",
                    "investment_score","amenities_score",
                ] if c in recs.columns
            ]
            styled = recs[display_cols].head(20).copy()
            styled["price"] = styled["price"].apply(lambda x: f"${x:,.0f}")
            styled["cap_rate"] = styled["cap_rate"].apply(lambda x: f"{x*100:.2f}%")
            styled["gross_rental_yield"] = styled["gross_rental_yield"].apply(lambda x: f"{x*100:.2f}%")
            styled["cash_on_cash_return"] = styled["cash_on_cash_return"].apply(lambda x: f"{x*100:.2f}%")
            styled["investment_score"] = styled["investment_score"].apply(lambda x: f"{x:.1f}")
            st.dataframe(styled, use_container_width=True)

            csv = recs.to_csv(index=False).encode()
            st.download_button(
                "⬇️  Export Recommendations as CSV",
                csv, "recommendations.csv", "text/csv"
            )

    # ── TAB 3: ANALYTICS ──────────────────────
    with tab3:
        st.markdown('<div class="section-title">Deep Analytics</div>', unsafe_allow_html=True)

        st.plotly_chart(fig_cap_vs_yield(df), use_container_width=True)

        r1, r2 = st.columns(2)
        with r1:
            fig = px.box(df, x="property_type", y="cap_rate",
                         color="property_type", color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOTLY_LAYOUT, title="Cap Rate by Property Type", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            fig = px.box(df, x="location", y="gross_rental_yield",
                         color="location", color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOTLY_LAYOUT, title="Gross Yield by Location", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Price per sqft
        fig = px.scatter(
            df, x="property_age", y="price_per_sqft",
            color="property_type", trendline="ols",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Property Age vs Price per Sq Ft")
        st.plotly_chart(fig, use_container_width=True)

        # Investment score treemap
        fig = px.treemap(
            df, path=["location","property_type"],
            values="investment_score",
            color="cap_rate",
            color_continuous_scale=["#1e2d40","#e5b94e"],
        )
        fig.update_layout(margin=dict(t=40, b=20, l=0, r=0),
                          paper_bgcolor=DARK_BG, font_color=TEXT,
                          title="Investment Score Treemap by Location & Type")
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: RAW DATA ───────────────────────
    with tab4:
        st.markdown('<div class="section-title">Full Dataset</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        csv_all = df.to_csv(index=False).encode()
        st.download_button("⬇️  Download Full Dataset", csv_all, "full_dataset.csv", "text/csv")


if __name__ == "__main__":
    main()