"""Adaptive Recommender Streamlit app (knowledge/content/collaborative + hybrid fallback)."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Default dataset (same folder as this app) — used for Streamlit Cloud / local deployment.
DATA_CSV = Path(__file__).resolve().parent / "data.csv"


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PropIQ — Adaptive Recommender",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e4dc; }
[data-testid="stSidebar"] { background: #16191f; border-right: 1px solid #2a2d35; }
[data-testid="stSidebar"] * { color: #c9c5bc !important; }

.hero {
    background: linear-gradient(135deg, #1a1d26 0%, #12151d 60%, #0d1018 100%);
    border: 1px solid #2a2d35;
    border-radius: 16px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #f5f0e8;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #8a8680;
    font-size: 1rem;
    margin-top: 0.35rem;
    font-weight: 300;
}
.accent { color: #e5b94e; }

.metric-card {
    background: #16191f;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #e5b94e;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #8a8680;
    margin-top: 0.35rem;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #f5f0e8;
    border-left: 3px solid #e5b94e;
    padding-left: 0.75rem;
    margin: 1.3rem 0 0.7rem 0;
}
[data-testid="stDataFrame"] { border: 1px solid #2a2d35; border-radius: 10px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

DARK_BG = "#13161d"
GRID = "#2a2d35"
ACCENT = "#e5b94e"
TEXT = "#c9c5bc"
PALETTE = ["#e5b94e", "#60a5fa", "#4ade80", "#f87171", "#a78bfa", "#fb923c"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(family="DM Sans", color=TEXT, size=12),
    xaxis=dict(gridcolor=GRID, showline=False),
    yaxis=dict(gridcolor=GRID, showline=False),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)


@dataclass
class SchemaInfo:
    user_id: Optional[str]
    item_id: str
    item_name: str
    rating: Optional[str]
    interaction: Optional[str]
    price: Optional[str]
    location: Optional[str]
    neighborhood: Optional[str]
    category: Optional[str]
    bedrooms: Optional[str]
    size_col: Optional[str]
    text_cols: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]
    methods_available: Dict[str, bool]


@dataclass
class UserPreferences:
    """Structured filters for knowledge-based property matching."""

    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_beds: Optional[float] = None
    max_beds: Optional[float] = None
    cities: List[str] = field(default_factory=list)
    neighborhoods: List[str] = field(default_factory=list)
    property_types: List[str] = field(default_factory=list)
    min_sqft: Optional[float] = None
    max_sqft: Optional[float] = None
    strict_budget: bool = True
    strict_beds: bool = False
    strict_area: bool = False


def snake_case(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return re.sub(r"_+", "_", col).strip("_")


def dedupe_column_names(cols: List[str]) -> List[str]:
    """Ensure unique column names after snake_case (PyArrow / Streamlit cannot handle duplicates)."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        base = c or "column"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def unique_ordered(cols: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in cols:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate column labels and ensure Arrow-serializable display."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def choose_item_id_column(cols: List[str], excluded: Optional[List[str]] = None) -> Optional[str]:
    excluded_set = set(excluded or [])
    priority = [
        "item_id",
        "property_id",
        "listing_id",
        "listing_url",
        "url",
        "product_id",
        "asset_id",
        "house_id",
        "state_id",
    ]
    for name in priority:
        if name in cols and name not in excluded_set:
            return name
    for col in cols:
        if col in excluded_set:
            continue
        if col == "id" or col.endswith("_id"):
            return col
    return None


def text_clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set:
    if not text:
        return set()
    return {tok for tok in text.split() if len(tok) > 2}


def robust_read_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, encoding="latin1")
        except Exception as ex:
            raise ValueError(f"Unable to parse CSV: {ex}") from ex


def read_local_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def generate_synthetic_data(n_items: int = 80, n_users: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    states = ["California", "Texas", "Florida", "New York", "Arizona", "Colorado", "Nevada", "Georgia"]
    property_types = ["Apartment", "House", "Condo", "Townhouse"]
    records = []
    for user in range(1, n_users + 1):
        seen = rng.choice(np.arange(1, n_items + 1), size=rng.integers(12, 22), replace=False)
        for item in seen:
            state = rng.choice(states)
            ptype = rng.choice(property_types)
            price = int(rng.integers(150_000, 1_100_000))
            beds = int(rng.integers(1, 6))
            baths = int(rng.integers(1, 4))
            sqft = int(rng.integers(700, 3600))
            cap_rate = float(rng.uniform(0.03, 0.11))
            rating = float(np.clip((cap_rate * 35) + rng.normal(3.5, 0.7), 1, 5))
            desc = f"{ptype} in {state} with {beds} bed and {baths} bath, near schools and transit."
            records.append(
                {
                    "user_id": user,
                    "property_id": item,
                    "state_name": state,
                    "property_type": ptype,
                    "price": price,
                    "bedrooms": beds,
                    "bathrooms": baths,
                    "square_footage": sqft,
                    "cap_rate": round(cap_rate, 4),
                    "vacancy_rate": round(float(rng.uniform(0.01, 0.12)), 4),
                    "amenities_score": int(rng.integers(1, 11)),
                    "rating": round(rating, 2),
                    "description": desc,
                }
            )
    return pd.DataFrame(records)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = dedupe_column_names([snake_case(c) for c in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype != "object":
            continue
        raw = df[col].astype(str).str.strip()
        cleaned = raw.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("%", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() >= 0.8 and raw.nunique(dropna=True) > 6:
            df[col] = numeric
        else:
            df[col] = raw.replace({"nan": np.nan, "none": np.nan, "": np.nan})

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        mode_val = df[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
        df[col] = df[col].fillna(fill_val).astype(str)

    return df


def choose_col(cols: List[str], keywords: List[str], excluded: Optional[List[str]] = None) -> Optional[str]:
    """Map logical roles to column names; exact matches first, then substring (guarded for short keywords)."""
    excluded = excluded or []
    for kw in keywords:
        for col in cols:
            if col in excluded:
                continue
            if col == kw:
                return col
    for kw in keywords:
        for col in cols:
            if col in excluded:
                continue
            if kw not in col:
                continue
            if kw == "id" and col != "id" and not col.endswith("_id"):
                continue
            if kw == "type" and col not in {"type", "property_type", "item_type", "listing_type"} and not col.endswith("_type"):
                continue
            return col
    return None


def infer_schema(df: pd.DataFrame) -> SchemaInfo:
    cols = list(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    user_id = choose_col(cols, ["user_id", "userid", "customer_id", "client_id", "investor_id", "member_id"])
    item_id = choose_item_id_column(cols, excluded=[user_id] if user_id else [])
    if item_id is None:
        item_id = "item_id"
        df[item_id] = np.arange(1, len(df) + 1)

    neighborhood = choose_col(
        cols,
        ["neighborhood", "locality", "sector", "society", "suburb", "micro_market"],
        excluded=[user_id, item_id] if user_id else [item_id],
    )
    location = choose_col(
        cols,
        ["city", "metro", "state_name", "state", "location", "region", "area"],
        excluded=[x for x in [user_id, item_id, neighborhood] if x],
    )

    item_name = choose_col(
        cols,
        ["listing_title", "item_name", "property_name", "listing_name", "title", "name"],
        excluded=[x for x in [user_id, item_id] if x],
    )
    if item_name is None:
        item_name = choose_col(
            cols,
            ["type", "property_type", "neighborhood", "city"],
            excluded=[x for x in [user_id, item_id, location, neighborhood] if x],
        )
    if item_name is None:
        item_name = item_id
    if item_name == location and neighborhood and item_name != neighborhood:
        item_name = neighborhood
    elif item_name == location:
        alt = choose_col(cols, ["type", "property_type", "neighborhood", "url"], excluded=[location, user_id, item_id] if user_id else [location, item_id])
        item_name = alt if alt else item_id

    rating = choose_col(cols, ["rating", "score", "preference_score", "satisfaction", "stars"])
    interaction = choose_col(
        cols,
        ["interaction", "clicks", "purchases", "views", "bookmarks", "engagement"],
        excluded=[rating] if rating else [],
    )
    price = choose_col(cols, ["price", "cost", "amount", "budget", "value", "rent", "monthly_rent"])
    category = choose_col(
        cols,
        ["property_type", "item_type", "category", "segment", "tags", "type", "listing_type"],
        excluded=[x for x in [user_id, item_id, item_name] if x],
    )
    bedrooms = choose_col(cols, ["beds", "bedrooms", "bhk", "bedroom", "no_of_bedrooms", "rooms"])
    size_col = choose_col(cols, ["size", "sqft", "square_feet", "square_footage", "area", "built_up_area", "carpet_area"])

    text_cols = [
        c
        for c in object_cols
        if c not in {item_name, location, neighborhood, category, item_id}
        and df[c].astype(str).str.len().mean() > 18
    ]
    categorical_cols = [
        c for c in object_cols if c not in text_cols and c not in {item_name} and c != item_id
    ]

    collab_possible = user_id is not None and item_id is not None and ((rating in numeric_cols) or interaction is not None)
    content_possible = len(categorical_cols) > 0 or len(text_cols) > 0 or len(numeric_cols) >= 2
    knowledge_possible = (
        (price is not None)
        or (category is not None)
        or (location is not None)
        or (neighborhood is not None)
        or (bedrooms is not None)
        or (len(numeric_cols) > 0)
    )

    return SchemaInfo(
        user_id=user_id,
        item_id=item_id,
        item_name=item_name,
        rating=rating if rating in numeric_cols else None,
        interaction=interaction,
        price=price,
        location=location,
        neighborhood=neighborhood,
        category=category,
        bedrooms=bedrooms,
        size_col=size_col,
        text_cols=text_cols[:4],
        categorical_cols=categorical_cols[:8],
        numeric_cols=numeric_cols,
        methods_available={
            "knowledge": knowledge_possible,
            "content": content_possible,
            "collaborative": collab_possible,
        },
    )


def extract_sqft_estimate(val: object) -> float:
    """Parse strings like '1085 sqft', '799-1258 sqft' into a single numeric estimate."""
    try:
        if val is None:
            return np.nan
        if isinstance(val, (bool, np.bool_)):
            return np.nan
        if isinstance(val, (int, np.integer)):
            return float(val)
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                return np.nan
            return float(val)
        if pd.isna(val):
            return np.nan
        s = str(val).lower().replace(",", "").strip()
        if s in {"", "nan", "none", "<na>", "nat"}:
            return np.nan
        nums = re.findall(r"[\d.]+", s)
        if not nums:
            return np.nan
        floats = [float(x) for x in nums]
        return float(np.median(floats))
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _safe_sqft_series(series: pd.Series) -> pd.Series:
    """Build numeric sqft column without Series.map (PyArrow string arrays can break .map in pandas 2.x)."""
    out: List[float] = []
    for v in series.tolist():
        out.append(extract_sqft_estimate(v))
    return pd.Series(out, index=series.index, dtype="float64")


def engineer_real_estate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "type" in df.columns and "neighborhood" in df.columns:
        df["listing_title"] = (
            df["type"].astype(str).str.strip() + " — " + df["neighborhood"].astype(str).str.strip()
        )
    elif "type" in df.columns and "city" in df.columns:
        df["listing_title"] = df["type"].astype(str).str.strip() + " — " + df["city"].astype(str).str.strip()
    elif "url" in df.columns:
        df["listing_title"] = df["url"].astype(str).str.slice(0, 96)

    size_candidates = ["size", "sqft", "square_feet", "square_footage", "built_up_area", "carpet_area", "area"]
    for sc in size_candidates:
        if sc not in df.columns:
            continue
        try:
            df["size_sqft_est"] = _safe_sqft_series(df[sc])
            break
        except Exception:
            continue

    if "price" in df.columns and "rental_income_potential" in df.columns:
        df["gross_rental_yield"] = (df["rental_income_potential"] * 12) / df["price"].replace(0, np.nan)
    if "price" in df.columns and "cap_rate" not in df.columns and "gross_rental_yield" in df.columns:
        df["cap_rate"] = df["gross_rental_yield"].clip(0, 0.25)
    if "year_built" in df.columns:
        current_year = datetime.now().year
        df["property_age"] = np.maximum(0, current_year - pd.to_numeric(df["year_built"], errors="coerce").fillna(current_year))
    if "price" in df.columns and "square_footage" in df.columns:
        sqft = pd.to_numeric(df["square_footage"], errors="coerce").replace(0, np.nan)
        df["price_per_sqft"] = df["price"] / sqft
    if "vacancy_rate" in df.columns and "gross_rental_yield" in df.columns:
        df["effective_yield"] = df["gross_rental_yield"] * (1 - df["vacancy_rate"])

    scoring_cols = [c for c in ["cap_rate", "gross_rental_yield", "effective_yield", "amenities_score"] if c in df.columns]
    if scoring_cols:
        score = pd.Series(0.0, index=df.index)
        for col in scoring_cols:
            series = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if col in df else 0)
            denom = (series.max() - series.min()) or 1.0
            score += (series - series.min()) / denom
        df["investment_score"] = (100 * score / len(scoring_cols)).round(2)

    return df


def build_item_catalog(df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
    item_id = schema.item_id
    if item_id not in df.columns:
        df[item_id] = np.arange(1, len(df) + 1)

    agg_map = {}
    for col in df.columns:
        if col == item_id:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = "mean"
        else:
            agg_map[col] = lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else s.iloc[0]

    catalog = df.groupby(item_id, as_index=False).agg(agg_map)
    catalog[schema.item_name] = catalog[schema.item_name].astype(str)
    return catalog


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    denom = s.max() - s.min()
    if denom == 0 or pd.isna(denom):
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / denom


def _range_fit(series: pd.Series, lo: float, hi: float, strict: bool) -> Tuple[pd.Series, pd.Series]:
    """Return (fit 0-1, soft reason mask) for numeric preference range."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if strict:
        inside = ((s >= lo) & (s <= hi)).astype(float)
        return inside, inside > 0
    mid = (lo + hi) / 2.0
    half = max((hi - lo) / 2.0, 1e-6)
    dist = (s - mid).abs()
    fit = (1.0 - (dist / (half + 1e-6)).clip(0, 1)).fillna(0.0)
    return fit, fit > 0.35


def _property_type_match(series: pd.Series, selected: List[str]) -> pd.Series:
    """Fuzzy match listing type strings against user-selected labels (e.g. 'Villa', 'BHK Apartment')."""
    if not selected:
        return pd.Series(0.0, index=series.index)
    sel = [text_clean(s) for s in selected if s]
    out = pd.Series(0.0, index=series.index)
    for idx, raw in series.items():
        t = text_clean(raw)
        if not t:
            continue
        best = 0.0
        for lab in sel:
            if not lab:
                continue
            if lab in t or t in lab:
                best = 1.0
                break
            lt, tt = tokenize(lab), tokenize(t)
            if lt and tt:
                inter = len(lt & tt)
                union = len(lt | tt) or 1
                best = max(best, inter / union)
        out.loc[idx] = best
    return out


def knowledge_scores(
    items: pd.DataFrame,
    schema: SchemaInfo,
    prefs: UserPreferences,
    ranking_col: Optional[str],
) -> Tuple[pd.Series, pd.Series]:
    score = pd.Series(0.0, index=items.index)
    reason_parts = pd.Series("", index=items.index)
    total_weight = 0.0

    if schema.price and schema.price in items.columns and prefs.min_price is not None and prefs.max_price is not None:
        price_col = items[schema.price]
        pfit, good = _range_fit(price_col, float(prefs.min_price), float(prefs.max_price), prefs.strict_budget)
        w = 0.28
        score += w * pfit
        reason_parts += np.where(good, "price-fit; ", "")
        total_weight += w

    if schema.bedrooms and schema.bedrooms in items.columns and prefs.min_beds is not None and prefs.max_beds is not None:
        bfit, good = _range_fit(items[schema.bedrooms], float(prefs.min_beds), float(prefs.max_beds), prefs.strict_beds)
        w = 0.20
        score += w * bfit
        reason_parts += np.where(good, "bedroom-fit; ", "")
        total_weight += w

    if "size_sqft_est" in items.columns and prefs.min_sqft is not None and prefs.max_sqft is not None:
        afit, good = _range_fit(
            items["size_sqft_est"], float(prefs.min_sqft), float(prefs.max_sqft), prefs.strict_area
        )
        w = 0.15
        score += w * afit
        reason_parts += np.where(good, "area-fit; ", "")
        total_weight += w

    if schema.location and schema.location in items.columns and prefs.cities:
        loc_match = items[schema.location].astype(str).isin(prefs.cities).astype(float)
        w = 0.18
        score += w * loc_match
        reason_parts += np.where(loc_match > 0, "city-match; ", "")
        total_weight += w

    if schema.neighborhood and schema.neighborhood in items.columns and prefs.neighborhoods:
        nh_match = items[schema.neighborhood].astype(str).isin(prefs.neighborhoods).astype(float)
        w = 0.15
        score += w * nh_match
        reason_parts += np.where(nh_match > 0, "neighborhood-match; ", "")
        total_weight += w

    if schema.category and schema.category in items.columns and prefs.property_types:
        tmatch = _property_type_match(items[schema.category], prefs.property_types)
        w = 0.25
        score += w * tmatch
        reason_parts += np.where(tmatch > 0.55, "property-type-match; ", "")
        total_weight += w

    if ranking_col and ranking_col in items.columns and pd.api.types.is_numeric_dtype(items[ranking_col]):
        quality = minmax(items[ranking_col])
        w = 0.10
        score += w * quality
        reason_parts += np.where(quality > 0.7, f"high-{ranking_col}; ", "")
        total_weight += w

    if total_weight == 0:
        return pd.Series(0.5, index=items.index), pd.Series("baseline-ranking", index=items.index)

    final = (score / total_weight).clip(0, 1)
    reasons = reason_parts.str.strip().str.rstrip(";")
    reasons = reasons.replace("", "preference-match")
    return final, reasons


def content_scores(items: pd.DataFrame, schema: SchemaInfo, anchor_item_id: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    if anchor_item_id is None or schema.item_id not in items.columns:
        popularity = pd.Series(0.5, index=items.index)
        return popularity, pd.Series("content-baseline", index=items.index)

    anchor_match = items[items[schema.item_id].astype(str) == str(anchor_item_id)]
    if anchor_match.empty:
        return pd.Series(0.5, index=items.index), pd.Series("content-baseline", index=items.index)

    anchor = anchor_match.iloc[0]

    cat_cols = [c for c in schema.categorical_cols if c in items.columns and c != schema.item_name][:5]
    num_cols = [c for c in schema.numeric_cols if c in items.columns and c not in {schema.item_id}][:6]
    txt_cols = [c for c in schema.text_cols if c in items.columns][:3]

    cat_score = pd.Series(0.0, index=items.index)
    if cat_cols:
        for col in cat_cols:
            cat_score += (items[col].astype(str) == str(anchor[col])).astype(float)
        cat_score = cat_score / len(cat_cols)

    num_score = pd.Series(0.0, index=items.index)
    if num_cols:
        tmp = pd.DataFrame(index=items.index)
        for col in num_cols:
            rng = (items[col].max() - items[col].min()) or 1.0
            tmp[col] = 1 - ((items[col].astype(float) - float(anchor[col])).abs() / rng).clip(0, 1)
        num_score = tmp.mean(axis=1)

    txt_score = pd.Series(0.0, index=items.index)
    if txt_cols:
        anchor_tokens = tokenize(" ".join(text_clean(anchor[c]) for c in txt_cols))

        def jaccard(row_text: str) -> float:
            row_tokens = tokenize(row_text)
            if not row_tokens or not anchor_tokens:
                return 0.0
            union = row_tokens | anchor_tokens
            return len(row_tokens & anchor_tokens) / len(union) if union else 0.0

        item_text = items[txt_cols].astype(str).agg(" ".join, axis=1).map(text_clean)
        txt_score = item_text.map(jaccard)

    # Weighted feature similarity across available modalities.
    weights = []
    weighted = pd.Series(0.0, index=items.index)
    if cat_cols:
        weighted += 0.4 * cat_score
        weights.append(0.4)
    if num_cols:
        weighted += 0.4 * num_score
        weights.append(0.4)
    if txt_cols:
        weighted += 0.2 * txt_score
        weights.append(0.2)

    if not weights:
        return pd.Series(0.5, index=items.index), pd.Series("content-baseline", index=items.index)

    final = (weighted / sum(weights)).clip(0, 1)
    final.loc[items[schema.item_id].astype(str) == str(anchor_item_id)] = 0
    reasons = np.where(final > 0.7, f"similar-to-{anchor_item_id}", "content-similarity")
    return final, pd.Series(reasons, index=items.index)


def collaborative_scores(df: pd.DataFrame, items: pd.DataFrame, schema: SchemaInfo, target_user: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    if not schema.methods_available["collaborative"] or target_user is None:
        return pd.Series(0.5, index=items.index), pd.Series("collab-baseline", index=items.index)

    user_col = schema.user_id
    item_col = schema.item_id
    value_col = schema.rating if schema.rating else schema.interaction

    matrix = (
        df.pivot_table(index=user_col, columns=item_col, values=value_col, aggfunc="mean", fill_value=0)
        .astype(float)
        .sort_index(axis=1)
    )
    if matrix.empty:
        return pd.Series(0.5, index=items.index), pd.Series("collab-baseline", index=items.index)

    if target_user not in matrix.index.astype(str).tolist():
        popularity = matrix.replace(0, np.nan).mean(axis=0).fillna(0)
        pop_norm = minmax(popularity)
        map_pop = items[schema.item_id].map(pop_norm.to_dict()).fillna(0.2)
        return map_pop, pd.Series("popular-among-users", index=items.index)

    matrix.index = matrix.index.astype(str)
    target_vec = matrix.loc[str(target_user)].values
    all_mat = matrix.values

    dot = all_mat @ target_vec
    norm_target = np.linalg.norm(target_vec)
    norms = np.linalg.norm(all_mat, axis=1)
    sim = dot / ((norms * norm_target) + 1e-9)
    sim_series = pd.Series(sim, index=matrix.index)
    sim_series.loc[str(target_user)] = 0
    sim_series = sim_series.clip(lower=0)

    weighted_sum = np.zeros(matrix.shape[1])
    sim_sum = np.zeros(matrix.shape[1])
    for usr, s in sim_series.items():
        if s <= 0:
            continue
        vec = matrix.loc[usr].values
        observed = vec > 0
        weighted_sum[observed] += s * vec[observed]
        sim_sum[observed] += s

    pred = np.divide(weighted_sum, sim_sum, out=np.zeros_like(weighted_sum), where=sim_sum > 0)
    pred_series = pd.Series(pred, index=matrix.columns)

    already_seen = matrix.loc[str(target_user)] > 0
    pred_series.loc[already_seen.values] = 0

    collab_norm = minmax(pred_series)
    mapped = items[schema.item_id].map(collab_norm.to_dict()).fillna(0.0)
    reasons = np.where(mapped > 0.6, "liked-by-similar-users", "collaborative-signal")
    return mapped, pd.Series(reasons, index=items.index)


def select_effective_method(method_choice: str, schema: SchemaInfo) -> str:
    available = schema.methods_available
    count = sum(1 for v in available.values() if v)

    if method_choice == "Auto":
        # Listing-only property CSVs (no user–item interactions): knowledge filters + optional content anchor.
        if available["knowledge"] and not available["collaborative"]:
            return "Knowledge-Based"
        if count >= 2:
            return "Hybrid"
        if available["collaborative"]:
            return "Collaborative"
        if available["content"]:
            return "Content-Based"
        return "Knowledge-Based"

    if method_choice == "Hybrid" and count >= 2:
        return "Hybrid"
    if method_choice == "Collaborative" and available["collaborative"]:
        return "Collaborative"
    if method_choice == "Content-Based" and available["content"]:
        return "Content-Based"
    if method_choice == "Knowledge-Based" and available["knowledge"]:
        return "Knowledge-Based"

    return select_effective_method("Auto", schema)


def run_recommender(
    df: pd.DataFrame,
    schema: SchemaInfo,
    method: str,
    top_n: int,
    prefs: UserPreferences,
    ranking_col: Optional[str],
    anchor_item_id: Optional[str],
    target_user: Optional[str],
) -> pd.DataFrame:
    items = build_item_catalog(df, schema).copy()

    k_score, k_reason = knowledge_scores(items, schema, prefs, ranking_col)
    c_score, c_reason = content_scores(items, schema, anchor_item_id)
    cf_score, cf_reason = collaborative_scores(df, items, schema, target_user)

    weights = {"knowledge": 0.0, "content": 0.0, "collab": 0.0}
    if method == "Knowledge-Based":
        weights["knowledge"] = 1.0
    elif method == "Content-Based":
        weights["content"] = 1.0
    elif method == "Collaborative":
        weights["collab"] = 1.0
    else:
        availability = schema.methods_available
        if availability["knowledge"]:
            weights["knowledge"] = 0.35
        if availability["content"]:
            weights["content"] = 0.35
        if availability["collaborative"]:
            weights["collab"] = 0.30
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

    final_score = (
        weights["knowledge"] * k_score
        + weights["content"] * c_score
        + weights["collab"] * cf_score
    ).clip(0, 1)

    items["knowledge_score"] = (k_score * 100).round(2)
    items["content_score"] = (c_score * 100).round(2)
    items["collab_score"] = (cf_score * 100).round(2)
    items["match_score"] = (final_score * 100).round(2)

    reason = []
    for idx in items.index:
        parts = []
        if weights["knowledge"] > 0 and k_score.loc[idx] >= 0.55:
            parts.append(k_reason.loc[idx])
        if weights["content"] > 0 and c_score.loc[idx] >= 0.55:
            parts.append(c_reason.loc[idx])
        if weights["collab"] > 0 and cf_score.loc[idx] >= 0.55:
            parts.append(cf_reason.loc[idx])
        reason.append("; ".join([p for p in parts if p]) if parts else "ranked-by-combined-signal")
    items["explanation"] = reason

    ranked = items.sort_values("match_score", ascending=False)
    return ranked.head(top_n).reset_index(drop=True)


def plottable_numeric_columns(df: pd.DataFrame, cols: List[str], min_valid: int = 2) -> List[str]:
    """Keep only columns that coerce to finite floats with enough points (uploaded CSVs often mismatch inferred dtypes)."""
    out: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() >= min_valid:
            out.append(c)
    return out


def fig_histogram(df: pd.DataFrame, col: str, title: str):
    x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    plot_df = pd.DataFrame({col: x}).dropna(subset=[col])
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric values to plot for this column.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT),
        )
        fig.update_layout(**PLOTLY_LAYOUT, title=title)
        return fig
    fig = px.histogram(plot_df, x=col, nbins=30, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    return fig


def fig_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str], title: str):
    d = df[[x, y]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])

    color_col = color if color and color in df.columns and color not in (x, y) else None
    if color_col is not None:
        d[color_col] = df.loc[d.index, color_col]
        # Plotly discrete color: avoid unhashable / mixed types and huge legend spam.
        nu = d[color_col].nunique(dropna=True)
        if nu == 0 or nu > 48:
            color_col = None
        elif pd.api.types.is_numeric_dtype(d[color_col]):
            if d[color_col].notna().sum() == 0:
                color_col = None
            elif nu <= 12:
                d[color_col] = d[color_col].astype(str)
        else:
            d[color_col] = d[color_col].astype(str)

    if len(d) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numeric x/y pairs for a scatter plot (check uploaded columns).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT),
        )
        fig.update_layout(**PLOTLY_LAYOUT, title=title)
        return fig

    fig = px.scatter(
        d,
        x=x,
        y=y,
        color=color_col,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    return fig


def fig_reco_bar(recs: pd.DataFrame, schema: SchemaInfo):
    y_col = schema.item_name if schema.item_name in recs.columns else schema.item_id
    plot_df = recs.sort_values("match_score", ascending=True).copy()
    plot_df[y_col] = plot_df[y_col].astype(str).str.slice(0, 100)
    fig = px.bar(
        plot_df,
        x="match_score",
        y=y_col,
        orientation="h",
        color="match_score",
        color_continuous_scale=["#2b3440", "#e5b94e"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Top Recommendations by Match Score", coloraxis_showscale=False)
    return fig


def format_summary_card(value: str, label: str) -> str:
    return (
        f"""<div class="metric-card"><div class="metric-value">{value}</div>"""
        f"""<div class="metric-label">{label}</div></div>"""
    )


def main():
    st.markdown(
        """
    <div class="hero">
        <div class="hero-title">Prop<span class="accent">IQ</span></div>
        <div class="hero-sub">Dynamic Knowledge + Content + Collaborative Recommender</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Data source")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
        if uploaded is not None:
            use_uploaded = st.toggle("Use uploaded CSV instead of data.csv", value=True)
        else:
            use_uploaded = False

        try:
            if uploaded is not None and use_uploaded:
                df_raw = robust_read_csv(uploaded)
                data_note = f"Uploaded dataset: {len(df_raw):,} rows."
            elif DATA_CSV.exists():
                df_raw = read_local_csv(DATA_CSV)
                data_note = f"Loaded `{DATA_CSV.name}`: {len(df_raw):,} rows."
            else:
                df_raw = generate_synthetic_data()
                data_note = "Using synthetic demo data (place `data.csv` next to the app for production data)."
        except Exception as ex:
            st.error(str(ex))
            return

        df = normalize_dataframe(df_raw)
        df = engineer_real_estate_features(df)
        schema = infer_schema(df)

        st.success(data_note)
        st.markdown("---")
        st.markdown("### Your property preferences")

        method_choice = st.selectbox(
            "Recommendation strategy",
            ["Auto", "Hybrid", "Knowledge-Based", "Content-Based", "Collaborative"],
            help="Auto uses knowledge-based ranking for listing-only datasets (typical property CSVs).",
        )
        effective_method = select_effective_method(method_choice, schema)

        top_n = st.slider("How many listings to show", min_value=3, max_value=50, value=12)

        price_col = schema.price if schema.price in df.columns else None
        min_price_v = max_price_v = None
        if price_col:
            pser = pd.to_numeric(df[price_col], errors="coerce")
            if pser.notna().sum() == 0:
                price_col = None
        if price_col:
            pmin = float(pser.min())
            pmax = float(pser.max())
            if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin >= pmax:
                price_col = None
        if price_col:
            mid_low = float(pser.quantile(0.15))
            mid_high = float(pser.quantile(0.85))
            st.caption("Budget (price range)")
            cpa, cpb = st.columns(2)
            with cpa:
                min_price_v = st.number_input("Min price", min_value=pmin, max_value=pmax, value=mid_low, format="%.0f")
            with cpb:
                max_price_v = st.number_input("Max price", min_value=pmin, max_value=pmax, value=mid_high, format="%.0f")
            strict_budget = st.checkbox("Strict budget (exclude listings fully outside range)", value=False)

        else:
            strict_budget = False

        min_beds_v = max_beds_v = None
        if schema.bedrooms and schema.bedrooms in df.columns:
            b = pd.to_numeric(df[schema.bedrooms], errors="coerce").dropna()
            if len(b):
                bmin, bmax = float(b.min()), float(b.max())
                st.caption("Bedrooms (BHK)")
                cba, cbb = st.columns(2)
                with cba:
                    min_beds_v = st.number_input("Min beds", min_value=bmin, max_value=bmax, value=bmin)
                with cbb:
                    max_beds_v = st.number_input("Max beds", min_value=bmin, max_value=bmax, value=bmax)
                strict_beds = st.checkbox("Strict bedroom count", value=False)
            else:
                strict_beds = False
        else:
            strict_beds = False

        cities_sel: List[str] = []
        if schema.location and schema.location in df.columns:
            loc_options = sorted(df[schema.location].astype(str).unique().tolist())
            cities_sel = st.multiselect("City / region (optional)", loc_options)

        nh_sel: List[str] = []
        if schema.neighborhood and schema.neighborhood in df.columns:
            nh_options = sorted(df[schema.neighborhood].astype(str).unique().tolist())
            nh_sel = st.multiselect("Neighborhood / locality (optional)", nh_options)

        types_sel: List[str] = []
        if schema.category and schema.category in df.columns:
            type_options = sorted(df[schema.category].astype(str).unique().tolist())
            types_sel = st.multiselect(
                "Property type / listing label (optional, partial match)",
                type_options,
                help="Matches text in the listing type column (e.g. Apartment, Villa, BHK).",
            )

        min_sqft_v = max_sqft_v = None
        strict_area = False
        if "size_sqft_est" in df.columns:
            sq = pd.to_numeric(df["size_sqft_est"], errors="coerce").dropna()
            if len(sq) >= 1:
                smin = float(sq.quantile(0.05))
                smax = float(sq.quantile(0.95))
                if smin < smax and np.isfinite(smin) and np.isfinite(smax):
                    st.caption("Area (estimated sq.ft from size text)")
                    csa, csb = st.columns(2)
                    with csa:
                        min_sqft_v = st.number_input("Min sq.ft", min_value=0.0, max_value=float(smax * 3), value=smin, step=50.0)
                    with csb:
                        max_sqft_v = st.number_input("Max sq.ft", min_value=0.0, max_value=float(smax * 3), value=smax, step=50.0)
                    strict_area = st.checkbox("Strict area range", value=False)

        ranking_candidates = [c for c in schema.numeric_cols if c not in {schema.item_id} and c in df.columns]
        if "size_sqft_est" in df.columns and "size_sqft_est" not in ranking_candidates:
            ranking_candidates = ["size_sqft_est"] + ranking_candidates
        ranking_col = st.selectbox("Extra ranking emphasis (optional numeric)", ranking_candidates) if ranking_candidates else None

        anchor_item_id = None
        if schema.methods_available["content"]:
            item_options = df[schema.item_id].astype(str).dropna().unique().tolist()
            anchor_item_id = st.selectbox(
                "Similar to listing (optional — content-based)",
                options=[None] + item_options[:500],
                format_func=lambda x: "None (preference-only)" if x is None else str(x)[:72],
            )

        target_user = None
        if schema.methods_available["collaborative"] and schema.user_id in df.columns:
            user_options = df[schema.user_id].astype(str).dropna().unique().tolist()
            target_user = st.selectbox("Target user ID", options=user_options)

        prefs = UserPreferences(
            min_price=min_price_v,
            max_price=max_price_v,
            min_beds=min_beds_v,
            max_beds=max_beds_v,
            cities=cities_sel,
            neighborhoods=nh_sel,
            property_types=types_sel,
            min_sqft=min_sqft_v,
            max_sqft=max_sqft_v,
            strict_budget=strict_budget,
            strict_beds=strict_beds,
            strict_area=strict_area,
        )

        st.markdown("---")
        run = st.button("Generate recommendations", use_container_width=True)

    available_text = ", ".join([k for k, v in schema.methods_available.items() if v]) or "none"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(format_summary_card(f"{len(df):,}", "rows"), unsafe_allow_html=True)
    with c2:
        st.markdown(format_summary_card(f"{df.shape[1]:,}", "columns"), unsafe_allow_html=True)
    with c3:
        st.markdown(format_summary_card(str(df[schema.item_id].nunique()), "unique items"), unsafe_allow_html=True)
    with c4:
        st.markdown(format_summary_card(available_text, "methods detected"), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Analysis", "🏆 Recommendations", "📈 Analytics", "🗂️ Raw Data"])

    with tab1:
        st.markdown('<div class="section-title">Automatic Dataset Analysis</div>', unsafe_allow_html=True)
        st.write(
            {
                "detected_user_id": schema.user_id,
                "detected_item_id": schema.item_id,
                "detected_item_name": schema.item_name,
                "detected_rating": schema.rating,
                "detected_interaction": schema.interaction,
                "detected_price": schema.price,
                "detected_bedrooms": schema.bedrooms,
                "detected_location": schema.location,
                "detected_neighborhood": schema.neighborhood,
                "detected_category": schema.category,
                "detected_size": schema.size_col,
                "text_features": schema.text_cols,
                "numeric_features_count": len(schema.numeric_cols),
                "categorical_features_count": len(schema.categorical_cols),
                "selected_method": effective_method,
            }
        )
        st.dataframe(arrow_safe_dataframe(df.head(20)), use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Recommendation Results</div>', unsafe_allow_html=True)
        if not run:
            st.info("Set your budget, bedrooms, and optional filters in the sidebar, then generate recommendations.")
        else:
            try:
                if prefs.min_price is not None and prefs.max_price is not None and prefs.min_price > prefs.max_price:
                    prefs.min_price, prefs.max_price = prefs.max_price, prefs.min_price
                if prefs.min_beds is not None and prefs.max_beds is not None and prefs.min_beds > prefs.max_beds:
                    prefs.min_beds, prefs.max_beds = prefs.max_beds, prefs.min_beds
                if prefs.min_sqft is not None and prefs.max_sqft is not None and prefs.min_sqft > prefs.max_sqft:
                    prefs.min_sqft, prefs.max_sqft = prefs.max_sqft, prefs.min_sqft

                recs = run_recommender(
                    df=df,
                    schema=schema,
                    method=effective_method,
                    top_n=top_n,
                    prefs=prefs,
                    ranking_col=ranking_col,
                    anchor_item_id=anchor_item_id,
                    target_user=target_user,
                )
            except Exception as ex:
                st.error(f"Recommendation step failed: {ex}")
                recs = pd.DataFrame()

            if recs.empty:
                st.warning("No recommendations available. Widen your price or bedroom range or clear optional filters.")
            else:
                st.success(f"{len(recs)} listings ranked with `{effective_method}`. Match % is higher when more preferences align.")
                st.caption(
                    "Knowledge scores reflect your filters (budget, beds, city, neighborhood, listing type). "
                    "Content/collaborative signals apply when you choose a reference listing or user data exists."
                )
                try:
                    st.plotly_chart(fig_reco_bar(recs, schema), use_container_width=True)
                except Exception as chart_ex:
                    st.caption(f"Chart skipped: {chart_ex}")

                display_cols = unique_ordered(
                    [
                        c
                        for c in [
                            schema.item_name,
                            schema.item_id,
                            schema.category,
                            schema.bedrooms,
                            schema.price,
                            schema.size_col,
                            "size_sqft_est",
                            schema.location,
                            schema.neighborhood,
                            "match_score",
                            "knowledge_score",
                            "content_score",
                            "collab_score",
                            "explanation",
                        ]
                        if c and c in recs.columns
                    ]
                )
                out = recs[display_cols].copy()
                if schema.price and schema.price in out.columns:
                    out[schema.price] = out[schema.price].map(lambda x: f"₹{x:,.0f}" if pd.notna(x) else x)
                if "match_score" in out.columns:
                    out["match_score"] = out["match_score"].map(lambda x: f"{x:.1f}%")
                st.dataframe(arrow_safe_dataframe(out), use_container_width=True)
                st.download_button(
                    "Export recommendations as CSV",
                    data=recs.to_csv(index=False).encode("utf-8"),
                    file_name="recommendations.csv",
                    mime="text/csv",
                )

    with tab3:
        st.markdown('<div class="section-title">Exploratory Analytics</div>', unsafe_allow_html=True)
        num_cols = plottable_numeric_columns(df, [c for c in schema.numeric_cols if c in df.columns])
        cat_cols = [c for c in schema.categorical_cols if c in df.columns]

        if num_cols:
            st.plotly_chart(fig_histogram(df, num_cols[0], f"Distribution: {num_cols[0]}"), use_container_width=True)

        if len(num_cols) >= 2:
            color_col = cat_cols[0] if cat_cols else None
            st.plotly_chart(
                fig_scatter(df, num_cols[0], num_cols[1], color=color_col, title=f"{num_cols[0]} vs {num_cols[1]}"),
                use_container_width=True,
            )

        if len(num_cols) >= 2:
            corr_cols = num_cols[:8]
            corr = df[corr_cols].corr().round(2)
            heat = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale="RdBu",
                    zmid=0,
                    text=corr.values,
                    texttemplate="%{text}",
                )
            )
            heat.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation Heatmap", height=460)
            st.plotly_chart(heat, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-title">Full Processed Dataset</div>', unsafe_allow_html=True)
        st.dataframe(arrow_safe_dataframe(df), use_container_width=True)
        st.download_button(
            "⬇️ Download Processed Dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="processed_dataset.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()