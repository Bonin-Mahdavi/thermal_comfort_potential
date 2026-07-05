"""Shared helpers for RQ1 and RQ2 notebooks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

import config as cfg


# ---------------------------------------------------------------------------
# Paths & I/O
# ---------------------------------------------------------------------------


def apply_plot_theme() -> None:
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.1,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
    )


def save_table(name: str, df: pd.DataFrame) -> Path:
    """Write a DataFrame to ``tables/{name}.xlsx``."""
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.TABLES_DIR / f"{name}.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def save_figure(fig: plt.Figure, name: str) -> Path:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.FIGURES_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Questionnaire loading & cleaning
# ---------------------------------------------------------------------------


def _export_url(sheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"


def _fetch_sheets(sheet_id: str) -> dict[str, pd.DataFrame]:
    return pd.read_excel(_export_url(sheet_id), sheet_name=None)


def _rename_columns(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    mapping = cfg.SHEET_COLUMN_MAPS[sheet_name]
    missing = set(df.columns) - set(mapping)
    if missing:
        raise ValueError(f"{sheet_name}: unmapped columns: {missing}")
    return df.rename(columns=mapping)


def _prepare_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    meta = cfg.SHEET_METADATA[sheet_name]
    renamed = _rename_columns(df, sheet_name)
    renamed["questionnaire_city"] = meta["questionnaire_city"]
    renamed["language"] = meta["language"]
    return renamed[cfg.MERGED_COLUMNS]


def _merge_sheets(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [_prepare_sheet(df, name) for name, df in sheets.items()]
    return pd.concat(frames, ignore_index=True)


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def _standardize_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in cfg.VALUE_MAPS.items():
        if col in out.columns:
            out[col] = out[col].replace(mapping)
    for col in cfg.POINT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].replace(cfg.COMFORT_SCALE_MAP)
    return _strip_strings(out)


def _fix_misplaced_summer_description(df: pd.DataFrame) -> pd.DataFrame:
    """Move climate labels from summer_visit into summer_description when mis-keyed."""
    out = df.copy()
    misplaced = out["summer_visit"].isin(cfg.SUMMER_DESCRIPTION_EN_LABELS) & out[
        "summer_description"
    ].isna()
    if not misplaced.any():
        return out
    out.loc[misplaced, "summer_description"] = out.loc[misplaced, "summer_visit"]
    out.loc[misplaced, "summer_visit"] = cfg.SUMMER_VISIT_WHEN_DESCRIPTION_MISPLACED
    return out


def _clean_birthplace(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def categorize(value):
        if pd.isna(value):
            return value
        key = value.strip() if isinstance(value, str) else value
        return cfg.BIRTHPLACE_MAP.get(key, "other_place")

    out["birthplace"] = out["birthplace"].apply(categorize)
    return out


def _normalize_occupation(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate occupation labels before keyword encoding."""
    if "occupation" not in df.columns:
        return df
    out = df.copy()
    out["occupation"] = out["occupation"].replace(cfg.QUESTIONNAIRE_OCCUPATION_LABEL_ALIASES)
    return out


def _keyword_value(col: str, value):
    if pd.isna(value):
        return value
    if col == "age" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return cfg.AGE_NUMERIC_KEYWORDS.get(int(value), "unknown")
    if col in cfg.POINT_COLUMNS:
        return cfg.COMFORT_KEYWORDS.get(value, pd.NA)
    mapping = cfg.KEYWORD_MAPS.get(col, {})
    return mapping.get(value, value if col == "birthplace" else "unknown")


def _apply_keywords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keyword_cols = set(cfg.KEYWORD_MAPS) | set(cfg.POINT_COLUMNS)
    for col in keyword_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda v, c=col: _keyword_value(c, v))
    for col in cfg.POINT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("Int64")
    return out


def load_cleaned_data(sheet_id: str | None = None) -> pd.DataFrame:
    sheet_id = sheet_id or cfg.QUESTIONNAIRE_SHEET_ID
    sheets = _fetch_sheets(sheet_id)
    unknown = set(sheets) - set(cfg.SHEET_COLUMN_MAPS)
    if unknown:
        raise ValueError(f"Unknown sheet tabs: {unknown}")

    merged = _merge_sheets(sheets)
    standardized = _fix_misplaced_summer_description(_standardize_values(merged))
    standardized = _normalize_occupation(standardized)
    cleaned = _apply_keywords(_clean_birthplace(standardized))
    cleaned = cleaned.drop(columns=["consent", "comments", "timestamp"], errors="ignore")
    return cleaned[cfg.MERGED_COLUMNS]


def load_survey() -> pd.DataFrame:
    return load_cleaned_data()


@dataclass
class AttributeBundle:
    """Attribute table loaded from Google Sheets with layer assignments."""

    df: pd.DataFrame
    layers_all: dict[str, list[str]]


def _attribute_export_url() -> str:
    return f"https://docs.google.com/spreadsheets/d/{cfg.SHEET_ID}/export?format=xlsx"


def parse_attribute_layers(group_row: pd.Series, column_row: pd.Series) -> dict[str, list[str]]:
    """Map merged group headers (row 1) to column names (row 2)."""
    layers: dict[str, list[str]] = {}
    current_layer: str | None = None

    for group_label, col_name in zip(group_row, column_row):
        if pd.notna(group_label) and str(group_label).strip():
            current_layer = str(group_label).strip()
        if pd.isna(col_name):
            continue
        col = str(col_name).strip()
        if not col or current_layer is None:
            continue
        if current_layer.lower().startswith("point ident"):
            continue
        layers.setdefault(current_layer, []).append(col)

    return layers


def load_attributes() -> AttributeBundle:
    """Load attribute table from Google Sheets and parse layer groups from the header rows."""
    raw = pd.read_excel(
        _attribute_export_url(),
        sheet_name=cfg.SHEET_TAB,
        header=None,
    )
    group_row = raw.iloc[cfg.GROUP_HEADER_ROW]
    column_row = raw.iloc[cfg.HEADER_ROW]
    layers_all = parse_attribute_layers(group_row, column_row)

    df = raw.iloc[cfg.HEADER_ROW + 1 :].copy()
    df.columns = [str(c).strip() if pd.notna(c) else "" for c in column_row]
    df = df.loc[df["Point ID"].notna()].copy()
    df["Point ID"] = df["Point ID"].astype(str).str.strip()
    df = df.set_index("Point ID")

    for col in df.columns:
        if col.endswith("(%)") or col in cfg.SVI_PROPORTION_COLUMNS:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().le(1.0).all():
                df[col] = df[col] * 100.0

    return AttributeBundle(df=df, layers_all=layers_all)


def comfort_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in cfg.POINT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Load survey and attribute tables (call after apply_plot_theme)."""
    df_survey = comfort_numeric(load_cleaned_data())
    bundle = load_attributes()
    return df_survey, bundle.df, bundle.layers_all


# ---------------------------------------------------------------------------
# Attribute screening (variation & collinearity helpers)
# ---------------------------------------------------------------------------


def normalized_shannon_entropy(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) == 0:
        return np.nan
    counts = values.astype(str).value_counts()
    k = len(counts)
    if k <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / np.log2(k))


def _layer_for_attribute(attribute: str, layers: dict[str, list[str]]) -> str:
    for layer_name, cols in layers.items():
        if attribute in cols:
            return layer_name
    return ""


def screening_mu_sigma_bounds(
    values: pd.Series,
    *,
    std_multiplier: float | None = None,
) -> tuple[float, float, float, float]:
    """Return μ, σ, and μ ± k·σ bounds for a screening distribution."""
    if std_multiplier is None:
        std_multiplier = cfg.SCREENING_STD_MULTIPLIER
    mu = float(values.mean())
    sigma = float(values.std(ddof=1))
    band = std_multiplier * sigma
    return mu, sigma, mu - band, mu + band


def _attribute_h_norm(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return normalized_shannon_entropy(numeric.dropna())
    return normalized_shannon_entropy(series.dropna())


_VARIATION_APPLIED_RULE_LABELS = {
    "constant": "Constant",
    "near_absent": "Near absent",
    "low_entropy": "Low entropy",
    "kept": "Kept",
}


def _format_applied_rules(
    failures: list[str],
    labels: dict[str, str],
) -> str:
    if not failures:
        return labels["kept"]
    return ", ".join(labels[rule] for rule in failures)


def _attribute_screening_rules(
    *,
    n_unique: int,
    n_nonzero: int,
    h_norm: float,
    min_nonzero_points: int | None = None,
) -> tuple[bool, str]:
    min_nonzero = (
        cfg.ENTROPY_MIN_NONZERO_POINTS
        if min_nonzero_points is None
        else min_nonzero_points
    )
    failures: list[str] = []
    if n_unique < cfg.ENTROPY_MIN_UNIQUE_VALUES:
        failures.append("constant")
    if n_nonzero < min_nonzero:
        failures.append("near_absent")
    if h_norm < cfg.ATTRIBUTE_MIN_H_NORM:
        failures.append("low_entropy")
    return not failures, _format_applied_rules(failures, _VARIATION_APPLIED_RULE_LABELS)


def format_entropy_screening_table(quality: pd.DataFrame) -> pd.DataFrame:
    """Public variation-screening table for reports and CSV export."""
    ordered = quality.sort_values(["kept", "attribute"], ascending=[True, True])
    return pd.DataFrame(
        {
            "Layer": ordered["layer"].values,
            "Attribute": ordered["attribute"].values,
            "Applied Rule": ordered["applied_rule"].values,
        }
    )


def attribute_quality_report(
    df_attr: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    layers: dict[str, list[str]] | None = None,
    min_nonzero_points: int | None = None,
) -> pd.DataFrame:
    layers = layers or cfg.ATTRIBUTE_LAYERS_ALL
    columns = columns or all_physical_predictors(layers)
    rows = []
    for col in columns:
        if col not in df_attr.columns:
            continue
        s = df_attr[col]
        numeric = pd.to_numeric(s, errors="coerce")
        n_nonzero = int((numeric.fillna(0) != 0).sum()) if numeric.notna().any() else int(s.notna().sum())
        n_unique = int(s.nunique(dropna=True))
        h_norm = _attribute_h_norm(s)
        kept, applied_rule = _attribute_screening_rules(
            n_unique=n_unique,
            n_nonzero=n_nonzero,
            h_norm=h_norm,
            min_nonzero_points=min_nonzero_points,
        )
        rows.append(
            {
                "attribute": col,
                "layer": _layer_for_attribute(col, layers),
                "H_norm": h_norm,
                "applied_rule": applied_rule,
                "kept": kept,
            }
        )
    report = pd.DataFrame(rows)
    return report.sort_values(["kept", "attribute"], ascending=[True, True])


def attribute_quality_report_by_city(
    df_attr: pd.DataFrame,
    *,
    layers: dict[str, list[str]] | None = None,
    min_nonzero_points: int | None = None,
) -> pd.DataFrame:
    """Variation screening on each city's ten survey points separately."""
    min_nonzero = (
        cfg.ENTROPY_MIN_NONZERO_POINTS_CITY
        if min_nonzero_points is None
        else min_nonzero_points
    )
    frames: list[pd.DataFrame] = []
    for city in cfg.CITIES:
        sub = filter_attributes_by_city(df_attr, city)
        rep = attribute_quality_report(
            sub,
            layers=layers,
            min_nonzero_points=min_nonzero,
        )
        frames.append(rep.assign(city=city))
    return pd.concat(frames, ignore_index=True)


def intersection_kept_items(
    quality_by_city: pd.DataFrame,
    *,
    city_col: str = "city",
    item_col: str = "attribute",
) -> set[str]:
    """Predictors that pass screening in every city."""
    kept_sets = [
        set(
            quality_by_city.loc[
                (quality_by_city[city_col] == city) & quality_by_city["kept"],
                item_col,
            ]
        )
        for city in cfg.CITIES
    ]
    if not kept_sets:
        return set()
    return set.intersection(*kept_sets)


def city_variation_removals(
    quality_by_city: pd.DataFrame,
    *,
    city_col: str = "city",
    item_col: str = "attribute",
) -> dict[str, list[str]]:
    """Items failing variation screening in each city alone."""
    return {
        city: sorted(
            quality_by_city.loc[
                (quality_by_city[city_col] == city) & ~quality_by_city["kept"],
                item_col,
            ].tolist()
        )
        for city in cfg.CITIES
    }


def city_kept_items(
    quality_by_city: pd.DataFrame,
    city: str,
    *,
    city_col: str = "city",
    item_col: str = "attribute",
) -> list[str]:
    """Items passing variation screening in one city."""
    return sorted(
        quality_by_city.loc[
            (quality_by_city[city_col] == city) & quality_by_city["kept"],
            item_col,
        ].tolist()
    )


def build_screened_attribute_tables_by_city(
    df_attr: pd.DataFrame,
    quality_by_city: pd.DataFrame,
    layers: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """Per-city attribute tables with §1 variation-screening removals applied."""
    screened: dict[str, pd.DataFrame] = {}
    for city in cfg.CITIES:
        kept = city_kept_items(quality_by_city, city, item_col="attribute")
        city_layers = filter_layers_by_columns(layers, set(kept))
        table = filter_attribute_dataframe(
            filter_attributes_by_city(df_attr, city),
            city_layers,
        )
        removed = set(city_variation_removals(quality_by_city, item_col="attribute")[city])
        leaked = sorted(removed & set(table.columns))
        if leaked:
            raise ValueError(
                f"{city}: §1 attribute removals still present after screening: {leaked}"
            )
        screened[city] = table
    return screened


def build_screened_questionnaire_columns_by_city(
    quality_by_city: pd.DataFrame,
) -> dict[str, list[str]]:
    """Per-city questionnaire column lists with §1 variation-screening removals applied."""
    screened: dict[str, list[str]] = {}
    for city in cfg.CITIES:
        kept = city_kept_items(quality_by_city, city, item_col="variable")
        removed = set(city_variation_removals(quality_by_city, item_col="variable")[city])
        leaked = sorted(removed & set(kept))
        if leaked:
            raise ValueError(
                f"{city}: §1 questionnaire removals still present after screening: {leaked}"
            )
        screened[city] = kept
    return screened


def format_city_variation_removals_table(
    quality_by_city: pd.DataFrame,
    *,
    city_col: str = "city",
    item_col: str = "attribute",
    layer_col: str | None = "layer",
) -> pd.DataFrame:
    """Long table of city-specific variation-screening removals."""
    dropped = quality_by_city.loc[~quality_by_city["kept"]].copy()
    if dropped.empty:
        return pd.DataFrame(columns=["City", "Layer", "Attribute", "Applied rule"])

    rename = {
        city_col: "City",
        item_col: "Attribute",
        "applied_rule": "Applied rule",
    }
    if layer_col and layer_col in dropped.columns:
        rename[layer_col] = "Layer"
        cols = [city_col, layer_col, item_col, "applied_rule"]
    else:
        cols = [city_col, item_col, "applied_rule"]

    return (
        dropped[cols]
        .rename(columns=rename)
        .sort_values(["City", "Attribute"])
        .reset_index(drop=True)
    )


def format_city_variation_screening_table(
    quality: pd.DataFrame,
    *,
    item_col: str = "attribute",
    layer_col: str = "layer",
) -> pd.DataFrame:
    """Wide city-stratified variation table for reports and CSV export."""
    work = quality.copy()

    layers = (
        work.drop_duplicates(item_col)
        .set_index(item_col)[layer_col]
        .to_dict()
    )
    rules_by_city = {
        city: work.loc[work["city"] == city].set_index(item_col)["applied_rule"].to_dict()
        for city in cfg.CITIES
    }
    attributes = sorted(
        work[item_col].unique(),
        key=lambda name: (layers.get(name, ""), name),
    )
    return pd.DataFrame(
        {
            "Layer": [layers.get(name, "") for name in attributes],
            "Attribute": attributes,
            "Applied Rule to Turin": [
                rules_by_city.get("Turin", {}).get(name, "") for name in attributes
            ],
            "Applied rule to Detmold": [
                rules_by_city.get("Detmold", {}).get(name, "") for name in attributes
            ],
        }
    )


def _group_for_questionnaire_variable(variable: str) -> str:
    for group_name, cols in cfg.QUESTIONNAIRE_SHAP_GROUPS.items():
        if variable in cols:
            return group_name
    return ""


_QUESTIONNAIRE_APPLIED_RULE_LABELS = {
    "constant": "Constant",
    "near_constant": "Near constant",
    "low_entropy": "Low entropy",
    "kept": "Kept",
}


def _questionnaire_screening_rules(
    series: pd.Series,
) -> tuple[bool, str, float, float]:
    """Near-zero variance screen for categorical questionnaire fields (Field, 2018)."""
    values = series.dropna()
    n = len(values)
    if n == 0:
        return False, _QUESTIONNAIRE_APPLIED_RULE_LABELS["constant"], 0.0, 1.0
    n_unique = int(values.nunique())
    h_norm = normalized_shannon_entropy(values)
    top_share = float(values.value_counts().iloc[0] / n)
    failures: list[str] = []
    if n_unique < cfg.QUESTIONNAIRE_MIN_UNIQUE_VALUES:
        failures.append("constant")
    if top_share >= cfg.QUESTIONNAIRE_MAX_DOMINANT_CATEGORY_SHARE:
        failures.append("near_constant")
    if h_norm < cfg.QUESTIONNAIRE_MIN_H_NORM:
        failures.append("low_entropy")
    return (
        not failures,
        _format_applied_rules(failures, _QUESTIONNAIRE_APPLIED_RULE_LABELS),
        h_norm,
        top_share,
    )


def questionnaire_quality_report(
    df_survey: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = columns or [
        c for c in cfg.QUESTIONNAIRE_CATEGORICAL_COLUMNS if c in df_survey.columns
    ]
    rows = []
    for col in columns:
        kept, applied_rule, h_norm, top_share = _questionnaire_screening_rules(
            df_survey[col]
        )
        rows.append(
            {
                "variable": col,
                "group": _group_for_questionnaire_variable(col),
                "H_norm": h_norm,
                "top_category_share": top_share,
                "applied_rule": applied_rule,
                "kept": kept,
            }
        )
    return pd.DataFrame(rows).sort_values(["kept", "variable"], ascending=[True, True])


def questionnaire_quality_report_by_city(
    df_survey: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Variation screening on each city's respondents separately."""
    frames: list[pd.DataFrame] = []
    for city in cfg.CITIES:
        sub = filter_survey_by_city(df_survey, city)
        rep = questionnaire_quality_report(sub, columns=columns)
        frames.append(rep.assign(city=city))
    return pd.concat(frames, ignore_index=True)


def _questionnaire_variable_sort_key(variable: str) -> tuple[int, int, str]:
    """Sort key: SHAP group order, then variable order within group."""
    group = _group_for_questionnaire_variable(variable)
    try:
        group_rank = cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER.index(group)
    except ValueError:
        group_rank = len(cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER)
    try:
        var_rank = cfg.QUESTIONNAIRE_SHAP_GROUPS[group].index(variable)
    except (KeyError, ValueError):
        var_rank = 999
    return group_rank, var_rank, variable


def format_questionnaire_screening_table(quality: pd.DataFrame) -> pd.DataFrame:
    """Wide questionnaire variation-screening table for reports and CSV export."""
    work = quality.copy()

    groups = (
        work.drop_duplicates("variable")
        .set_index("variable")["group"]
        .to_dict()
    )
    rules_by_city = {
        city: work.loc[work["city"] == city].set_index("variable")["applied_rule"].to_dict()
        for city in cfg.CITIES
    }
    variables = sorted(work["variable"].unique(), key=_questionnaire_variable_sort_key)
    return pd.DataFrame(
        {
            "Group": [groups.get(variable, "") for variable in variables],
            "Attribute": [
                cfg.QUESTIONNAIRE_VARIABLE_LABELS.get(variable, variable)
                for variable in variables
            ],
            "Applied Rule to Turin": [
                rules_by_city.get("Turin", {}).get(variable, "") for variable in variables
            ],
            "Applied rule to Detmold": [
                rules_by_city.get("Detmold", {}).get(variable, "") for variable in variables
            ],
        }
    )


def ordinal_encode_questionnaire(
    df_survey: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Encode questionnaire categories for Spearman collinearity screening."""
    encoded: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df_survey.columns:
            continue
        series = df_survey[col]
        order = cfg.COVARIATE_CATEGORY_ORDERS.get(col)
        if order:
            mapping = {value: idx for idx, value in enumerate(order)}
            encoded[col] = series.map(mapping)
        else:
            categories = sorted(series.dropna().astype(str).unique())
            mapping = {value: idx for idx, value in enumerate(categories)}
            encoded[col] = series.astype(str).map(mapping)
    return pd.DataFrame(encoded)


def questionnaire_spearman_correlation(
    df_survey: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Pairwise Spearman ρ among questionnaire covariates (ordinal encoding)."""
    encoded = ordinal_encode_questionnaire(df_survey, columns)
    return encoded.corr(method="spearman")


def screened_questionnaire_columns_for_city(
    city: str,
    *,
    variation_quality: pd.DataFrame | None = None,
    collinearity_drops: dict[str, list[str]] | None = None,
) -> list[str]:
    """Questionnaire fields retained after city-specific variation and collinearity screens."""
    collinearity_drops = collinearity_drops or cfg.QUESTIONNAIRE_COLLINEARITY_MANUAL_DROPS
    if variation_quality is None:
        return [
            c
            for c in cfg.QUESTIONNAIRE_CATEGORICAL_COLUMNS
            if c not in collinearity_drops.get(city, [])
        ]
    kept = variation_quality.loc[
        (variation_quality["city"] == city) & variation_quality["kept"],
        "variable",
    ].tolist()
    drops = set(collinearity_drops.get(city, []))
    return [c for c in kept if c not in drops]


def questionnaire_model_columns_by_city(
    df_survey: pd.DataFrame,
    *,
    variation_quality: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Questionnaire fields retained after Preprocessing §1–§2 for each city."""
    if variation_quality is None:
        variation_quality = questionnaire_quality_report_by_city(df_survey)
    return {
        city: screened_questionnaire_columns_for_city(
            city,
            variation_quality=variation_quality,
        )
        for city in cfg.CITIES
    }


def ordered_questionnaire_profile_columns(
    columns_by_city: dict[str, list[str]],
) -> list[str]:
    """Union of city questionnaire columns in questionnaire-group display order."""
    union = {col for cols in columns_by_city.values() for col in cols}
    ordered: list[str] = []
    for group_cols in cfg.QUESTIONNAIRE_SHAP_GROUPS.values():
        for col in group_cols:
            if col in union and col not in ordered:
                ordered.append(col)
    for col in cfg.RQ5_CATEGORY_COLUMNS:
        if col in union and col not in ordered:
            ordered.append(col)
    return ordered


def format_questionnaire_collinearity_pairs_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Flagged questionnaire covariate pairs with human-readable variable labels."""
    table = format_collinearity_pairs_table(
        pairs,
        left_col="Variable",
        right_col="Partner variable",
        flagged_abs_rho_threshold=cfg.QUESTIONNAIRE_COLLINEARITY_FLAGGED_ABS_RHO_THRESHOLD,
    )
    for col in ("Variable", "Partner variable"):
        table[col] = table[col].map(cfg.QUESTIONNAIRE_VARIABLE_LABELS).fillna(table[col])
    return table


def filter_layers_by_columns(
    layers: dict[str, list[str]],
    keep: set[str],
) -> dict[str, list[str]]:
    return {name: [c for c in cols if c in keep] for name, cols in layers.items()}


def all_physical_predictors(layers: dict[str, list[str]] | None = None) -> list[str]:
    preds: list[str] = []
    for cols in (layers or cfg.ATTRIBUTE_LAYERS_ALL).values():
        preds.extend(cols)
    return preds


def prepare_attribute_layers(
    df_attr: pd.DataFrame,
    *,
    layers_all: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]], pd.DataFrame]:
    layers_all = layers_all or cfg.ATTRIBUTE_LAYERS_ALL
    report = attribute_quality_report(
        df_attr,
        layers=layers_all,
    )
    keep = set(report.loc[report["kept"], "attribute"])
    filtered_all = filter_layers_by_columns(layers_all, keep)
    filtered_main = filter_layers_by_columns(cfg.ATTRIBUTE_LAYERS_MAIN, keep)
    return filtered_all, filtered_main, report


def attribute_entropy_screening_report(quality: pd.DataFrame) -> str:
    """Interpretive conclusion for the variation-based attribute screening step."""
    dropped = quality.loc[~quality["kept"]]
    n_kept = int(quality["kept"].sum())
    min_nonzero = cfg.ENTROPY_MIN_NONZERO_POINTS
    min_unique = cfg.ENTROPY_MIN_UNIQUE_VALUES
    if dropped.empty:
        return (
            "Conclusion — uninformative attribute screening:\n"
            f"All {n_kept} candidate streetscape attributes passed the variation screen "
            f"(n_unique ≥ {min_unique}, n_nonzero ≥ {min_nonzero}) and were retained "
            "for later modelling."
        )
    reason_labels = {
        "constant": f"constant across points (n_unique < {min_unique})",
        "near_absent": f"near-absent across points (n_nonzero < {min_nonzero})",
        "low_entropy": (
            f"low normalized Shannon entropy (H_norm < {cfg.ATTRIBUTE_MIN_H_NORM:g})"
        ),
    }
    removed_lines = "\n".join(
        f"- {row.attribute}: {reason_labels.get(row.applied_rule, row.applied_rule)}"
        for row in dropped.itertuples()
    )
    return (
        "Conclusion — uninformative attribute screening:\n"
        "Following standard data-exploration practice (Zuur et al., 2010) and "
        "place-comparison streetscape studies (Zhu et al., 2026), attributes were "
        "removed only when they were constant across points or near-absent at fewer "
        "than three of twenty survey locations:\n"
        f"{removed_lines}\n"
        f"{n_kept} predictors retained for collinearity screening and RQ2 modelling."
    )


def plot_screening_mu_sigma_distribution(
    summary: pd.DataFrame,
    *,
    value_col: str,
    kept_col: str,
    mu_col: str,
    sigma_col: str,
    lower_col: str,
    upper_col: str,
    xlabel: str,
    title: str,
) -> plt.Figure:
    """KDE of a screening metric with μ ± σ bounds and kept/removed rug marks."""
    values = summary[value_col]
    kept = summary[kept_col]
    mu = float(summary[mu_col].iloc[0])
    lower = float(summary[lower_col].iloc[0])
    upper = float(summary[upper_col].iloc[0])
    sigma = float(summary[sigma_col].iloc[0])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.kdeplot(
        values,
        ax=ax,
        color="#4C72B0",
        fill=True,
        alpha=cfg.KDE_FILL_ALPHA,
        linewidth=2.0,
        zorder=3,
    )
    sns.rugplot(values[kept], ax=ax, color="#4C72B0", height=0.045, alpha=0.9, label="Kept")
    sns.rugplot(
        values[~kept],
        ax=ax,
        color="#C44E52",
        height=0.06,
        alpha=0.95,
        label="Removed",
    )

    k = cfg.SCREENING_STD_MULTIPLIER
    ax.axvline(mu, color="#212121", linestyle="-", linewidth=1.4, zorder=2, label=f"μ ({mu:.2f})")
    ax.axvline(
        lower,
        color="#757575",
        linestyle="--",
        linewidth=1.3,
        zorder=2,
        label=f"μ − {k:g}σ ({lower:.2f})",
    )
    ax.axvline(
        upper,
        color="#757575",
        linestyle="--",
        linewidth=1.3,
        zorder=2,
        label=f"μ + {k:g}σ ({upper:.2f})",
    )

    ax.set_xlabel(f"{xlabel} (σ = {sigma:.2f})")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="semibold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig


def plot_attribute_entropy_distribution(
    quality: pd.DataFrame,
    *,
    title: str = "Normalized Shannon entropy across streetscape attributes",
) -> plt.Figure:
    """KDE of normalized Shannon entropy (H_norm) with kept/removed rug marks."""
    if "H_norm" not in quality.columns:
        raise ValueError("quality DataFrame must include H_norm for this plot")
    values = quality["H_norm"]
    kept = quality["kept"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.kdeplot(
        values,
        ax=ax,
        color="#4C72B0",
        fill=True,
        alpha=cfg.KDE_FILL_ALPHA,
        linewidth=2.0,
        zorder=3,
    )
    sns.rugplot(values[kept], ax=ax, color="#4C72B0", height=0.045, alpha=0.9, label="Kept")
    sns.rugplot(
        values[~kept],
        ax=ax,
        color="#C44E52",
        height=0.06,
        alpha=0.95,
        label="Removed",
    )
    ax.set_xlabel("Normalized Shannon entropy (H_norm)")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.02, 1.05)
    ax.set_title(title, fontweight="semibold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig


def filter_attribute_dataframe(
    df_attr: pd.DataFrame,
    layers_all: dict[str, list[str]],
) -> pd.DataFrame:
    """Keep screened physical predictors only (no point metadata such as LCZ or coordinates)."""
    predictors = all_physical_predictors(layers_all)
    cols = [c for c in predictors if c in df_attr.columns]
    out = df_attr[cols].copy()
    if df_attr.index.name:
        out.index = df_attr.index
    return out


def collinearity_groups(columns: list[str], corr: pd.DataFrame, threshold: float = 0.85) -> list[list[str]]:
    parent = {c: c for c in columns}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            if i < j and a in corr.columns and b in corr.columns:
                rho = corr.loc[a, b]
                if pd.notna(rho) and abs(float(rho)) >= threshold:
                    union(a, b)

    buckets: dict[str, list[str]] = {}
    for c in columns:
        buckets.setdefault(find(c), []).append(c)
    return [sorted(g) for g in buckets.values() if len(g) > 1]


def select_representatives(groups: list[list[str]], corr: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group in groups:
        rep = group[0]
        best = -1.0
        for col in group:
            others = [c for c in group if c != col]
            if not others:
                rep = col
                break
            mean_abs = float(corr.loc[col, others].abs().mean())
            if mean_abs > best:
                best = mean_abs
                rep = col
        for col in group:
            mapping[col] = rep
    return mapping


def cross_layer_sum_correlation_table(
    cross_corr: pd.DataFrame,
    *,
    layer1_name: str,
    layer3a_name: str,
) -> pd.DataFrame:
    """Sum of Spearman ρ of each variable against all variables in the opposite layer."""
    rows: list[dict[str, object]] = []
    for var in cross_corr.index:
        rows.append(
            {
                "attribute": var,
                "layer": layer1_name,
                "sum_cross_rho": float(cross_corr.loc[var].sum()),
                "n_opposite_layer": int(cross_corr.shape[1]),
            }
        )
    for var in cross_corr.columns:
        rows.append(
            {
                "attribute": var,
                "layer": layer3a_name,
                "sum_cross_rho": float(cross_corr[var].sum()),
                "n_opposite_layer": int(cross_corr.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def collinearity_group_top_correlations(
    corr: pd.DataFrame,
    groups: list[list[str]],
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top pairwise Spearman ρ partners for each attribute in collinearity groups."""
    rows: list[dict[str, object]] = []
    for group_id, group in enumerate(groups, start=1):
        for var in group:
            partners = [c for c in corr.columns if c != var]
            rhos = corr.loc[var, partners].dropna()
            top = rhos.reindex(rhos.abs().sort_values(ascending=False).index).head(top_n)
            for rank, (partner, rho) in enumerate(top.items(), start=1):
                rows.append(
                    {
                        "group_id": group_id,
                        "attribute": var,
                        "rank": rank,
                        "correlated_with": partner,
                        "spearman_rho": float(rho),
                    }
                )
    return pd.DataFrame(rows)


def matrix_abs_rho_sum(
    corr: pd.DataFrame,
    attribute: str,
    predictors: list[str],
) -> float:
    """Sum of |Spearman ρ| between one attribute and all other predictors in the full matrix."""
    partners = [c for c in predictors if c != attribute and c in corr.columns]
    return float(corr.loc[attribute, partners].abs().sum())


def matrix_rho_sum(
    corr: pd.DataFrame,
    attribute: str,
    predictors: list[str],
) -> float:
    """Sum of Spearman ρ (signed) between one attribute and all other predictors."""
    partners = [c for c in predictors if c != attribute and c in corr.columns]
    return float(corr.loc[attribute, partners].sum())


def matrix_collinearity_rho_sum(
    corr: pd.DataFrame,
    attribute: str,
    predictors: list[str],
    *,
    use_signed_rho_sum: bool = False,
) -> float:
    """Σρ or Σ|ρ| for collinearity screening."""
    if use_signed_rho_sum:
        return matrix_rho_sum(corr, attribute, predictors)
    return matrix_abs_rho_sum(corr, attribute, predictors)


def collinearity_rho_sum_label(*, use_signed_rho_sum: bool) -> str:
    return "sum_rho" if use_signed_rho_sum else "sum_abs_rho"


def collinearity_rho_sum_symbol(*, use_signed_rho_sum: bool) -> str:
    return "Σρ" if use_signed_rho_sum else "Σ|ρ|"


def collinearity_rho_sum_distribution_stats(
    values: pd.Series,
    *,
    std_multiplier: float | None = None,
) -> tuple[float, float, float, float]:
    """Return μ, σ, and μ ± k·σ bounds for a Σρ distribution."""
    return screening_mu_sigma_bounds(values, std_multiplier=std_multiplier)


def matrix_top_correlations(
    corr: pd.DataFrame,
    attribute: str,
    predictors: list[str],
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top |Spearman ρ| partners for an attribute in the full correlation matrix."""
    partners = [c for c in predictors if c != attribute and c in corr.columns]
    rhos = corr.loc[attribute, partners].dropna()
    top = rhos.reindex(rhos.abs().sort_values(ascending=False).index).head(top_n)
    rows: list[dict[str, object]] = []
    for rank, (partner, rho) in enumerate(top.items(), start=1):
        rows.append(
            {
                "attribute": attribute,
                "rank": rank,
                "correlated_with": partner,
                "spearman_rho": float(rho),
            }
        )
    return pd.DataFrame(rows)


def matrix_abs_rho_sum_table(
    corr: pd.DataFrame,
    predictors: list[str],
    layers: dict[str, list[str]],
) -> pd.DataFrame:
    """Sum of |ρ| in the full matrix for each Layer 1 predictor."""
    layer1_name, layer1, _, _ = layer_predictor_groups(layers, predictors)
    rows = [
        {
            "attribute": var,
            "layer": layer1_name,
            "sum_abs_rho": matrix_abs_rho_sum(corr, var, predictors),
            "n_partners": len(predictors) - 1,
        }
        for var in layer1
        if var in predictors
    ]
    return pd.DataFrame(rows)


def cross_layer_top_correlations(
    cross_corr: pd.DataFrame,
    attribute: str,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top |Spearman ρ| cross-layer partners for a Layer 1 or Layer 3a attribute."""
    if attribute in cross_corr.index:
        rhos = cross_corr.loc[attribute].dropna()
        partner_layer = "Layer 3a — SVI pixel %"
    elif attribute in cross_corr.columns:
        rhos = cross_corr[attribute].dropna()
        partner_layer = "Layer 1 — RS / GIS"
    else:
        return pd.DataFrame(
            columns=["attribute", "partner_layer", "rank", "correlated_with", "spearman_rho"]
        )

    top = rhos.reindex(rhos.abs().sort_values(ascending=False).index).head(top_n)
    rows: list[dict[str, object]] = []
    for rank, (partner, rho) in enumerate(top.items(), start=1):
        rows.append(
            {
                "attribute": attribute,
                "partner_layer": partner_layer,
                "rank": rank,
                "correlated_with": partner,
                "spearman_rho": float(rho),
            }
        )
    return pd.DataFrame(rows)


def vegetation_collinearity_evidence(
    corr: pd.DataFrame,
    *,
    vegetation: str = "Vegetation (%)",
    partners: tuple[str, ...] = ("Building (%)", "NDVI_mean"),
) -> pd.DataFrame:
    """Spearman ρ between Vegetation and its within-layer redundancy partners."""
    rows: list[dict[str, object]] = []
    for partner in partners:
        if vegetation in corr.index and partner in corr.columns:
            rows.append(
                {
                    "attribute": vegetation,
                    "correlated_with": partner,
                    "spearman_rho": float(corr.loc[vegetation, partner]),
                }
            )
    return pd.DataFrame(rows)


def screen_attribute_predictors(
    predictors: list[str],
    corr: pd.DataFrame,
    cross_corr: pd.DataFrame | None = None,
    layers: dict[str, list[str]] | None = None,
    *,
    use_signed_rho_sum: bool = False,
    std_multiplier: float | None = None,
    sum_rho_threshold: float | None = None,
    sum_abs_rho_threshold: float | None = None,
    **_: object,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop predictors whose Σρ or Σ|ρ| lies outside μ ± k·σ of the distribution.
    """
    del cross_corr, layers, sum_rho_threshold, sum_abs_rho_threshold
    sum_col = collinearity_rho_sum_label(use_signed_rho_sum=use_signed_rho_sum)
    sums = {
        var: matrix_collinearity_rho_sum(
            corr, var, predictors, use_signed_rho_sum=use_signed_rho_sum
        )
        for var in predictors
    }
    sum_series = pd.Series(sums)
    mu, sigma, lower, upper = screening_mu_sigma_bounds(
        sum_series,
        std_multiplier=std_multiplier,
    )
    drop_set = {
        var for var, value in sums.items() if value < lower or value > upper
    }
    screening = pd.DataFrame(
        {
            "attribute": predictors,
            sum_col: [round(sums[var], 3) for var in predictors],
            "rho_sum_mu": round(mu, 3),
            "rho_sum_sigma": round(sigma, 3),
            "rho_sum_lower": round(lower, 3),
            "rho_sum_upper": round(upper, 3),
            "kept_for_modelling": [var not in drop_set for var in predictors],
        }
    )
    kept_set = {var for var in predictors if var not in drop_set}
    return kept_set, screening, pd.DataFrame(), pd.DataFrame()


def _collinearity_drop_line(
    name: str,
    corr: pd.DataFrame,
    predictors: list[str],
    screening_summary: pd.DataFrame,
    *,
    use_signed_rho_sum: bool,
    top_n: int = 2,
) -> str:
    sum_col = collinearity_rho_sum_label(use_signed_rho_sum=use_signed_rho_sum)
    rho_sum = float(
        screening_summary.loc[screening_summary["attribute"] == name, sum_col].iloc[0]
    )
    top = matrix_top_correlations(corr, name, predictors, top_n=top_n)
    if use_signed_rho_sum:
        partners = ", ".join(
            f"{row.correlated_with} (ρ = {row.spearman_rho:+.3f})"
            for row in top.itertuples()
        )
    else:
        partners = ", ".join(
            f"{row.correlated_with} (|ρ| = {abs(row.spearman_rho):.3f})"
            for row in top.itertuples()
        )
    return (
        f"- {name}: {collinearity_rho_sum_symbol(use_signed_rho_sum=use_signed_rho_sum)} "
        f"= {rho_sum:.3f}; strongest associations with {partners}"
    )


def collinearity_screening_report(
    screening_summary: pd.DataFrame,
    corr: pd.DataFrame,
    predictors: list[str],
    *,
    use_signed_rho_sum: bool,
    top_n: int = 2,
) -> str:
    """Interpretive conclusion for attribute collinearity screening."""
    sum_col = collinearity_rho_sum_label(use_signed_rho_sum=use_signed_rho_sum)
    symbol = collinearity_rho_sum_symbol(use_signed_rho_sum=use_signed_rho_sum)
    dropped = (
        screening_summary.loc[~screening_summary["kept_for_modelling"]]
        .sort_values(sum_col, ascending=False)["attribute"]
        .tolist()
    )
    if not dropped:
        return (
            f"Conclusion — collinearity screening ({symbol}):\n"
            "No predictors were removed for redundancy; the retained attributes provide "
            "sufficiently distinct information for later modelling."
        )
    removed_lines = "\n".join(
        _collinearity_drop_line(
            name,
            corr,
            predictors,
            screening_summary,
            use_signed_rho_sum=use_signed_rho_sum,
            top_n=top_n,
        )
        for name in dropped
    )
    n_kept = int(screening_summary["kept_for_modelling"].sum())
    redundancy_note = _collinearity_redundancy_note(
        screening_summary,
        use_signed_rho_sum=use_signed_rho_sum,
    )
    return (
        f"Conclusion — collinearity screening ({symbol}):\n"
        "The following predictors showed unusually high overall redundancy "
        f"({symbol} outside μ ± σ) and were removed so retained attributes stay "
        "interpretable:\n"
        f"{removed_lines}\n"
        f"{redundancy_note}\n"
        f"The remaining {n_kept} streetscape predictors can be used together in RQ2 "
        "without carrying multiple measures of the same underlying quality."
    )


def _collinearity_redundancy_note(
    screening_summary: pd.DataFrame,
    *,
    use_signed_rho_sum: bool,
) -> str:
    """Summarise how Σρ or Σ|ρ| outliers drove removal."""
    symbol = collinearity_rho_sum_symbol(use_signed_rho_sum=use_signed_rho_sum)
    mu = float(screening_summary["rho_sum_mu"].iloc[0])
    sigma = float(screening_summary["rho_sum_sigma"].iloc[0])
    lower = float(screening_summary["rho_sum_lower"].iloc[0])
    upper = float(screening_summary["rho_sum_upper"].iloc[0])
    multiplier = cfg.SCREENING_STD_MULTIPLIER
    return (
        f"Screening used μ ± {multiplier:g}σ on the per-attribute {symbol} distribution "
        f"([{lower:.2f}, {upper:.2f}]; μ = {mu:.2f}, σ = {sigma:.2f}), reflecting "
        "concentrated associations with other streetscape measures."
    )


def plot_collinearity_rho_sum_distribution(
    screening_summary: pd.DataFrame,
    *,
    use_signed_rho_sum: bool,
    title: str | None = None,
) -> plt.Figure:
    """KDE of Σρ or Σ|ρ| with μ ± k·σ bounds."""
    sum_col = collinearity_rho_sum_label(use_signed_rho_sum=use_signed_rho_sum)
    symbol = collinearity_rho_sum_symbol(use_signed_rho_sum=use_signed_rho_sum)
    if title is None:
        title = f"{symbol} across streetscape predictors"
    return plot_screening_mu_sigma_distribution(
        screening_summary,
        value_col=sum_col,
        kept_col="kept_for_modelling",
        mu_col="rho_sum_mu",
        sigma_col="rho_sum_sigma",
        lower_col="rho_sum_lower",
        upper_col="rho_sum_upper",
        xlabel=f"Net redundancy ({symbol})",
        title=title,
    )


def pairwise_spearman_rho_pairs(
    corr: pd.DataFrame,
    predictors: list[str],
    *,
    min_abs_rho: float | None = None,
) -> pd.DataFrame:
    """All unique predictor pairs; optional |ρ| floor (None keeps every pair)."""
    rows: list[dict[str, object]] = []
    for i, attribute in enumerate(predictors):
        if attribute not in corr.index:
            continue
        for partner in predictors[i + 1 :]:
            if partner not in corr.columns:
                continue
            rho = float(corr.loc[attribute, partner])
            if min_abs_rho is not None and abs(rho) < min_abs_rho:
                continue
            rows.append(
                {
                    "attribute_a": attribute,
                    "attribute_b": partner,
                    "spearman_rho": rho,
                    "abs_rho": abs(rho),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["attribute_a", "attribute_b", "spearman_rho", "abs_rho"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_rho", ascending=False)
        .reset_index(drop=True)
    )


def pairwise_high_abs_rho_pairs(
    corr: pd.DataFrame,
    predictors: list[str],
    *,
    abs_rho_threshold: float | None = None,
) -> pd.DataFrame:
    """Unique predictor pairs with |Spearman ρ| at or above the threshold."""
    if abs_rho_threshold is None:
        abs_rho_threshold = cfg.COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD
    return pairwise_spearman_rho_pairs(
        corr,
        predictors,
        min_abs_rho=abs_rho_threshold,
    )


def _pairwise_flagged_importance(pairs: pd.DataFrame) -> dict[str, tuple[int, float]]:
    """How often each predictor appears among flagged high-|ρ| pairs."""
    counts: dict[str, int] = {}
    sum_abs: dict[str, float] = {}
    for row in pairs.itertuples():
        for attribute in (row.attribute_a, row.attribute_b):
            counts[attribute] = counts.get(attribute, 0) + 1
            sum_abs[attribute] = sum_abs.get(attribute, 0.0) + float(row.abs_rho)
    return {attribute: (counts[attribute], sum_abs[attribute]) for attribute in counts}


def _hub_priority_rank(attribute: str) -> int:
    try:
        return cfg.COLLINEARITY_HUB_PRIORITY.index(attribute)
    except ValueError:
        return len(cfg.COLLINEARITY_HUB_PRIORITY)


def _primary_flagged_attribute(
    left: str,
    right: str,
    importance: dict[str, tuple[int, float]],
) -> tuple[str, str]:
    """Place the more connected predictor in the Attribute column."""
    left_count = importance.get(left, (0, 0.0))[0]
    right_count = importance.get(right, (0, 0.0))[0]
    if left_count > right_count:
        return left, right
    if right_count > left_count:
        return right, left
    left_rank = _hub_priority_rank(left)
    right_rank = _hub_priority_rank(right)
    if left_rank != right_rank:
        return (left, right) if left_rank < right_rank else (right, left)
    return (left, right) if left <= right else (right, left)


def table_name_slug(name: str) -> str:
    """Filesystem-safe slug for table export names."""
    slug = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
    return slug or "attribute"


def _build_collinearity_pair_rows(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[int, float]]]:
    importance = _pairwise_flagged_importance(pairs)
    rows: list[dict[str, object]] = []
    for row in pairs.itertuples():
        attribute, partner = _primary_flagged_attribute(
            row.attribute_a,
            row.attribute_b,
            importance,
        )
        pair_count, _ = importance[attribute]
        rows.append(
            {
                "Attribute": f"{attribute} ({pair_count})",
                "Partner Attribute": partner,
                "Spearman ρ": round(float(row.spearman_rho), 3),
                "|ρ|": round(float(row.abs_rho), 3),
                "_sort_count": pair_count,
                "_sort_hub_rank": _hub_priority_rank(attribute),
                "_sort_attribute": attribute,
            }
        )
    return pd.DataFrame(rows), importance


def _is_svi_pixel_share(attribute: str) -> bool:
    return attribute.endswith(" (%)")


def _orient_flagged_pair(
    attribute_a: str,
    attribute_b: str,
) -> tuple[str, str]:
    """Put the street-view pixel-share (%) variable first when only one is present."""
    a_pct = _is_svi_pixel_share(attribute_a)
    b_pct = _is_svi_pixel_share(attribute_b)
    if a_pct and not b_pct:
        return attribute_a, attribute_b
    if b_pct and not a_pct:
        return attribute_b, attribute_a
    if attribute_a <= attribute_b:
        return attribute_a, attribute_b
    return attribute_b, attribute_a


def format_collinearity_pairs_table(
    pairs: pd.DataFrame,
    *,
    left_col: str = "Attribute",
    right_col: str = "Partner Attribute",
    flagged_abs_rho_threshold: float | None = None,
) -> pd.DataFrame:
    """Pairwise Spearman ρ table sorted by |ρ| descending."""
    flagged_threshold = (
        cfg.COLLINEARITY_PAIRWISE_FLAGGED_ABS_RHO_THRESHOLD
        if flagged_abs_rho_threshold is None
        else flagged_abs_rho_threshold
    )
    columns = [left_col, right_col, "Spearman ρ", "|ρ|", "Flagged Correlation"]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for row in pairs.itertuples():
        attribute, partner = _orient_flagged_pair(row.attribute_a, row.attribute_b)
        abs_rho = float(row.abs_rho)
        rows.append(
            {
                left_col: attribute,
                right_col: partner,
                "Spearman ρ": round(float(row.spearman_rho), 3),
                "|ρ|": round(abs_rho, 3),
                "Flagged Correlation": abs_rho > flagged_threshold,
            }
        )
    table = pd.DataFrame(rows)
    return (
        table.sort_values(
            ["|ρ|", left_col, right_col],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def collinearity_pairs_by_hub(
    pairs: pd.DataFrame,
    *,
    min_hub_flagged_pairs: int = 3,
) -> list[tuple[str, str, pd.DataFrame]]:
    """One partner table per predictor with enough flagged links (symmetric count)."""
    if pairs.empty:
        return []
    importance = _pairwise_flagged_importance(pairs)
    eligible = [
        attribute
        for attribute, (count, _) in importance.items()
        if count >= min_hub_flagged_pairs
    ]
    eligible.sort(
        key=lambda attribute: (
            -importance[attribute][0],
            _hub_priority_rank(attribute),
            attribute,
        )
    )
    hub_tables: list[tuple[str, str, pd.DataFrame]] = []
    for attribute in eligible:
        partner_rows: list[dict[str, object]] = []
        for row in pairs.itertuples():
            if attribute == row.attribute_a:
                partner = row.attribute_b
            elif attribute == row.attribute_b:
                partner = row.attribute_a
            else:
                continue
            partner_rows.append(
                {
                    "Partner Attribute": partner,
                    "Spearman ρ": round(float(row.spearman_rho), 3),
                    "|ρ|": round(float(row.abs_rho), 3),
                }
            )
        partner_table = (
            pd.DataFrame(partner_rows)
            .sort_values("Partner Attribute")
            .reset_index(drop=True)
        )
        pair_count = importance[attribute][0]
        hub_tables.append((attribute, f"{attribute} ({pair_count})", partner_table))
    return hub_tables


def collinearity_hub_recurrence_report(
    pairs: pd.DataFrame,
    *,
    min_hub_flagged_pairs: int = 3,
) -> str:
    """Interpretive summary for the hub-recurrence screening criterion."""
    if pairs.empty:
        return (
            "Conclusion — hub recurrence screening:\n"
            "No predictor pairs exceeded the |ρ| threshold; hub recurrence was not applied."
        )
    importance = _pairwise_flagged_importance(pairs)
    hubs = sorted(
        importance.items(),
        key=lambda item: (-item[1][0], -item[1][1], item[0]),
    )
    priority_hubs = [
        f"- {attribute} ({count} flagged pair(s))"
        for attribute, (count, _) in hubs
        if count >= min_hub_flagged_pairs
    ]
    excluded = [
        f"- {attribute} ({count} flagged pair(s))"
        for attribute, (count, _) in hubs
        if count < min_hub_flagged_pairs
    ]
    excluded_lines = "\n".join(excluded) if excluded else "- none"
    priority_lines = "\n".join(priority_hubs) if priority_hubs else "- none"
    n_priority_attributes = len(priority_hubs)
    return (
        f"Conclusion — hub recurrence screening (≥ {min_hub_flagged_pairs} flagged links):\n"
        "After the |ρ| cut-off, we retain predictors that participate in at least "
        f"{min_hub_flagged_pairs} flagged pairwise links, counting both ends of each pair "
        "(symmetric link count). This avoids crediting a correlation only to the display "
        "hub when another variable is equally entangled — e.g. Fence and Sky remain "
        "priority predictors even when Albedo also links to them (Dormann et al., 2013; "
        "Graham, 2003). After a substantive drop, the matrix should be re-screened because "
        "link counts change (UK Urban Analytics Platform, 2024; Zhu et al., 2026).\n"
        f"Priority predictors (≥ {min_hub_flagged_pairs} flagged links):\n"
        f"{priority_lines}\n"
        f"Excluded as lower-link predictors:\n"
        f"{excluded_lines}\n"
        f"{n_priority_attributes} predictor(s) retained for per-variable review tables."
    )


def _flagged_link_counts(
    df_attr: pd.DataFrame,
    predictors: list[str],
    *,
    abs_rho_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Flagged pairs and symmetric link count per predictor."""
    if abs_rho_threshold is None:
        abs_rho_threshold = cfg.COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD
    corr = attribute_spearman_correlation(df_attr, predictors)
    pairs = pairwise_high_abs_rho_pairs(
        corr, predictors, abs_rho_threshold=abs_rho_threshold
    )
    importance = _pairwise_flagged_importance(pairs)
    link_counts = {attribute: count for attribute, (count, _) in importance.items()}
    for attribute in predictors:
        link_counts.setdefault(attribute, 0)
    return pairs, link_counts


def collinearity_link_count_table(
    link_counts: dict[str, int],
    *,
    min_flagged_links: int = 3,
) -> pd.DataFrame:
    """Sorted link-count table for one screening state."""
    rows = [
        {
            "Attribute": attribute,
            "Flagged links": count,
            "Meets recurrence criterion": count >= min_flagged_links,
        }
        for attribute, count in link_counts.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["Flagged links", "Attribute"], ascending=[False, True])
        .reset_index(drop=True)
    )


def collinearity_recursive_cascade(
    df_attr: pd.DataFrame,
    predictors: list[str],
    *,
    abs_rho_threshold: float | None = None,
    min_flagged_links: int = 3,
) -> pd.DataFrame:
    """
    Top-down recursive screening: each round removes all predictors with at least
    min_flagged_links flagged pairwise links, then recomputes on survivors.
    """
    if abs_rho_threshold is None:
        abs_rho_threshold = cfg.COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD

    remaining = list(predictors)
    removed_cumulative: list[str] = []
    round_rows: list[dict[str, object]] = []
    round_number = 1

    while len(remaining) >= 2:
        pairs, link_counts = _flagged_link_counts(
            df_attr, remaining, abs_rho_threshold=abs_rho_threshold
        )
        priority = sorted(
            [
                attribute
                for attribute, count in link_counts.items()
                if count >= min_flagged_links
            ],
            key=lambda attribute: (
                -link_counts[attribute],
                _hub_priority_rank(attribute),
                attribute,
            ),
        )
        max_links = max(link_counts.values()) if link_counts else 0
        round_rows.append(
            {
                "Round": round_number,
                "Active predictors": len(remaining),
                "Flagged pairs": len(pairs),
                "Max flagged links": max_links,
                "Predictors with ≥3 links": ", ".join(priority) if priority else "—",
                "Removed this round": "—",
                "Removed cumulative": ", ".join(removed_cumulative) if removed_cumulative else "—",
            }
        )
        if not priority:
            break
        round_rows[-1]["Removed this round"] = ", ".join(priority)
        removed_cumulative.extend(priority)
        remaining = [attribute for attribute in remaining if attribute not in priority]
        round_number += 1

    return pd.DataFrame(round_rows)


def collinearity_screening_after_removals(
    df_attr: pd.DataFrame,
    predictors: list[str],
    removed: list[str],
    *,
    abs_rho_threshold: float | None = None,
    min_flagged_links: int = 3,
) -> pd.DataFrame:
    """Link-count snapshot after a hypothetical or actual removal step."""
    remaining = [attribute for attribute in predictors if attribute not in removed]
    if len(remaining) < 2:
        return pd.DataFrame(
            columns=["Attribute", "Flagged links", "Meets recurrence criterion"]
        )
    _, link_counts = _flagged_link_counts(
        df_attr, remaining, abs_rho_threshold=abs_rho_threshold
    )
    return collinearity_link_count_table(
        link_counts, min_flagged_links=min_flagged_links
    )


def collinearity_recursive_cascade_report(
    cascade: pd.DataFrame,
    *,
    min_flagged_links: int = 3,
) -> str:
    """Interpretive summary of the recursive top-down screening view."""
    if cascade.empty:
        return "Conclusion — recursive collinearity cascade:\nNo screening rounds were computed."
    round_lines = "\n".join(
        f"- Round {int(row['Round'])}: {int(row['Active predictors'])} predictors, "
        f"{int(row['Flagged pairs'])} flagged pairs, max links = {int(row['Max flagged links'])}; "
        f"priority = {row['Predictors with ≥3 links']}"
        for _, row in cascade.iterrows()
    )
    final = cascade.iloc[-1]
    return (
        "Conclusion — recursive collinearity cascade:\n"
        f"Criterion 2 (≥ {min_flagged_links} flagged links) is applied top-down: each round "
        "removes all predictors that still qualify, then the correlation matrix is recomputed "
        "on survivors (Dormann et al., 2013; Graham, 2003). This shows that once the first "
        "entangled block is removed, remaining predictors typically fall to ≤ 2 links — so "
        "recursive screening naturally terminates.\n"
        f"{round_lines}\n"
        f"Final state: {int(final['Active predictors'])} active predictors, "
        f"max flagged links = {int(final['Max flagged links'])}."
    )


def all_pairwise_abs_rho_values(
    corr: pd.DataFrame,
    predictors: list[str],
) -> pd.Series:
    """Unique |Spearman ρ| for every predictor pair in the correlation matrix."""
    values: list[float] = []
    for index, attribute in enumerate(predictors):
        if attribute not in corr.index:
            continue
        for partner in predictors[index + 1 :]:
            if partner not in corr.columns:
                continue
            rho = corr.loc[attribute, partner]
            if pd.notna(rho):
                values.append(abs(float(rho)))
    return pd.Series(values, name="abs_rho")


def pairwise_abs_rho_threshold_report(
    corr: pd.DataFrame,
    predictors: list[str],
    *,
    abs_rho_threshold: float | None = None,
) -> str:
    """Summarise why a pairwise |ρ| threshold separates the bulk from upper-tail pairs."""
    if abs_rho_threshold is None:
        abs_rho_threshold = cfg.COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD
    values = all_pairwise_abs_rho_values(corr, predictors)
    n_pairs = len(values)
    n_flagged = int((values >= abs_rho_threshold).sum())
    median = float(values.median())
    mean = float(values.mean())
    p75 = float(values.quantile(0.75))
    p90 = float(values.quantile(0.90))
    maximum = float(values.max())
    pct_flagged = 100.0 * n_flagged / n_pairs if n_pairs else 0.0
    return (
        f"Conclusion — pairwise |ρ| threshold selection (|ρ| ≥ {abs_rho_threshold:g}):\n"
        f"Among {n_pairs} unique predictor pairs, |Spearman ρ| is concentrated at low "
        f"values (median = {median:.2f}, mean = {mean:.2f}; 75th percentile = {p75:.2f}, "
        f"90th percentile = {p90:.2f}). A cut-off of {abs_rho_threshold:g} lies at the "
        "upper tail of this study-specific distribution and flags pairs that depart "
        f"clearly from the bulk ({n_flagged} pairs, {pct_flagged:.1f}% of all pairs; "
        f"maximum |ρ| = {maximum:.2f}). This data-informed rule targets correlation "
        "outliers among our twenty survey points rather than importing a fixed literature "
        "constant without inspecting the empirical correlation structure."
    )


def plot_pairwise_abs_rho_distribution(
    corr: pd.DataFrame,
    predictors: list[str],
    *,
    abs_rho_threshold: float | None = None,
    title: str = "Pairwise |ρ| distribution across streetscape predictors",
) -> plt.Figure:
    """KDE of unique pairwise |ρ| with screening threshold."""
    if abs_rho_threshold is None:
        abs_rho_threshold = cfg.COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD
    values = all_pairwise_abs_rho_values(corr, predictors)
    flagged = values >= abs_rho_threshold

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.kdeplot(
        values,
        ax=ax,
        color="#4C72B0",
        fill=True,
        alpha=cfg.KDE_FILL_ALPHA,
        linewidth=2.0,
        zorder=3,
    )
    sns.rugplot(
        values[~flagged],
        ax=ax,
        color="#4C72B0",
        height=0.045,
        alpha=0.9,
        label="Below threshold",
    )
    sns.rugplot(
        values[flagged],
        ax=ax,
        color="#C44E52",
        height=0.06,
        alpha=0.95,
        label="Flagged pair",
    )
    ax.axvline(
        abs_rho_threshold,
        color="#212121",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
        label=f"|ρ| = {abs_rho_threshold:g}",
    )
    ax.set_xlabel("|Spearman ρ|")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontweight="semibold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig


def _collinearity_union_find_groups(
    predictors: list[str],
    corr: pd.DataFrame,
    *,
    abs_rho_threshold: float,
) -> list[list[str]]:
    """Connected components among predictors with |ρ| at or above the threshold."""
    parent = {column: column for column in predictors}

    def find(column: str) -> str:
        while parent[column] != column:
            parent[column] = parent[parent[column]]
            column = parent[column]
        return column

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index, left in enumerate(predictors):
        if left not in corr.index:
            continue
        for right in predictors[index + 1 :]:
            if right not in corr.columns:
                continue
            rho = corr.loc[left, right]
            if pd.notna(rho) and abs(float(rho)) >= abs_rho_threshold:
                union(left, right)

    buckets: dict[str, list[str]] = {}
    for column in predictors:
        buckets.setdefault(find(column), []).append(column)
    return [sorted(group) for group in buckets.values() if len(group) > 1]


def identify_collinearity_redundancy_groups(
    predictors: list[str],
    corr: pd.DataFrame,
    *,
    abs_rho_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Flag redundancy groups (|ρ| ≥ threshold) for manual substantive selection.

    Returns group summary, within-group pairwise |ρ|, per-attribute membership,
    and the full list of high-|ρ| pairs in the initial matrix.
    """
    if abs_rho_threshold is None:
        abs_rho_threshold = cfg.COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD

    redundancy_groups = _collinearity_union_find_groups(
        predictors, corr, abs_rho_threshold=abs_rho_threshold
    )
    redundancy_groups = sorted(
        redundancy_groups,
        key=lambda group: (-len(group), group[0]),
    )
    pairs_high = pairwise_high_abs_rho_pairs(
        corr, predictors, abs_rho_threshold=abs_rho_threshold
    )

    max_abs_rho = {
        var: float(corr.loc[var, [partner for partner in predictors if partner != var]].abs().max())
        for var in predictors
        if var in corr.index
    }
    n_high_partners = {
        var: int(
            (corr.loc[var, [partner for partner in predictors if partner != var]].abs() >= abs_rho_threshold).sum()
        )
        for var in predictors
        if var in corr.index
    }

    attribute_to_group: dict[str, int] = {}
    for group_id, group in enumerate(redundancy_groups, start=1):
        for attribute in group:
            attribute_to_group[attribute] = group_id

    summary_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    for group_id, group in enumerate(redundancy_groups, start=1):
        group_pairs: list[dict[str, object]] = []
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                rho = float(corr.loc[left, right])
                pair_row = {
                    "group_id": group_id,
                    "attribute_a": left,
                    "attribute_b": right,
                    "spearman_rho": rho,
                    "abs_rho": abs(rho),
                }
                group_pairs.append(pair_row)
                pairwise_rows.append(pair_row)
        strongest = max(group_pairs, key=lambda row: row["abs_rho"])
        summary_rows.append(
            {
                "group_id": group_id,
                "n_members": len(group),
                "attributes": ", ".join(group),
                "strongest_pair": (
                    f"{strongest['attribute_a']} ↔ {strongest['attribute_b']}"
                ),
                "max_abs_rho": round(float(strongest["abs_rho"]), 3),
            }
        )

    groups_summary = pd.DataFrame(
        summary_rows,
        columns=[
            "group_id",
            "n_members",
            "attributes",
            "strongest_pair",
            "max_abs_rho",
        ],
    )
    groups_pairwise = (
        pd.DataFrame(pairwise_rows)
        if pairwise_rows
        else pd.DataFrame(
            columns=[
                "group_id",
                "attribute_a",
                "attribute_b",
                "spearman_rho",
                "abs_rho",
            ]
        )
    )
    if not groups_pairwise.empty:
        groups_pairwise = groups_pairwise.sort_values(
            ["group_id", "abs_rho"], ascending=[True, False]
        ).reset_index(drop=True)

    membership = pd.DataFrame(
        {
            "attribute": predictors,
            "group_id": [attribute_to_group.get(var) for var in predictors],
            "in_redundancy_group": [var in attribute_to_group for var in predictors],
            "max_abs_rho": [round(max_abs_rho[var], 3) for var in predictors],
            "n_partners_abs_rho_ge_threshold": [
                n_high_partners[var] for var in predictors
            ],
            "abs_rho_threshold": abs_rho_threshold,
        }
    )
    return groups_summary, groups_pairwise, membership, pairs_high


def apply_pairwise_collinearity_manual_drops(
    predictors: list[str],
    membership: pd.DataFrame,
    manual_drops: list[str],
) -> tuple[set[str], pd.DataFrame]:
    """Apply substantive drop choices after reviewing redundancy groups."""
    drop_set = set(manual_drops)
    unknown = drop_set - set(predictors)
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Manual drops not in predictor list: {unknown_list}")

    screening = membership.copy()
    screening["manual_drop"] = screening["attribute"].isin(drop_set)
    screening["kept_for_modelling"] = ~screening["manual_drop"]
    kept_set = {var for var in predictors if var not in drop_set}
    return kept_set, screening


def collinearity_redundancy_groups_report(
    groups_summary: pd.DataFrame,
    groups_pairwise: pd.DataFrame,
    membership: pd.DataFrame,
    pairs_high: pd.DataFrame,
    *,
    abs_rho_threshold: float | None = None,
) -> str:
    """Interpretive summary of redundancy groups awaiting manual selection."""
    if abs_rho_threshold is None:
        abs_rho_threshold = float(membership["abs_rho_threshold"].iloc[0])
    n_groups = len(groups_summary)
    if n_groups == 0:
        return (
            f"Conclusion — pairwise collinearity review (|ρ| ≥ {abs_rho_threshold:g}):\n"
            f"No redundancy groups were formed at |ρ| = {abs_rho_threshold:g}; "
            f"all {len(membership)} predictors can be retained."
        )

    group_lines = "\n".join(
        f"- Group {int(row.group_id)} ({int(row.n_members)} members): {row.attributes}\n"
        f"  Strongest link: {row.strongest_pair} (|ρ| = {row.max_abs_rho:.3f})"
        for row in groups_summary.itertuples()
    )
    pair_lines = "\n".join(
        f"- [{int(row.group_id)}] {row.attribute_a} ↔ {row.attribute_b}: "
        f"|ρ| = {row.abs_rho:.3f} (ρ = {row.spearman_rho:+.3f})"
        for row in groups_pairwise.itertuples()
    )
    isolated = membership.loc[~membership["in_redundancy_group"], "attribute"].tolist()
    isolated_line = (
        ", ".join(isolated)
        if isolated
        else "none"
    )
    return (
        f"Conclusion — pairwise collinearity review (|ρ| ≥ {abs_rho_threshold:g}):\n"
        f"{n_groups} redundancy group(s) were identified via connected components on "
        f"pairs with |Spearman ρ| ≥ {abs_rho_threshold:g}. Review each group and list "
        "your substantive drops in MANUAL_DROPS before updating kept_set:\n"
        f"{group_lines}\n"
        f"Within-group |ρ|:\n"
        f"{pair_lines}\n"
        f"Predictors outside any redundancy group: {isolated_line}.\n"
        f"All flagged pairs in the initial matrix ({len(pairs_high)}):\n"
        + "\n".join(
            f"- {row.attribute_a} ↔ {row.attribute_b}: |ρ| = {row.abs_rho:.3f} "
            f"(ρ = {row.spearman_rho:+.3f})"
            for row in pairs_high.itertuples()
        )
    )


def collinearity_pairwise_manual_screening_report(
    screening_summary: pd.DataFrame,
    groups_summary: pd.DataFrame,
    manual_drops: list[str],
    *,
    abs_rho_threshold: float | None = None,
) -> str:
    """Interpretive conclusion after applying manual pairwise drops."""
    if abs_rho_threshold is None:
        abs_rho_threshold = float(screening_summary["abs_rho_threshold"].iloc[0])
    dropped = screening_summary.loc[screening_summary["manual_drop"], "attribute"].tolist()
    n_kept = int(screening_summary["kept_for_modelling"].sum())
    if not dropped:
        if groups_summary.empty:
            return (
                f"Conclusion — pairwise collinearity screening (|ρ| ≥ {abs_rho_threshold:g}):\n"
                f"No redundancy groups at |ρ| = {abs_rho_threshold:g}; all {n_kept} "
                "attributes retained (MANUAL_DROPS is empty)."
            )
        return (
            f"Conclusion — pairwise collinearity screening (|ρ| ≥ {abs_rho_threshold:g}):\n"
            f"{len(groups_summary)} redundancy group(s) were flagged but MANUAL_DROPS "
            f"is empty — all {n_kept} attributes are still retained. Edit MANUAL_DROPS "
            "after reviewing the group tables above."
        )

    drop_lines = "\n".join(f"- {attribute}" for attribute in dropped)
    group_lines = "\n".join(
        f"- Group {int(row.group_id)}: kept "
        f"{', '.join(sorted(set(row.attributes.split(', ')) - set(manual_drops)))}"
        for row in groups_summary.itertuples()
    )
    return (
        f"Conclusion — pairwise collinearity screening (|ρ| ≥ {abs_rho_threshold:g}):\n"
        f"Substantive manual drops ({len(dropped)}):\n"
        f"{drop_lines}\n"
        f"Remaining members per redundancy group:\n"
        f"{group_lines}\n"
        f"{n_kept} streetscape predictors retained for modelling."
    )


def screen_attribute_predictors_pairwise(
    predictors: list[str],
    corr: pd.DataFrame,
    *,
    abs_rho_threshold: float | None = None,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop predictors iteratively while any pair has |ρ| >= threshold.

    When a high-|ρ| pair remains, remove the member with the higher Σ|ρ| on the
    active set (standard automated tie-break when substantive priority is not applied).
    """
    if abs_rho_threshold is None:
        abs_rho_threshold = cfg.COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD

    pairs_initial = pairwise_high_abs_rho_pairs(
        corr, predictors, abs_rho_threshold=abs_rho_threshold
    )
    max_abs_rho = {
        var: float(corr.loc[var, [p for p in predictors if p != var]].abs().max())
        for var in predictors
        if var in corr.index
    }
    n_high_partners = {
        var: int(
            (corr.loc[var, [p for p in predictors if p != var]].abs() >= abs_rho_threshold).sum()
        )
        for var in predictors
        if var in corr.index
    }

    remaining = set(predictors)
    drop_rows: list[dict[str, object]] = []
    step = 0
    while len(remaining) > 1:
        high_pairs: list[tuple[str, str, float]] = []
        remaining_list = sorted(remaining)
        for i, attribute in enumerate(remaining_list):
            for partner in remaining_list[i + 1 :]:
                rho = float(corr.loc[attribute, partner])
                if abs(rho) >= abs_rho_threshold:
                    high_pairs.append((attribute, partner, abs(rho)))
        if not high_pairs:
            break
        attribute, partner, abs_rho = max(high_pairs, key=lambda item: item[2])
        sum_a = matrix_abs_rho_sum(corr, attribute, remaining_list)
        sum_b = matrix_abs_rho_sum(corr, partner, remaining_list)
        dropped = attribute if sum_a >= sum_b else partner
        kept_partner = partner if dropped == attribute else attribute
        step += 1
        drop_rows.append(
            {
                "step": step,
                "dropped": dropped,
                "partner": kept_partner,
                "spearman_rho": float(corr.loc[dropped, kept_partner]),
                "abs_rho": abs_rho,
                "sum_abs_rho_dropped": round(sum_a if dropped == attribute else sum_b, 3),
                "sum_abs_rho_partner": round(sum_b if dropped == attribute else sum_a, 3),
            }
        )
        remaining.remove(dropped)

    drop_set = {row["dropped"] for row in drop_rows}
    partner_map = {row["dropped"]: row["partner"] for row in drop_rows}
    screening = pd.DataFrame(
        {
            "attribute": predictors,
            "max_abs_rho": [round(max_abs_rho[var], 3) for var in predictors],
            "n_partners_abs_rho_ge_threshold": [
                n_high_partners[var] for var in predictors
            ],
            "abs_rho_threshold": abs_rho_threshold,
            "dropped_partner": [partner_map.get(var) for var in predictors],
            "kept_for_modelling": [var not in drop_set for var in predictors],
        }
    )
    drops_log = pd.DataFrame(drop_rows)
    return remaining, screening, pairs_initial, drops_log


def plot_pairwise_max_abs_rho_screening(
    screening_summary: pd.DataFrame,
    *,
    abs_rho_threshold: float | None = None,
    title: str = "Maximum |ρ| per streetscape predictor",
) -> plt.Figure:
    """KDE of each attribute's strongest |ρ| with any partner; threshold as vertical line."""
    if abs_rho_threshold is None:
        abs_rho_threshold = float(screening_summary["abs_rho_threshold"].iloc[0])
    values = screening_summary["max_abs_rho"]
    kept = screening_summary["kept_for_modelling"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.kdeplot(
        values,
        ax=ax,
        color="#4C72B0",
        fill=True,
        alpha=cfg.KDE_FILL_ALPHA,
        linewidth=2.0,
        zorder=3,
    )
    sns.rugplot(values[kept], ax=ax, color="#4C72B0", height=0.045, alpha=0.9, label="Kept")
    sns.rugplot(
        values[~kept],
        ax=ax,
        color="#C44E52",
        height=0.06,
        alpha=0.95,
        label="Removed",
    )
    ax.axvline(
        abs_rho_threshold,
        color="#212121",
        linestyle="--",
        linewidth=1.4,
        zorder=2,
        label=f"|ρ| = {abs_rho_threshold:g}",
    )
    ax.set_xlabel("Maximum |Spearman ρ| with any other predictor")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontweight="semibold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig


def collinearity_pairwise_screening_report(
    screening_summary: pd.DataFrame,
    pairs_high: pd.DataFrame,
    drops_log: pd.DataFrame,
    *,
    abs_rho_threshold: float | None = None,
) -> str:
    """Interpretive conclusion for pairwise |ρ| collinearity screening."""
    if abs_rho_threshold is None:
        abs_rho_threshold = float(screening_summary["abs_rho_threshold"].iloc[0])
    dropped = screening_summary.loc[~screening_summary["kept_for_modelling"], "attribute"].tolist()
    n_kept = int(screening_summary["kept_for_modelling"].sum())
    if not dropped:
        return (
            f"Conclusion — pairwise collinearity screening (|ρ| ≥ {abs_rho_threshold:g}):\n"
            f"No predictor pairs exceeded |ρ| = {abs_rho_threshold:g}; all {n_kept} "
            "attributes were retained for modelling."
        )
    removed_lines = "\n".join(
        f"- {row.dropped}: |ρ| = {row.abs_rho:.3f} with {row.partner} "
        f"(ρ = {row.spearman_rho:+.3f}); removed as the more redundant member "
        f"(Σ|ρ| = {row.sum_abs_rho_dropped:.3f} vs {row.sum_abs_rho_partner:.3f})"
        for row in drops_log.itertuples()
    )
    pair_lines = "\n".join(
        f"- {row.attribute_a} ↔ {row.attribute_b}: |ρ| = {row.abs_rho:.3f} "
        f"(ρ = {row.spearman_rho:+.3f})"
        for row in pairs_high.itertuples()
    )
    return (
        f"Conclusion — pairwise collinearity screening (|ρ| ≥ {abs_rho_threshold:g}):\n"
        f"Following Dormann et al. (2013), predictor pairs with |Spearman ρ| at or above "
        f"{abs_rho_threshold:g} were flagged as redundant. Where such pairs remained, the "
        "attribute with the higher Σ|ρ| on the active set was dropped:\n"
        f"{removed_lines}\n"
        f"High-|ρ| pairs in the initial matrix:\n"
        f"{pair_lines}\n"
        f"The remaining {n_kept} streetscape predictors can be used together in RQ2 "
        "without carrying the strongest pairwise redundancies."
    )


def attribute_spearman_correlation(
    df_attr: pd.DataFrame,
    predictors: list[str],
) -> pd.DataFrame:
    """Pairwise Spearman ρ among attribute predictors (symmetric matrix, diagonal = 1)."""
    numeric = df_attr[predictors].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="spearman")


def _short_attribute_label(name: str) -> str:
    return (
        str(name)
        .replace(" (%)", "")
        .replace("_mean", "")
        .replace("LST _mean(°C)", "LST")
    )


def _short_layer_name(name: str) -> str:
    if name.startswith("Layer 1"):
        return "Layer 1 — RS / GIS"
    if "3a" in name:
        return "Layer 3a — SVI pixel %"
    return name


def _layer_band_color(layer_name: str) -> str:
    return cfg.ATTRIBUTE_LAYER_BAND_COLORS.get(layer_name, "#5C5C5C")


def order_predictors_by_layer(
    predictors: list[str],
    layers: dict[str, list[str]],
) -> tuple[list[str], list[tuple[str, int, int]], dict[str, str]]:
    """Order attributes by layer (Layer 1, then Layer 3a, …) and record group spans."""
    ordered: list[str] = []
    boundaries: list[tuple[str, int, int]] = []
    col_layer: dict[str, str] = {}

    for layer_name, cols in layers.items():
        layer_cols = [c for c in cols if c in predictors]
        if not layer_cols:
            continue
        start = len(ordered)
        ordered.extend(layer_cols)
        boundaries.append((layer_name, start, len(ordered)))
        col_layer.update({c: layer_name for c in layer_cols})

    for col in predictors:
        if col not in col_layer:
            start = len(ordered)
            ordered.append(col)
            col_layer[col] = "Other"
            boundaries.append(("Other", start, len(ordered)))

    return ordered, boundaries, col_layer


def layer_predictor_groups(
    layers: dict[str, list[str]],
    predictors: list[str],
) -> tuple[str, list[str], str, list[str]]:
    """Return Layer 1 and Layer 3a predictor lists and their group names."""
    layer1_name = next((n for n in layers if n.startswith("Layer 1")), "Layer 1 — RS / GIS")
    layer3a_name = next((n for n in layers if "3a" in n), "Layer 3a — SVI Pixel %")
    layer1 = [c for c in layers.get(layer1_name, []) if c in predictors]
    layer3a = [c for c in layers.get(layer3a_name, []) if c in predictors]
    return layer1_name, layer1, layer3a_name, layer3a


def cross_layer_spearman_correlation(
    df_attr: pd.DataFrame,
    predictors: list[str],
    *,
    layers: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Spearman ρ between Layer 1 (rows) and Layer 3a (columns) predictors only."""
    layers = layers or cfg.ATTRIBUTE_LAYERS_ALL
    layer1_name, layer1, layer3a_name, layer3a = layer_predictor_groups(layers, predictors)
    corr = attribute_spearman_correlation(df_attr, predictors)
    cross = corr.loc[layer1, layer3a]
    cross.index.name = "layer1_attribute"
    cross.columns.name = "layer3a_attribute"
    return cross, layer1_name, layer3a_name


def _spearman_diverging_colormap():
    """Burgundy at ρ = −1, white at 0, forest green at ρ = +1; bold already at |ρ| ≈ 0.5."""
    full_red = (0.78, 0.15, 0.18, 1.0)
    mid_red = (0.80, 0.28, 0.30, 1.0)
    white = (1.0, 1.0, 1.0, 1.0)
    mid_green = (0.22, 0.62, 0.32, 1.0)
    full_green = (0.10, 0.45, 0.28, 1.0)
    return mcolors.LinearSegmentedColormap.from_list(
        "spearman_diverging",
        [full_red, mid_red, white, mid_green, full_green],
    )


def _cross_layer_corr_colormap():
    return _spearman_diverging_colormap()


def _corr_strength_colormap():
    """White at ρ = 0; full red at ρ = ±1; strong red already at |ρ| ≈ 0.5."""
    full_red = plt.cm.RdBu_r(0.0)
    mid_red = plt.cm.RdBu_r(0.15)
    white = (1.0, 1.0, 1.0, 1.0)
    # Stops at ρ = −1, −0.5, 0, +0.5, +1 (evenly spaced in [−1, 1]).
    return mcolors.LinearSegmentedColormap.from_list(
        "spearman_strength",
        [full_red, mid_red, white, mid_red, full_red],
    )


def plot_cross_layer_correlation(
    cross_corr: pd.DataFrame,
    *,
    layer1_name: str,
    layer3a_name: str,
    title: str = "Spearman ρ — Layer 1 (RS / GIS) vs Layer 3a (SVI pixel %)",
) -> plt.Figure:
    """Rectangular heatmap: Layer 1 attributes (rows) × Layer 3a attributes (columns)."""
    row_labels = [_short_attribute_label(c) for c in cross_corr.index]
    col_labels = [_short_attribute_label(c) for c in cross_corr.columns]
    nrows, ncols = cross_corr.shape
    cell_in = 0.52
    fig, ax = plt.subplots(
        figsize=(cell_in * ncols + 3.0, cell_in * nrows + 2.6),
    )
    sns.heatmap(
        cross_corr,
        ax=ax,
        vmin=-1,
        vmax=1,
        cmap=_cross_layer_corr_colormap(),
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ  (red → −1, white → 0, green → +1)", "shrink": 0.9},
        yticklabels=row_labels,
        xticklabels=col_labels,
    )

    layer1_color = _layer_band_color(layer1_name)
    layer3a_color = _layer_band_color(layer3a_name)
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=8)
    for tick in ax.get_yticklabels():
        tick.set_color(layer1_color)
    for tick in ax.get_xticklabels():
        tick.set_color(layer3a_color)

    ax.set_ylabel(_short_layer_name(layer1_name), fontweight="semibold", color=layer1_color)
    ax.set_xlabel(_short_layer_name(layer3a_name), fontweight="semibold", color=layer3a_color)
    ax.set_title(title, fontweight="semibold", pad=12)
    fig.tight_layout()
    return fig


def plot_attribute_correlation_matrix(
    corr: pd.DataFrame,
    layers: dict[str, list[str]],
    *,
    title: str = "Spearman ρ — streetscape attributes",
) -> plt.Figure:
    """Lower-triangular Spearman matrix ordered by attribute layer (diagonal empty)."""
    predictors = [c for c in corr.columns if c in corr.index]
    ordered, boundaries, col_layer = order_predictors_by_layer(predictors, layers)
    corr_ord = corr.loc[ordered, ordered]
    n = len(ordered)
    mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    labels = [_short_attribute_label(c) for c in ordered]

    cell_in = 0.42
    fig, ax = plt.subplots(figsize=(cell_in * n + 4.6, cell_in * n + 4.2))
    sns.heatmap(
        corr_ord,
        mask=mask,
        ax=ax,
        vmin=-1,
        vmax=1,
        cmap=_spearman_diverging_colormap(),
        square=True,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ  (red → −1, white → 0, green → +1)", "shrink": 0.85},
        xticklabels=labels,
        yticklabels=labels,
    )

    for _layer_name, start, _end in boundaries:
        if start > 0:
            ax.axhline(start, color="0.15", lw=1.8, zorder=5)
            ax.axvline(start, color="0.15", lw=1.8, zorder=5)

    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(_layer_band_color(col_layer[ordered[i]]))
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(_layer_band_color(col_layer[ordered[i]]))

    y_layer_trans = blended_transform_factory(ax.transAxes, ax.transData)
    x_layer_trans = blended_transform_factory(ax.transData, ax.transAxes)
    for layer_name, start, end in boundaries:
        mid = (start + end) / 2.0
        color = _layer_band_color(layer_name)
        short_name = _short_layer_name(layer_name)
        ax.text(
            -0.11,
            mid,
            short_name,
            transform=y_layer_trans,
            rotation=90,
            va="center",
            ha="center",
            fontsize=8.5,
            fontweight="semibold",
            color=color,
            clip_on=False,
        )
        ax.text(
            mid,
            -0.10,
            short_name,
            transform=x_layer_trans,
            va="top",
            ha="center",
            fontsize=8.5,
            fontweight="semibold",
            color=color,
            clip_on=False,
        )

    ax.set_title(title, fontweight="semibold", pad=12)
    fig.subplots_adjust(left=0.20, bottom=0.16, right=0.92, top=0.94)
    return fig


# ---------------------------------------------------------------------------
# Inter-rater agreement
# ---------------------------------------------------------------------------


def _pairwise_distribution_summary(scores: pd.Series, **meta) -> dict:
    """Summarise a distribution of pairwise agreement scores (Spearman ρ or ordinal)."""
    row = dict(meta)
    if scores.empty:
        row["n_pairs"] = 0
        return row
    d = scores.describe()
    row.update(
        {
            "n_pairs": int(d["count"]),
            "mean": d["mean"],
            "std": d["std"],
            "min": d["min"],
            "p25": d["25%"],
            "median": d["50%"],
            "p75": d["75%"],
            "max": d["max"],
        }
    )
    return row


def inter_rater_spearman(df_city: pd.DataFrame) -> pd.Series:
    """Pairwise Spearman ρ between respondent 10-image comfort profiles."""
    ratings = df_city[cfg.POINT_COLUMNS].dropna(how="any")
    rhos: list[float] = []
    for i, j in combinations(range(len(ratings)), 2):
        a = ratings.iloc[i]
        b = ratings.iloc[j]
        if a.nunique() < 2 or b.nunique() < 2:
            continue
        rho, _ = spearmanr(a, b)
        if pd.notna(rho):
            rhos.append(float(rho))
    return pd.Series(rhos)


def inter_rater_spearman_city_summary(df_survey: pd.DataFrame) -> pd.DataFrame:
    """City-level inter-rater reliability: pairwise Spearman ρ across 10-image profiles."""
    rows = []
    for city in cfg.CITIES:
        sub = df_survey.loc[df_survey["questionnaire_city"] == city]
        rhos = inter_rater_spearman(sub)
        rows.append(_pairwise_distribution_summary(rhos, city=city))
    return pd.DataFrame(rows).round(3)


def inter_rater_spearman_summary(df_survey: pd.DataFrame) -> pd.DataFrame:
    """Alias for :func:`inter_rater_spearman_city_summary`."""
    return inter_rater_spearman_city_summary(df_survey)


def inter_rater_image_agreement_scores(
    df_city: pd.DataFrame,
    image_col: str,
) -> pd.Series:
    """Pairwise ordinal agreement on one image (one rating per respondent).

    Each pair receives ``1 - |a - b| / (max_scale - min_scale)`` on the comfort
    scale (1–5), so 1 is perfect agreement and 0 is maximum disagreement.
    """
    scale_span = max(cfg.RATING_ORDER) - min(cfg.RATING_ORDER)
    ratings = df_city[image_col].dropna().astype(float)
    scores: list[float] = []
    values = ratings.to_numpy()
    for i, j in combinations(range(len(values)), 2):
        scores.append(1.0 - abs(values[i] - values[j]) / scale_span)
    return pd.Series(scores)


def inter_rater_image_agreement_summary(df_survey: pd.DataFrame) -> pd.DataFrame:
    """Image-level inter-rater reliability per city and streetscape image."""
    rows = []
    for city in cfg.CITIES:
        sub = df_survey.loc[df_survey["questionnaire_city"] == city]
        for image_col in cfg.POINT_COLUMNS:
            scores = inter_rater_image_agreement_scores(sub, image_col)
            ratings = sub[image_col].dropna().astype(float)
            row = _pairwise_distribution_summary(
                scores,
                city=city,
                image=image_col,
                n_raters=int(ratings.shape[0]),
            )
            if not ratings.empty:
                row["mean_rating"] = float(ratings.mean())
                row["std_rating"] = float(ratings.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows).round(3)


def inter_rater_spearman_report(summary: pd.DataFrame) -> str:
    """Interpretive conclusion for the inter-rater Spearman agreement check."""
    both_positive = bool((summary["median"] > 0).all())
    if not both_positive:
        return (
            "Conclusion — inter-rater agreement:\n"
            "Pairwise correlations do not support pooling ratings as a shared scale in every city; "
            "interpret city-level results with caution."
        )
    return (
        "Conclusion — inter-rater agreement:\n"
        "Respondents in Detmold and Turin largely agree on the relative ordering of the ten "
        "streetscape images.\n"
        "That pattern is inconsistent with arbitrary or personal use of the "
        "comfort scale: people are applying the same ordinal categories to similar visual cues. \n"
        "We therefore treat within-city ratings as comparable and pool them for city-scale "
        "profiles and group comparisons in RQ1."
    )


def plot_inter_rater_spearman_distribution(
    df_survey: pd.DataFrame,
    *,
    title: str = "Pairwise Spearman ρ — inter-rater reliability",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, city in zip(axes, cfg.CITIES):
        sub = df_survey.loc[df_survey["questionnaire_city"] == city]
        rhos = inter_rater_spearman(sub)
        color = cfg.CITY_COLORS[city]
        if not rhos.empty:
            sns.kdeplot(
                rhos,
                ax=ax,
                color=color,
                fill=True,
                alpha=cfg.KDE_FILL_ALPHA,
                linewidth=2.0,
                zorder=5,
            )
        ax.set_title(city, fontweight="semibold")
        ax.set_xlabel("Spearman ρ")
        ax.set_xlim(-1, 1)
        sns.despine(ax=ax, top=True, right=True)
    axes[0].set_ylabel("Density")
    fig.suptitle(title, fontweight="semibold", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RQ1 — city-level & respondent profiles
# ---------------------------------------------------------------------------


def city_level_comfort_stats(df_survey: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in cfg.CITIES:
        sub = df_survey.loc[df_survey["questionnaire_city"] == city, cfg.POINT_COLUMNS]
        vals = pd.to_numeric(sub.stack(), errors="coerce").dropna()
        rows.append(
            {
                "city": city,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n_ratings": int(len(vals)),
                "n_respondents": int(sub.notna().all(axis=1).sum()),
            }
        )
    return pd.DataFrame(rows).round(3)


def plot_city_level_distributions(df_survey: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of comfort rating counts by city."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cfg.RATING_ORDER))
    width = 0.36

    for i, city in enumerate(cfg.CITIES):
        sub = df_survey.loc[df_survey["questionnaire_city"] == city, cfg.POINT_COLUMNS]
        vals = pd.to_numeric(sub.stack(), errors="coerce").dropna()
        counts = vals.value_counts().reindex(cfg.RATING_ORDER, fill_value=0)
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            counts.values,
            width,
            label=city,
            color=cfg.CITY_COLORS[city],
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cfg.RATING_ORDER)
    ax.set_xlabel("Comfort rating")
    ax.set_ylabel("Count")
    ax.set_title("City-level comfort rating distributions", fontweight="semibold")
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_city_level_mean_std(stats: pd.DataFrame) -> plt.Figure:
    """Mean comfort with ±1 SD error bars — mirrors ``city_level_comfort_stats``."""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = np.arange(len(stats))
    means = stats["mean"].to_numpy(dtype=float)
    stds = stats["std"].to_numpy(dtype=float)
    colors = [cfg.CITY_COLORS[city] for city in stats["city"]]

    ax.bar(x, means, width=0.48, color=colors, alpha=0.92, zorder=2)
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        color="#333333",
        capsize=7,
        linewidth=1.4,
        zorder=3,
    )
    for xi, mean, std in zip(x, means, stds):
        ax.text(
            xi,
            mean + std + 0.1,
            f"{mean:.2f} ± {std:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stats["city"])
    ax.set_ylim(0.5, 5.2)
    ax.set_ylabel("Mean comfort rating")
    ax.set_title("City-level mean comfort (±1 SD)", fontweight="semibold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def _category_axis_labels(categories: list) -> list[str]:
    return [str(c).replace("_", " ") for c in categories]


def apply_questionnaire_profile_harmonization(df_survey: pd.DataFrame) -> pd.DataFrame:
    """Apply RQ1 a-priori questionnaire category rules at respondent level."""
    out = df_survey.copy()
    if "gender" in out.columns:
        out = out.loc[~out["gender"].isin(cfg.RQ1_EXCLUDED_GENDER_CATEGORIES)].copy()

    for variable, group_spec in cfg.QUESTIONNAIRE_PROFILE_CATEGORY_GROUPS.items():
        if variable not in out.columns:
            continue
        labels = cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS.get(variable, [])
        mapping: dict[object, str] = {}
        for group_idx, member_cats in enumerate(group_spec):
            label = (
                labels[group_idx]
                if group_idx < len(labels)
                else _format_group_categories(member_cats)
            )
            for cat in member_cats:
                mapping[cat] = label
                mapping[str(cat)] = label
        out[variable] = out[variable].replace(mapping)

    return out.reset_index(drop=True)


def apply_rq1_profile_sample(df_survey: pd.DataFrame) -> pd.DataFrame:
    """RQ1 respondent-profile subset: harmonised categories and excluded genders."""
    return apply_questionnaire_profile_harmonization(df_survey)


def category_mean_ratings_table(
    df_survey: pd.DataFrame,
    variable: str,
    *,
    point_col: str | None = None,
) -> pd.DataFrame:
    tmp = df_survey[["questionnaire_city", variable, *cfg.POINT_COLUMNS]].copy()
    if point_col:
        tmp["rating"] = pd.to_numeric(tmp[point_col], errors="coerce")
        scope = point_col
    else:
        tmp["rating"] = tmp[cfg.POINT_COLUMNS].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        scope = "all_points"
    grouped = (
        tmp.dropna(subset=[variable, "rating"])
        .groupby(["questionnaire_city", variable], observed=True)["rating"]
        .agg(mean="mean", n="count")
        .reset_index()
    )
    grouped.insert(0, "scope", scope)
    return grouped.rename(columns={variable: "category"})


def aggregate_summary_by_profile_groups(
    summary: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """Collapse category means to predefined RQ1 profile groups when configured."""
    group_spec = cfg.QUESTIONNAIRE_PROFILE_CATEGORY_GROUPS.get(variable)
    if not group_spec:
        return summary

    labels = cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS.get(variable, [])
    rows: list[dict[str, object]] = []
    scope = summary["scope"].iloc[0] if not summary.empty and "scope" in summary.columns else "all_points"

    for city in cfg.CITIES:
        city_sub = summary.loc[summary["questionnaire_city"] == city]
        if city_sub.empty:
            continue
        by_cat = city_sub.set_index("category")
        for group_idx, member_cats in enumerate(group_spec):
            present = [cat for cat in member_cats if cat in by_cat.index]
            if not present:
                continue
            total_n = int(by_cat.loc[present, "n"].sum())
            if total_n == 0:
                continue
            weighted_mean = float(
                (by_cat.loc[present, "mean"] * by_cat.loc[present, "n"]).sum() / total_n
            )
            label = (
                labels[group_idx]
                if group_idx < len(labels)
                else _format_group_categories(present)
            )
            rows.append(
                {
                    "scope": scope,
                    "questionnaire_city": city,
                    "category": label,
                    "mean": weighted_mean,
                    "n": total_n,
                }
            )

    if not rows:
        return summary
    return pd.DataFrame(rows)


def rq1_category_mean_ratings_table(
    df_survey: pd.DataFrame,
    variable: str,
    *,
    point_col: str | None = None,
) -> pd.DataFrame:
    """Mean comfort by category with RQ1 a-priori profile groups applied when configured."""
    return aggregate_summary_by_profile_groups(
        category_mean_ratings_table(df_survey, variable, point_col=point_col),
        variable,
    )


def category_mean_ratings_table_profile_grouped(
    df_survey: pd.DataFrame,
    variable: str,
    *,
    point_col: str | None = None,
) -> pd.DataFrame:
    """Category mean comfort table with predefined profile groups applied when configured."""
    return aggregate_summary_by_profile_groups(
        category_mean_ratings_table(df_survey, variable, point_col=point_col),
        variable,
    )


def _ordered_categories(summary: pd.DataFrame, variable: str) -> list:
    order = list(cfg.COVARIATE_CATEGORY_ORDERS.get(variable, ()))
    categories = summary["category"].unique().tolist()
    profile_labels = cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS.get(variable)
    if profile_labels and set(map(str, categories)).issubset(set(profile_labels)):
        return [label for label in profile_labels if label in categories]
    if order:
        return [c for c in order if c in categories] + [c for c in categories if c not in order]
    return sorted(categories, key=str)


def _draw_not_retained_panel(ax: plt.Axes, *, variable: str, city: str) -> None:
    """Placeholder when a questionnaire field was removed for this city in preprocessing."""
    ax.clear()
    ax.set_axis_off()
    label = cfg.QUESTIONNAIRE_VARIABLE_LABELS.get(variable, variable.replace("_", " "))
    ax.text(
        0.5,
        0.5,
        f"Not retained\n(§1–§2 screening)",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="#666666",
    )
    ax.set_title(f"{label} — {city}", fontweight="semibold", fontsize=9)


def _draw_category_mean_ratings_on_ax(
    ax: plt.Axes,
    summary: pd.DataFrame,
    variable: str,
    *,
    show_ylabel: bool = True,
    city: str | None = None,
) -> None:
    plot_summary = (
        summary.loc[summary["questionnaire_city"] == city] if city else summary
    )
    categories = _ordered_categories(plot_summary, variable)
    if not categories:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No category data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#666666",
        )
        ax.set_title(variable.replace("_", " "), fontweight="semibold", fontsize=9)
        return

    x = np.arange(len(categories))
    cities = [city] if city else list(cfg.CITIES)
    width = 0.72 if city else 0.36

    for i, city_name in enumerate(cities):
        sub = summary.loc[summary["questionnaire_city"] == city_name].set_index("category")
        vals = [float(sub.loc[c, "mean"]) if c in sub.index else np.nan for c in categories]
        offset = 0.0 if city else (i - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=city_name,
            color=cfg.CITY_COLORS[city_name],
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(_category_axis_labels(categories), rotation=45, ha="right", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Mean comfort", fontsize=8)
    ax.set_ylim(0.5, 5.2)
    label = cfg.QUESTIONNAIRE_VARIABLE_LABELS.get(variable, variable.replace("_", " "))
    ax.set_title(label, fontweight="semibold", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    sns.despine(ax=ax)


def plot_category_mean_ratings(
    summary: pd.DataFrame,
    variable: str,
    *,
    title: str,
    city: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(6.5, len(_ordered_categories(summary, variable)) * 0.55), 4.2))
    _draw_category_mean_ratings_on_ax(ax, summary, variable, city=city)
    if city is None:
        ax.legend(frameon=False)
    ax.set_title(title, fontweight="semibold")
    fig.tight_layout()
    return fig


def plot_category_mean_ratings_grid(
    tables_by_variable: dict[str, pd.DataFrame],
    *,
    city: str | None = None,
    variables: list[str] | None = None,
    ncols: int = 4,
    suptitle: str | None = None,
) -> plt.Figure:
    """Grid of mean-comfort bar charts, one panel per questionnaire variable."""
    from matplotlib.patches import Patch

    variables = variables or list(tables_by_variable)
    if suptitle is None:
        suptitle = (
            f"Mean comfort by respondent profile — {city}"
            if city
            else "Mean comfort by respondent profile — all images (Detmold vs Turin)"
        )
    n = len(variables)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.1), sharey=True)
    axes_grid = np.atleast_2d(axes)

    for idx, variable in enumerate(variables):
        row, col = divmod(idx, ncols)
        _draw_category_mean_ratings_on_ax(
            axes_grid[row, col],
            tables_by_variable[variable],
            variable,
            show_ylabel=col == 0,
            city=city,
        )

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes_grid[row, col].axis("off")

    if city is None:
        legend_handles = [
            Patch(facecolor=cfg.CITY_COLORS[city_name], alpha=0.92, label=city_name)
            for city_name in cfg.CITIES
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(cfg.CITIES),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(suptitle, fontweight="semibold", y=1.05, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98] if city is None else None)
    return fig


def plot_category_mean_ratings_city_pair_figures(
    tables_by_variable: dict[str, pd.DataFrame],
    *,
    variables: list[str] | None = None,
    variables_by_city: dict[str, list[str]] | None = None,
    variables_per_figure: int = 3,
    suptitle_prefix: str = "Mean comfort by respondent profile",
) -> list[plt.Figure]:
    """Paginated figures: each row is one variable with Detmold (left) and Turin (right)."""
    variables = variables or list(tables_by_variable)
    figures: list[plt.Figure] = []

    for batch_start in range(0, len(variables), variables_per_figure):
        batch = variables[batch_start : batch_start + variables_per_figure]
        nrows = len(batch)
        fig, axes = plt.subplots(nrows, 2, figsize=(10.5, nrows * 3.25), sharey=True)
        axes_grid = np.atleast_2d(axes)

        for row, variable in enumerate(batch):
            label = cfg.QUESTIONNAIRE_VARIABLE_LABELS.get(variable, variable.replace("_", " "))
            for col, city in enumerate(cfg.CITIES):
                ax = axes_grid[row, col]
                retained = (
                    variable in variables_by_city.get(city, [])
                    if variables_by_city is not None
                    else True
                )
                if retained:
                    _draw_category_mean_ratings_on_ax(
                        ax,
                        tables_by_variable[variable],
                        variable,
                        show_ylabel=col == 0,
                        city=city,
                    )
                    ax.set_title(f"{label} — {city}", fontweight="semibold", fontsize=9)
                else:
                    _draw_not_retained_panel(ax, variable=variable, city=city)

        page = batch_start // variables_per_figure + 1
        n_pages = int(np.ceil(len(variables) / variables_per_figure))
        fig.suptitle(
            f"{suptitle_prefix} (page {page}/{n_pages})",
            fontweight="semibold",
            y=1.02,
            fontsize=12,
        )
        fig.tight_layout()
        figures.append(fig)

    return figures


def _format_category_label(category: str) -> str:
    return str(category).replace("_", " ")


def _format_group_categories(members: list[str]) -> str:
    return ", ".join(_format_category_label(member) for member in members)


def _category_rows_for_city(
    summary: pd.DataFrame,
    variable: str,
    city: str,
) -> list[dict]:
    """One row per response category for a city, in display order."""
    sub = summary.loc[summary["questionnaire_city"] == city].copy()
    if sub.empty:
        return []

    order = list(cfg.COVARIATE_CATEGORY_ORDERS.get(variable, ()))
    cats = sub["category"].astype(str).tolist()
    profile_labels = cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS.get(variable, [])
    if profile_labels and set(cats).issubset(set(profile_labels)):
        cats = [c for c in profile_labels if c in cats]
    elif order:
        cats = [c for c in order if c in cats] + [c for c in cats if c not in order]
    else:
        cats = sorted(cats, key=str)

    return [
        {
            "mean": float(sub.loc[sub["category"] == cat, "mean"].iloc[0]),
            "n": int(sub.loc[sub["category"] == cat, "n"].iloc[0]),
            "members": [cat],
        }
        for cat in cats
    ]


def _weighted_group_mean(groups: list[dict]) -> float:
    total_n = sum(group["n"] for group in groups)
    if total_n == 0:
        return float(np.mean([group["mean"] for group in groups]))
    return sum(group["mean"] * group["n"] for group in groups) / total_n


def _merge_closest_group_pair(groups: list[dict]) -> list[dict]:
    best_diff = None
    best_pair = None
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            diff = abs(groups[i]["mean"] - groups[j]["mean"])
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_pair = (i, j)

    i, j = best_pair
    merged = {
        "mean": _weighted_group_mean([groups[i], groups[j]]),
        "n": groups[i]["n"] + groups[j]["n"],
        "members": groups[i]["members"] + groups[j]["members"],
    }
    remaining = [groups[k] for k in range(len(groups)) if k not in (i, j)]
    remaining.append(merged)
    return remaining


def _collapse_to_max_groups(groups: list[dict], *, max_groups: int) -> list[dict]:
    """Repeatedly merge the closest pair until at most ``max_groups`` remain."""
    collapsed = [group.copy() for group in groups]
    while len(collapsed) > max_groups:
        collapsed = _merge_closest_group_pair(collapsed)
    return collapsed


def _groups_from_profile_definition(
    summary: pd.DataFrame,
    variable: str,
    city: str,
) -> list[dict] | None:
    """Build merge groups from config when a profile grouping rule exists."""
    group_spec = cfg.QUESTIONNAIRE_PROFILE_CATEGORY_GROUPS.get(variable)
    if not group_spec:
        return None

    base = _category_rows_for_city(summary, variable, city)
    if not base:
        return None

    by_cat = {row["members"][0]: row for row in base}
    labels = cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS.get(variable, [])

    # Summary already collapsed (e.g. from category_mean_ratings_table_profile_grouped).
    if labels and set(by_cat).issubset(set(labels)):
        ordered = [by_cat[label] for label in labels if label in by_cat]
        return [
            {
                "mean": row["mean"],
                "n": row["n"],
                "members": row["members"],
                "label": row["members"][0],
            }
            for row in ordered
        ]

    groups: list[dict] = []
    for group_idx, member_cats in enumerate(group_spec):
        present = [cat for cat in member_cats if cat in by_cat]
        if not present:
            continue
        sub_groups = [by_cat[cat] for cat in present]
        label = (
            labels[group_idx]
            if group_idx < len(labels)
            else _format_group_categories(present)
        )
        groups.append(
            {
                "mean": _weighted_group_mean(sub_groups),
                "n": sum(group["n"] for group in sub_groups),
                "members": present,
                "label": label,
            }
        )
    return groups or None


def _city_variable_group_rows(
    summary: pd.DataFrame,
    variable: str,
    city: str,
    *,
    max_groups: int,
) -> list[dict]:
    """Exactly up to two rows for one variable in one city (group 1 = lower mean)."""
    profile_groups = _groups_from_profile_definition(summary, variable, city)
    if profile_groups is not None:
        groups = profile_groups
    else:
        groups = _collapse_to_max_groups(
            _category_rows_for_city(summary, variable, city),
            max_groups=max_groups,
        )
    groups = sorted(groups, key=lambda group: group["mean"])
    rows = []
    for group_no, group in enumerate(groups, start=1):
        rows.append(
            {
                "variable": variable,
                "group": group_no,
                "categories": group.get("label") or _format_group_categories(group["members"]),
                "mean": round(group["mean"], 3),
                "n": group["n"],
            }
        )
    return rows


def build_city_category_comparison_tables(
    tables_by_variable: dict[str, pd.DataFrame],
    *,
    variables: list[str] | None = None,
    variables_by_city: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """One row per response category per variable and city (no further merging)."""
    variables = variables or list(tables_by_variable)
    by_city: dict[str, list[dict]] = {city: [] for city in cfg.CITIES}

    for variable in variables:
        if variable not in tables_by_variable:
            continue
        summary = tables_by_variable[variable]
        for city in cfg.CITIES:
            if variables_by_city is not None and variable not in variables_by_city.get(city, []):
                continue
            for group_no, row in enumerate(
                _category_rows_for_city(summary, variable, city),
                start=1,
            ):
                by_city[city].append(
                    {
                        "variable": variable,
                        "group": group_no,
                        "category": row["members"][0],
                        "mean": round(row["mean"], 3),
                        "n": row["n"],
                    }
                )

    return {city: pd.DataFrame(rows) for city, rows in by_city.items()}


def variable_category_comparison_table(
    summary: pd.DataFrame,
    variable: str,
    *,
    cities: list[str] | None = None,
) -> pd.DataFrame:
    """One table for a variable: both cities as rows with deviation from city mean."""
    cities = cities or list(cfg.CITIES)
    rows: list[dict[str, object]] = []

    for city in cities:
        city_sub = summary.loc[summary["questionnaire_city"] == city]
        if city_sub.empty:
            continue
        city_mean = float((city_sub["mean"] * city_sub["n"]).sum() / city_sub["n"].sum())
        for row in _category_rows_for_city(summary, variable, city):
            rows.append(
                {
                    "City": city,
                    "Category": row["members"][0],
                    "Count": row["n"],
                    "Mean Rating": round(row["mean"], 3),
                    "Difference from Average Rating": round(row["mean"] - city_mean, 3),
                }
            )

    return pd.DataFrame(rows)


def rq1_variable_category_tables(
    tables_by_variable: dict[str, pd.DataFrame],
    *,
    variables: list[str] | None = None,
    variables_by_city: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return one comparison table per variable (Detmold and Turin rows combined)."""
    variables = variables or list(tables_by_variable)
    out: dict[str, pd.DataFrame] = {}

    for variable in variables:
        if variable not in tables_by_variable:
            continue
        cities = [
            city
            for city in cfg.CITIES
            if variables_by_city is None or variable in variables_by_city.get(city, [])
        ]
        table = variable_category_comparison_table(
            tables_by_variable[variable],
            variable,
            cities=cities,
        )
        if not table.empty:
            out[variable] = table

    return out


def annotate_category_influence(
    df: pd.DataFrame,
    *,
    margin: float | None = None,
) -> pd.DataFrame:
    """Add per-category deviation from the city-wide weighted mean for each variable."""
    if margin is None:
        margin = cfg.RQ1_CATEGORY_INFLUENCE_MARGIN
    renamed = df.rename(columns={"category": "categories"})
    annotated = annotate_group_influence(renamed, margin=margin)
    if "categories" in annotated.columns:
        annotated = annotated.rename(columns={"categories": "category"})
    return annotated


def identify_influenced_categories(
    df: pd.DataFrame,
    *,
    margin: float | None = None,
) -> pd.DataFrame:
    """Categories whose mean comfort differs from the variable city mean by at least ``margin``."""
    if margin is None:
        margin = cfg.RQ1_CATEGORY_INFLUENCE_MARGIN
    annotated = annotate_category_influence(df, margin=margin)
    cols = ["variable", "category", "mean", "n", "diff_from_city_mean"]
    if "group" in annotated.columns:
        cols.insert(1, "group")
    return annotated.loc[annotated["influenced"], cols].reset_index(drop=True)


def merge_category_means(
    summary: pd.DataFrame,
    variable: str,
    *,
    max_groups: int = 2,
    city: str | None = None,
) -> pd.DataFrame:
    """Collapse category means to at most ``max_groups`` per city."""
    cities = [city] if city else list(cfg.CITIES)
    rows = []
    for city_name in cities:
        for row in _city_variable_group_rows(
            summary, variable, city_name, max_groups=max_groups
        ):
            if city is None:
                row = {**row, "city": city_name}
            rows.append(row)
    return pd.DataFrame(rows)


def build_city_merged_category_tables(
    tables_by_variable: dict[str, pd.DataFrame],
    *,
    variables: list[str] | None = None,
    variables_by_city: dict[str, list[str]] | None = None,
    max_groups: int = 2,
) -> dict[str, pd.DataFrame]:
    """Return one table per city: two rows per variable (one row per group)."""
    variables = variables or list(tables_by_variable)
    by_city: dict[str, list[dict]] = {city: [] for city in cfg.CITIES}

    for variable in variables:
        if variable not in tables_by_variable:
            continue
        summary = tables_by_variable[variable]
        for city in cfg.CITIES:
            if variables_by_city is not None and variable not in variables_by_city.get(city, []):
                continue
            by_city[city].extend(
                _city_variable_group_rows(summary, variable, city, max_groups=max_groups)
            )

    return {city: pd.DataFrame(rows) for city, rows in by_city.items()}


def annotate_group_splits(df: pd.DataFrame, *, gap_threshold: float = 0.5) -> pd.DataFrame:
    """Add the mean gap between group 2 and group 1 for each variable."""
    parts = []
    for _, sub in df.groupby("variable", sort=False):
        sub = sub.sort_values("group").copy()
        if len(sub) < 2:
            sub["group_mean_diff"] = np.nan
            sub["notable_split"] = False
        else:
            gap = round(float(sub.iloc[1]["mean"] - sub.iloc[0]["mean"]), 3)
            sub["group_mean_diff"] = gap
            sub["notable_split"] = gap >= gap_threshold
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def filter_notable_group_splits(df: pd.DataFrame, *, gap_threshold: float = 0.5) -> pd.DataFrame:
    """Keep variables whose two groups differ by at least ``gap_threshold`` in mean comfort."""
    annotated = annotate_group_splits(df, gap_threshold=gap_threshold)
    notable = annotated.loc[annotated["notable_split"], "variable"].unique()
    cols = ["variable", "group", "categories", "mean", "group_mean_diff"]
    return annotated.loc[annotated["variable"].isin(notable), cols].reset_index(drop=True)


def annotate_group_influence(df: pd.DataFrame, *, margin: float = 0.15) -> pd.DataFrame:
    """Add per-group deviation from the city-wide mean for that variable."""
    parts = []
    for _, sub in df.groupby("variable", sort=False):
        sub = sub.copy()
        overall = float((sub["mean"] * sub["n"]).sum() / sub["n"].sum())
        sub["diff_from_city_mean"] = (sub["mean"] - overall).round(3)
        sub["influenced"] = sub["diff_from_city_mean"].abs() >= margin
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def identify_influenced_groups(merged: pd.DataFrame, *, margin: float = 0.15) -> pd.DataFrame:
    """Subset of groups whose mean differs from the variable city mean by at least ``margin``."""
    annotated = annotate_group_influence(merged, margin=margin)
    cols = ["variable", "group", "categories", "mean", "n", "diff_from_city_mean"]
    if "city" in annotated.columns:
        cols.insert(1, "city")
    return annotated.loc[annotated["influenced"], cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# RQ2 — ML dataset preparation
# ---------------------------------------------------------------------------


@dataclass
class MLPreparationArtifacts:
    questionnaire_feature_columns: list[str] = field(default_factory=list)
    attribute_feature_columns: list[str] = field(default_factory=list)
    attribute_scaler: MinMaxScaler | None = None


def _one_hot_encode(
    df: pd.DataFrame,
    columns: list[str],
    *,
    prefix_sep: str = "__",
) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return df.copy()
    encoded = pd.get_dummies(
        df[existing].astype(str).replace({"nan": np.nan, "None": np.nan}),
        columns=existing,
        prefix=[c.replace(" ", "_").replace(".", "") for c in existing],
        prefix_sep=prefix_sep,
        dtype=bool,
    )
    other_cols = [c for c in df.columns if c not in existing]
    return pd.concat(
        [df[other_cols].reset_index(drop=True), encoded.reset_index(drop=True)],
        axis=1,
    )


def _scale_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    scaler: MinMaxScaler | None = None,
    *,
    fit: bool = True,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    existing = [c for c in columns if c in df.columns and df[c].notna().any()]
    out = df.copy()
    if not existing:
        return out, scaler or MinMaxScaler()
    if scaler is None:
        scaler = MinMaxScaler()
    values = out[existing].apply(pd.to_numeric, errors="coerce")
    scaled = scaler.fit_transform(values) if fit else scaler.transform(values)
    out[existing] = scaled
    return out, scaler


def prepare_questionnaire_features(
    df: pd.DataFrame,
    *,
    drop_nulls: bool = True,
    encode_cols: list[str] | None = None,
    harmonize_profile: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    work = (
        apply_questionnaire_profile_harmonization(df)
        if harmonize_profile
        else df.copy()
    )
    work["respondent_id"] = work.index.astype(int)
    if encode_cols is None:
        encode_cols = [c for c in cfg.QUESTIONNAIRE_CATEGORICAL_COLUMNS if c in work.columns]
    else:
        encode_cols = [c for c in encode_cols if c in work.columns]
    id_cols = ["questionnaire_city"] if "questionnaire_city" in work.columns else []
    keep_cols = ["respondent_id", *id_cols, *encode_cols, *cfg.RATING_COLUMNS]
    work = work[keep_cols]
    if drop_nulls:
        work = work.dropna(subset=[*encode_cols, *id_cols]).reset_index(drop=True)
    encoded = _one_hot_encode(work, encode_cols)
    questionnaire_feature_columns = [
        c
        for c in encoded.columns
        if c not in {"respondent_id", "questionnaire_city", *cfg.RATING_COLUMNS}
        and c not in cfg.ML_EXCLUDED_FEATURE_COLUMNS
    ]
    return encoded, questionnaire_feature_columns


def reshape_ratings_long(
    df: pd.DataFrame,
    questionnaire_feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_cols = ["respondent_id", *questionnaire_feature_columns]
    for _, row in df.iterrows():
        city = row["questionnaire_city"]
        shared = {col: row[col] for col in base_cols if col in row}
        shared["questionnaire_city"] = city
        for point_num, rating_col in enumerate(cfg.RATING_COLUMNS, start=1):
            rating = row[rating_col]
            if pd.isna(rating):
                continue
            rows.append(
                {
                    **shared,
                    "point_num": point_num,
                    "point_id": cfg.point_id(str(city), point_num),
                    "rating": int(rating),
                }
            )
    return pd.DataFrame(rows)


def _attribute_feature_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical attribute columns from the loaded table."""
    exclude = set(cfg.NON_PREDICTOR_ATTRIBUTE_COLUMNS) | set(cfg.ATTRIBUTE_EXCLUDE_COLUMNS)
    categorical = [c for c in cfg.ATTRIBUTE_CATEGORICAL_COLUMNS if c in df.columns]
    numeric = [
        c
        for c in df.columns
        if c not in exclude
        and c not in categorical
        and c != "point_id"
    ]
    return numeric, categorical


def prepare_attribute_features(
    df_attr: pd.DataFrame,
    *,
    fit: bool = True,
    scaler: MinMaxScaler | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler, list[str]]:
    work = df_attr.copy()
    if "Point ID" in work.columns:
        work = work.set_index("Point ID")
    work.index.name = "point_id"
    work = work.reset_index()
    work = work.drop(columns=[c for c in cfg.ATTRIBUTE_EXCLUDE_COLUMNS if c in work.columns])

    numeric_cols, categorical_cols = _attribute_feature_column_groups(work)
    for col in numeric_cols:
        if col.endswith("(%)") or col in cfg.SVI_PROPORTION_COLUMNS:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            if work[col].dropna().le(1.0).all():
                work[col] = work[col] * 100.0

    encoded = _one_hot_encode(work, categorical_cols)
    scaled, scaler = _scale_numeric_columns(
        encoded,
        numeric_cols,
        scaler=scaler,
        fit=fit,
    )
    attribute_feature_columns = [
        c
        for c in scaled.columns
        if c not in {"point_id", *cfg.ATTRIBUTE_EXCLUDE_COLUMNS}
        and c not in categorical_cols
        and c not in cfg.ML_EXCLUDED_FEATURE_COLUMNS
    ]
    return scaled.set_index("point_id"), scaler, attribute_feature_columns


def attach_image_attributes(
    df_long: pd.DataFrame,
    df_attr_prepared: pd.DataFrame,
) -> pd.DataFrame:
    merged = df_long.merge(
        df_attr_prepared,
        left_on="point_id",
        right_index=True,
        how="inner",
        validate="many_to_one",
    )
    return merged.drop(columns=["point_num"], errors="ignore")


def build_ml_dataset(
    df_questionnaire: pd.DataFrame,
    df_attributes: pd.DataFrame,
    *,
    questionnaire_encode_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, MLPreparationArtifacts]:
    df_q, questionnaire_feature_columns = prepare_questionnaire_features(
        df_questionnaire,
        encode_cols=questionnaire_encode_cols,
    )
    df_long = reshape_ratings_long(df_q, questionnaire_feature_columns)
    df_attr_prepared, attribute_scaler, attribute_feature_columns = prepare_attribute_features(df_attributes)
    df_ml = attach_image_attributes(df_long, df_attr_prepared)
    artifacts = MLPreparationArtifacts(
        questionnaire_feature_columns=questionnaire_feature_columns,
        attribute_feature_columns=attribute_feature_columns,
        attribute_scaler=attribute_scaler,
    )
    return df_ml, artifacts


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def filter_survey_by_city(df_survey: pd.DataFrame, city: str) -> pd.DataFrame:
    return df_survey.loc[df_survey["questionnaire_city"] == city].reset_index(drop=True)


def filter_attributes_by_city(df_attr: pd.DataFrame, city: str) -> pd.DataFrame:
    prefix = f"{cfg.CITY_PREFIX[city]}-"
    if df_attr.index.name in {"Point ID", "point_id"}:
        mask = df_attr.index.astype(str).str.startswith(prefix)
        return df_attr.loc[mask].copy()
    point_col = "point_id" if "point_id" in df_attr.columns else "Point ID"
    if point_col not in df_attr.columns:
        raise ValueError("Cannot infer point identifiers for city filter.")
    mask = df_attr[point_col].astype(str).str.startswith(prefix)
    return df_attr.loc[mask].copy()


def build_ml_dataset_for_city(
    df_questionnaire: pd.DataFrame,
    df_attributes: pd.DataFrame,
    city: str,
    *,
    questionnaire_encode_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, MLPreparationArtifacts]:
    """Build a respondent–image ML table for one survey city and its image points."""
    if questionnaire_encode_cols is None:
        questionnaire_encode_cols = questionnaire_model_columns_by_city(df_questionnaire)[city]
    return build_ml_dataset(
        filter_survey_by_city(df_questionnaire, city),
        filter_attributes_by_city(df_attributes, city),
        questionnaire_encode_cols=questionnaire_encode_cols,
    )


def compute_regression_metrics(
    *,
    city: str,
    model_name: str,
    y_train: pd.Series,
    y_pred_train: np.ndarray,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    n_features: int,
) -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def _adjusted_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
        n = len(y_true)
        if n <= n_features + 1:
            return float("nan")
        r2 = r2_score(y_true, y_pred)
        return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))

    def _row(split: str, y_true: pd.Series, y_pred: np.ndarray, n: int) -> dict[str, object]:
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "city": city,
            "model": model_name,
            "split": split,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
            "adjusted_r2": _adjusted_r2(y_true, y_pred),
            "n": n,
        }

    return pd.DataFrame(
        [
            _row("train", y_train, y_pred_train, len(y_train)),
            _row("test", y_test, y_pred_test, len(y_test)),
        ]
    ).round(4)


REGRESSION_METRICS_DISPLAY_COLUMNS = {
    "city": "City",
    "model": "Model",
    "split": "Split",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R²",
    "adjusted_r2": "Adjusted R²",
    "n": "n",
}


def format_regression_metrics_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Rename regression metric columns for tables and exports."""
    rename = {
        col: label
        for col, label in REGRESSION_METRICS_DISPLAY_COLUMNS.items()
        if col in table.columns
    }
    return table.rename(columns=rename)


def prepare_city_ml_splits(
    df_survey: pd.DataFrame,
    df_attr_model: pd.DataFrame,
    *,
    test_size: float | None = None,
    random_state: int = 42,
) -> dict[str, dict[str, object]]:
    """Build respondent-level train/test splits per city for RQ2 model comparison."""
    if test_size is None:
        test_size = cfg.RQ2_TEST_SIZE
    splits: dict[str, dict[str, object]] = {}
    for city in cfg.CITIES:
        df_ml, artifacts = build_ml_dataset_for_city(df_survey, df_attr_model, city)
        train_df, test_df = train_test_split_by_respondent(
            df_ml,
            test_size=test_size,
            random_state=random_state,
        )
        X_train, y_train, feature_names = split_features_target(train_df, artifacts)
        X_test, y_test, _ = split_features_target(test_df, artifacts)
        splits[city] = {
            "df_ml": df_ml,
            "train_df": train_df,
            "test_df": test_df,
            "artifacts": artifacts,
            "feature_names": feature_names,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_groups": train_df["respondent_id"].to_numpy(),
        }
    return splits


def rq2_hyperparameter_grids() -> dict[str, dict[str, list]]:
    """Search spaces for RQ2 hyperparameter tuning (random search on train MAE)."""
    return {
        "decision_tree": {
            "max_depth": [None, 3, 5, 8, 12, 16, 20],
            "min_samples_leaf": [1, 2, 4, 8, 16],
            "min_samples_split": [2, 5, 10, 20],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "svr": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "epsilon": [0.01, 0.05, 0.1, 0.2],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
        "elastic_net": {
            "alpha": np.logspace(-4, 1, 20).tolist(),
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        },
        "lightgbm": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [-1, 3, 5, 8, 12],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [5, 10, 20, 40],
        },
        "random_forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 8, 12, 16],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.5, 0.7, 1.0],
        },
        "xgboost": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        "catboost": {
            "iterations": [200, 300, 500, 800],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 10],
        },
    }


def create_rq2_estimator(
    model_name: str,
    params: dict[str, object] | None = None,
    *,
    random_state: int | None = None,
):
    """Instantiate an RQ2 candidate regressor with optional tuned hyperparameters."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    random_state = cfg.RQ2_TUNING_RANDOM_STATE if random_state is None else random_state
    params = dict(params or {})
    if model_name == "ols":
        return LinearRegression(**params)
    if model_name == "elastic_net":
        return ElasticNet(random_state=random_state, max_iter=10_000, **params)
    if model_name == "decision_tree":
        return DecisionTreeRegressor(random_state=random_state, **params)
    if model_name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(**params)),
            ]
        )
    if model_name == "svr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", **params)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1, **params)
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0,
            **params,
        )
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(random_state=random_state, verbose=False, **params)
    raise ValueError(f"Unknown RQ2 model: {model_name}")


def _sample_rq2_param_dict(
    param_grid: dict[str, list],
    rng: np.random.Generator,
) -> dict[str, object]:
    sampled: dict[str, object] = {}
    for key, values in param_grid.items():
        value = values[int(rng.integers(0, len(values)))]
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        sampled[key] = value
    return sampled


def _sanitize_ml_column_names(columns: pd.Index) -> list[str]:
    """Make one-hot / attribute column names safe for boosted-tree learners."""
    safe_names: list[str] = []
    seen: dict[str, int] = {}
    for col in columns:
        name = re.sub(r"[\[\]<>,]", "_", str(col)).strip().replace(" ", "_")
        if not name:
            name = "feature"
        count = seen.get(name, 0)
        if count:
            name = f"{name}_{count}"
        seen[name] = count + 1
        safe_names.append(name)
    return safe_names


def rq2_design_matrix(model_name: str, X: pd.DataFrame):
    """Return model input with consistent feature names (boosters need safe column labels)."""
    if model_name in {"xgboost", "lightgbm", "catboost"}:
        out = X.astype(float).copy()
        out.columns = _sanitize_ml_column_names(out.columns)
        return out
    return X.astype(float).copy()


def rq2_lookup_hyperparameters(city: str, model_name: str) -> dict[str, object] | None:
    """Return fixed hyperparameters for a city/model when configured."""
    if not cfg.RQ2_USE_FIXED_HYPERPARAMETERS:
        return None
    city_params = cfg.RQ2_HYPERPARAMETERS.get(city)
    if city_params is None:
        return None
    if model_name not in city_params:
        return None
    return dict(city_params[model_name])


def _fit_rq2_regression_model(
    model_name: str,
    params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int,
) -> dict[str, object]:
    """Fit one RQ2 regressor with known hyperparameters and return metrics + estimator."""
    from sklearn.metrics import mean_absolute_error

    X_tr = rq2_design_matrix(model_name, X_train)
    X_te = rq2_design_matrix(model_name, X_test)
    estimator = create_rq2_estimator(model_name, params, random_state=random_state)
    estimator.fit(X_tr, y_train)
    train_mae = float(mean_absolute_error(y_train, estimator.predict(X_tr)))
    test_mae = float(mean_absolute_error(y_test, estimator.predict(X_te)))
    return {
        "model_name": model_name,
        "best_params": params,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "best_estimator": estimator,
    }


def tune_rq2_regression_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    city: str | None = None,
    random_state: int | None = None,
    n_iter: int | None = None,
    use_fixed_hyperparameters: bool | None = None,
) -> dict[str, object]:
    """Fit an RQ2 regressor using fixed config hyperparameters or random search on train MAE."""
    from sklearn.metrics import mean_absolute_error

    random_state = cfg.RQ2_TUNING_RANDOM_STATE if random_state is None else random_state
    use_fixed = (
        cfg.RQ2_USE_FIXED_HYPERPARAMETERS
        if use_fixed_hyperparameters is None
        else use_fixed_hyperparameters
    )
    if use_fixed and city is not None:
        fixed_params = rq2_lookup_hyperparameters(city, model_name)
        if fixed_params is not None:
            return _fit_rq2_regression_model(
                model_name,
                fixed_params,
                X_train,
                y_train,
                X_test,
                y_test,
                random_state=random_state,
            )

    n_iter = cfg.RQ2_TUNING_N_ITER if n_iter is None else n_iter
    rng = np.random.default_rng(random_state)

    if model_name == "ols":
        return _fit_rq2_regression_model(
            model_name,
            {},
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=random_state,
        )

    param_grid = rq2_hyperparameter_grids()[model_name]
    if model_name == "elastic_net":
        n_iter = min(n_iter, len(param_grid["alpha"]) * len(param_grid["l1_ratio"]))

    best_train_mae = float("inf")
    best_params: dict[str, object] = {}
    X_tr = rq2_design_matrix(model_name, X_train)
    X_te = rq2_design_matrix(model_name, X_test)
    for _ in range(n_iter):
        params = _sample_rq2_param_dict(param_grid, rng)
        estimator = create_rq2_estimator(model_name, params, random_state=random_state)
        estimator.fit(X_tr, y_train)
        train_mae = float(mean_absolute_error(y_train, estimator.predict(X_tr)))
        if train_mae < best_train_mae:
            best_train_mae = train_mae
            best_params = params

    return _fit_rq2_regression_model(
        model_name,
        best_params,
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=random_state,
    )


def format_rq2_tuning_row(city: str, tuning_result: dict[str, object]) -> dict[str, object]:
    """Flatten tuning output for tabular export."""
    return {
        "city": city,
        "model": tuning_result["model_name"],
        "train_mae": round(float(tuning_result["train_mae"]), 4),
        "test_mae": round(float(tuning_result["test_mae"]), 4),
        "best_params": json.dumps(tuning_result["best_params"], sort_keys=True),
    }


def build_rq2_city_performance_table(
    comparison: pd.DataFrame,
    city: str,
    *,
    split: str = "test",
) -> pd.DataFrame:
    """Build a VATA Table 3-style performance table for one city (test metrics)."""
    subset = comparison.loc[
        (comparison["city"] == city) & (comparison["split"] == split)
    ]
    if subset.empty:
        raise ValueError(f"No rows with city={city!r} and split={split!r}.")

    rows: list[dict[str, object]] = []
    for model_name in cfg.RQ2_TUNED_MODELS:
        model_rows = subset.loc[subset["model"] == model_name]
        if model_rows.empty:
            continue
        record = model_rows.iloc[0]
        rows.append(
            {
                "Model": cfg.RQ2_MODEL_REPORT_NAMES.get(model_name, model_name),
                "MAE": round(float(record["mae"]), 4),
                "MSE": round(float(record["mse"]), 4),
                "RMSE": round(float(record["rmse"]), 4),
                "Adjusted R2": round(float(record["adjusted_r2"]), 4),
            }
        )
    if not rows:
        raise ValueError(f"No tuned models found for city={city!r}.")

    return (
        pd.DataFrame(rows)
        .sort_values("MAE", ascending=False)
        .reset_index(drop=True)
    )


def build_rq2_performance_tables_by_city(
    comparison: pd.DataFrame,
    *,
    split: str = "test",
) -> dict[str, pd.DataFrame]:
    """Return per-city model performance tables keyed by city."""
    return {
        city: build_rq2_city_performance_table(comparison, city, split=split)
        for city in cfg.CITIES
        if city in comparison["city"].unique()
    }


def evaluate_regression_model(
    estimator,
    *,
    city: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> tuple[object, pd.DataFrame]:
    """Fit a regression estimator and return it with train/test metrics."""
    fitted = estimator
    fitted.fit(X_train, y_train)
    y_pred_train = fitted.predict(X_train)
    y_pred_test = fitted.predict(X_test)
    metrics = compute_regression_metrics(
        city=city,
        model_name=model_name,
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test,
        n_features=len(feature_names),
    )
    return fitted, metrics


def select_best_model_per_city(
    comparison: pd.DataFrame,
    *,
    split: str = "test",
    metric: str = "mae",
) -> pd.DataFrame:
    """Return one row per city for the model with the lowest error on the chosen split."""
    subset = comparison.loc[comparison["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"No rows with split={split!r} in comparison table.")
    best_idx = subset.groupby("city", sort=False)[metric].idxmin()
    return subset.loc[best_idx].reset_index(drop=True)


def build_shap_explainer(model: object, X_background: pd.DataFrame):
    """Choose a SHAP explainer compatible with the fitted model class."""
    import shap
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        return shap.Explainer(model.predict, X_background)

    class_name = model.__class__.__name__
    if class_name in {
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "ExtraTreesRegressor",
        "LGBMRegressor",
        "XGBRegressor",
        "CatBoostRegressor",
    }:
        return shap.TreeExplainer(model)
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, X_background)
    return shap.Explainer(model.predict, X_background)


def prepare_model_attribute_data(
    df_attr_raw: pd.DataFrame,
    attr_layers_sheet: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return variation- and collinearity-screened attribute table for RQ2 modelling."""
    attr_quality = attribute_quality_report_by_city(df_attr_raw, layers=attr_layers_sheet)
    kept = intersection_kept_items(attr_quality, item_col="attribute")
    attr_layers_all = filter_layers_by_columns(attr_layers_sheet, kept)
    predictors = all_physical_predictors(attr_layers_all)
    predictors_model = [p for p in predictors if p not in cfg.COLLINEARITY_PAIRWISE_MANUAL_DROPS]
    attr_layers_model = filter_layers_by_columns(attr_layers_all, set(predictors_model))
    df_attr_model = filter_attribute_dataframe(df_attr_raw, attr_layers_model)
    return df_attr_model, attr_layers_model


def split_features_target(
    df: pd.DataFrame,
    artifacts: MLPreparationArtifacts | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if artifacts is None:
        feature_cols = [
            c
            for c in df.columns
            if c not in {"rating", "questionnaire_city", *cfg.ML_EXCLUDED_FEATURE_COLUMNS}
        ]
    else:
        feature_cols = [
            *artifacts.questionnaire_feature_columns,
            *artifacts.attribute_feature_columns,
        ]
        feature_cols = [
            c
            for c in feature_cols
            if c in df.columns and c not in cfg.ML_EXCLUDED_FEATURE_COLUMNS
        ]
    x = df[feature_cols].astype(float).fillna(0.0)
    y = df["rating"].astype(int)
    return x, y, feature_cols


def train_test_split_by_respondent(
    df: pd.DataFrame,
    *,
    test_size: float | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size is None:
        test_size = cfg.RQ2_TEST_SIZE
    rng = np.random.default_rng(random_state)
    respondent_ids = df["respondent_id"].unique()
    rng.shuffle(respondent_ids)
    n_test = max(1, int(round(len(respondent_ids) * test_size)))
    test_ids = set(respondent_ids[:n_test])
    train_ids = set(respondent_ids[n_test:])
    train_df = df[df["respondent_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["respondent_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# RQ2 — SHAP reporting (encoded model features, human-readable labels)
# ---------------------------------------------------------------------------


def _feature_encoding_prefix(column: str) -> str:
    return column.replace(" ", "_").replace(".", "")


def _parse_encoded_feature(feature: str) -> tuple[str, str | None]:
    """Return source column and one-hot level for encoded questionnaire features."""
    if "__" not in feature:
        return feature, None
    prefix, level = feature.split("__", 1)
    for col in [*cfg.QUESTIONNAIRE_CATEGORICAL_COLUMNS, *cfg.ATTRIBUTE_CATEGORICAL_COLUMNS]:
        if _feature_encoding_prefix(col) == prefix:
            return col, level
    return prefix, level


def _keyword_display_label(column: str, keyword: str) -> str:
    profile_labels = cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS.get(column, [])
    if keyword in profile_labels:
        return keyword
    if column == "birthplace":
        return cfg.BIRTHPLACE_KEYWORD_LABELS.get(keyword, keyword.replace("_", " "))
    for english, mapped in cfg.KEYWORD_MAPS.get(column, {}).items():
        if mapped == keyword:
            return english
    return keyword.replace("_", " ")


def human_readable_ml_feature_name(feature: str) -> str:
    """Readable label for a model feature, e.g. ``gender__m`` → ``Gender (Male)``."""
    column, level = _parse_encoded_feature(feature)
    if level is None:
        return str(feature)
    variable_label = cfg.QUESTIONNAIRE_VARIABLE_LABELS.get(
        column,
        column.replace("_", " ").title(),
    )
    return f"{variable_label} ({_keyword_display_label(column, level)})"


def _questionnaire_shap_group_name(column: str) -> str | None:
    for group_name, variables in cfg.QUESTIONNAIRE_SHAP_GROUPS.items():
        if column in variables:
            return group_name
    return None


def _attribute_shap_group_name(feature: str, attr_layers: dict[str, list[str]]) -> str:
    column, _ = _parse_encoded_feature(feature)
    if column in attr_layers.get("Layer 1 — RS / GIS", []):
        return "RS / GIS"
    if column in attr_layers.get("Layer 3a — SVI Pixel %", []):
        return "Streetscape"
    return "Other"


def _shap_values_by_feature(
    feature_names: list[str],
    shap_values: np.ndarray,
) -> dict[str, tuple[float, float]]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    return {
        feature: (float(abs_val), float(signed_val))
        for feature, abs_val, signed_val in zip(feature_names, mean_abs, mean_signed)
    }


def _ordered_questionnaire_levels(column: str) -> list[str]:
    """Canonical category order for one questionnaire column."""
    if column == "gender":
        return ["f", "m"]
    if column in cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS:
        return list(cfg.QUESTIONNAIRE_PROFILE_GROUP_LABELS[column])
    if column == "city_relationship":
        return ["resident", "commuter"]
    order = cfg.COVARIATE_CATEGORY_ORDERS.get(column)
    if order:
        return list(order)
    return list(cfg.KEYWORD_MAPS.get(column, {}).values())


def _ordered_questionnaire_features(questionnaire_features: list[str]) -> list[str]:
    """Questionnaire model features in group/variable/category order."""
    feature_set = set(questionnaire_features)
    ordered: list[str] = []
    for group in cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER:
        for column in cfg.QUESTIONNAIRE_SHAP_GROUPS[group]:
            for level in _ordered_questionnaire_levels(column):
                feature = f"{_feature_encoding_prefix(column)}__{level}"
                if feature in feature_set:
                    ordered.append(feature)
    for feature in questionnaire_features:
        if feature not in ordered:
            ordered.append(feature)
    return ordered


def _canonical_attribute_comparison_order() -> list[tuple[str, str]]:
    """Streetscape / RS-GIS attribute labels in layer order."""
    rows: list[tuple[str, str]] = []
    for group, layer_cols in (
        ("RS / GIS", cfg.LAYER_1_RS_GIS),
        ("Streetscape", cfg.LAYER_3A_SVI_PIXEL_PCT),
    ):
        for column in layer_cols:
            label = human_readable_ml_feature_name(column)
            rows.append((group, label))
    return rows


def _comparison_table_row_order(
    tables_by_city: dict[str, pd.DataFrame],
    *,
    group_filter: str | None = None,
    group_order: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Preferred row order for a SHAP comparison table."""
    if group_filter is None and group_order == list(cfg.ATTRIBUTE_SHAP_GROUP_ORDER):
        for city in (*cfg.CITIES, *tables_by_city.keys()):
            if city not in tables_by_city:
                continue
            city_table = tables_by_city[city]
            return list(
                zip(
                    city_table["group"].astype(str),
                    city_table["feature"].astype(str),
                )
            )
        return _canonical_attribute_comparison_order()

    canonical: list[tuple[str, str]] = []
    if group_filter:
        groups = [group_filter]
    else:
        groups = group_order or list(cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER)
    for group in groups:
        for column in cfg.QUESTIONNAIRE_SHAP_GROUPS.get(group, []):
            for level in _ordered_questionnaire_levels(column):
                feature = f"{_feature_encoding_prefix(column)}__{level}"
                canonical.append((group, human_readable_ml_feature_name(feature)))
    return canonical


def _sort_shap_comparison_rows(
    table: pd.DataFrame,
    row_order: list[tuple[str, str]],
) -> pd.DataFrame:
    """Sort comparison table by Group then variable/category order."""
    order_map = {pair: idx for idx, pair in enumerate(row_order)}
    base = len(order_map)
    row_orders: list[int] = []
    for group, attribute in zip(table["Group"].astype(str), table["Attribute"].astype(str)):
        row_orders.append(order_map.get((group, attribute), base))
    out = table.copy()
    out["_row_order"] = row_orders
    out = (
        out.sort_values(["Group", "_row_order", "Attribute"], kind="stable")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    return out


def _finalize_shap_table(
    rows: list[dict[str, object]],
    group_order: list[str],
) -> pd.DataFrame:
    if not rows:
        raise ValueError("No SHAP rows to report.")
    table = pd.DataFrame(rows)
    categories = [*group_order, "Other"]
    table["group"] = pd.Categorical(table["group"], categories=categories, ordered=True)
    table["_row_order"] = range(len(table))
    return (
        table.sort_values(["group", "_row_order"], kind="stable")
        .drop(columns="_row_order")
        .loc[:, ["group", "feature", "mean_abs_shap", "mean_shap"]]
        .reset_index(drop=True)
    )


def build_shap_attribute_table(
    feature_names: list[str],
    shap_values: np.ndarray,
    *,
    attribute_features: list[str],
    attr_layers: dict[str, list[str]],
) -> pd.DataFrame:
    """SHAP table for streetscape / RS-GIS model features."""
    shap_lookup = _shap_values_by_feature(feature_names, shap_values)
    rows: list[dict[str, object]] = []
    for feature in attribute_features:
        if feature not in shap_lookup:
            continue
        mean_abs_shap, mean_shap = shap_lookup[feature]
        rows.append(
            {
                "group": _attribute_shap_group_name(feature, attr_layers),
                "feature": human_readable_ml_feature_name(feature),
                "mean_abs_shap": mean_abs_shap,
                "mean_shap": mean_shap,
            }
        )
    return _finalize_shap_table(rows, cfg.ATTRIBUTE_SHAP_GROUP_ORDER)


def build_shap_questionnaire_table(
    feature_names: list[str],
    shap_values: np.ndarray,
    *,
    questionnaire_features: list[str],
) -> pd.DataFrame:
    """SHAP table for one-hot questionnaire model features."""
    shap_lookup = _shap_values_by_feature(feature_names, shap_values)
    rows: list[dict[str, object]] = []
    for feature in _ordered_questionnaire_features(questionnaire_features):
        if feature not in shap_lookup:
            continue
        column, _ = _parse_encoded_feature(feature)
        group = _questionnaire_shap_group_name(column)
        if group is None:
            continue
        mean_abs_shap, mean_shap = shap_lookup[feature]
        rows.append(
            {
                "group": group,
                "feature": human_readable_ml_feature_name(feature),
                "mean_abs_shap": mean_abs_shap,
                "mean_shap": mean_shap,
            }
        )
    return _finalize_shap_table(rows, cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER)


def build_shap_cross_city_comparison_table(
    tables_by_city: dict[str, pd.DataFrame],
    *,
    group_filter: str | None = None,
    group_order: list[str] | None = None,
    cities: tuple[str, ...] = ("Turin", "Detmold"),
) -> pd.DataFrame:
    """Merge per-city SHAP tables into cross-city signed and absolute mean SHAP columns."""
    if group_order is None:
        group_order = (
            [group_filter]
            if group_filter
            else list(cfg.ATTRIBUTE_SHAP_GROUP_ORDER)
        )

    merged: pd.DataFrame | None = None
    for city in cities:
        if city not in tables_by_city:
            raise KeyError(f"Missing SHAP table for city={city!r}.")
        city_table = tables_by_city[city].copy()
        if group_filter:
            city_table = city_table.loc[city_table["group"] == group_filter].copy()
        city_table = city_table.rename(
            columns={
                "group": "Group",
                "feature": "Attribute",
                "mean_shap": f"{city} Mean SHAP",
                "mean_abs_shap": f"{city} Mean |SHAP|",
            }
        )
        city_table = city_table.loc[
            :,
            ["Group", "Attribute", f"{city} Mean SHAP", f"{city} Mean |SHAP|"],
        ]
        if merged is None:
            merged = city_table
        else:
            merged = merged.merge(city_table, on=["Group", "Attribute"], how="outer")

    if merged is None or merged.empty:
        raise ValueError("No SHAP rows available for the requested comparison.")

    merged["Group"] = pd.Categorical(merged["Group"], categories=group_order, ordered=True)
    shap_cols: list[str] = []
    for city in cities:
        shap_cols.extend([f"{city} Mean SHAP", f"{city} Mean |SHAP|"])
    row_order = _comparison_table_row_order(
        tables_by_city,
        group_filter=group_filter,
        group_order=group_order,
    )
    merged = _sort_shap_comparison_rows(merged, row_order)
    return merged.loc[:, ["Group", "Attribute", *shap_cols]]


def build_shap_questionnaire_variable_table(
    feature_names: list[str],
    shap_values: np.ndarray,
    *,
    questionnaire_features: list[str],
) -> pd.DataFrame:
    """One row per questionnaire source variable; mean |SHAP| averaged across response categories."""
    shap_lookup = _shap_values_by_feature(feature_names, shap_values)
    rows: list[dict[str, object]] = []
    for group in cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER:
        for column in cfg.QUESTIONNAIRE_SHAP_GROUPS.get(group, []):
            abs_vals: list[float] = []
            signed_vals: list[float] = []
            for level in _ordered_questionnaire_levels(column):
                feature = f"{_feature_encoding_prefix(column)}__{level}"
                if feature not in shap_lookup:
                    continue
                mean_abs_shap, mean_shap = shap_lookup[feature]
                abs_vals.append(mean_abs_shap)
                signed_vals.append(mean_shap)
            if not abs_vals:
                continue
            rows.append(
                {
                    "group": group,
                    "feature": cfg.QUESTIONNAIRE_VARIABLE_LABELS.get(
                        column,
                        column.replace("_", " ").title(),
                    ),
                    "mean_abs_shap": float(np.mean(abs_vals)),
                    "mean_shap": float(np.mean(signed_vals)),
                }
            )
    return _finalize_shap_table(rows, cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER)


def _combine_attr_and_questionnaire_shap_tables(
    attr_table: pd.DataFrame,
    questionnaire_table: pd.DataFrame,
) -> pd.DataFrame:
    """Stack attribute and questionnaire-variable SHAP tables with a predictor-type label."""
    attr_part = attr_table.copy()
    attr_part["predictor_type"] = "Streetscape attribute"
    quest_part = questionnaire_table.copy()
    quest_part["predictor_type"] = "Questionnaire"
    combined = pd.concat([attr_part, quest_part], ignore_index=True)
    group_order = [
        *cfg.ATTRIBUTE_SHAP_GROUP_ORDER,
        *cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER,
    ]
    categories = [*group_order, "Other"]
    combined["group"] = pd.Categorical(combined["group"], categories=categories, ordered=True)
    combined["_row_order"] = range(len(combined))
    return (
        combined.sort_values(["group", "_row_order"], kind="stable")
        .drop(columns="_row_order")
        .loc[:, ["predictor_type", "group", "feature", "mean_abs_shap", "mean_shap"]]
        .reset_index(drop=True)
    )


def build_shap_global_importance_comparison_table(
    attr_tables_by_city: dict[str, pd.DataFrame],
    questionnaire_tables_by_city: dict[str, pd.DataFrame],
    *,
    cities: tuple[str, ...] = ("Turin", "Detmold"),
) -> pd.DataFrame:
    """Cross-city mean |SHAP| for streetscape attributes and questionnaire variables."""
    combined_by_city = {
        city: _combine_attr_and_questionnaire_shap_tables(
            attr_tables_by_city[city],
            questionnaire_tables_by_city[city],
        )
        for city in cities
    }
    merged: pd.DataFrame | None = None
    for city in cities:
        city_table = combined_by_city[city].rename(
            columns={
                "group": "Group",
                "feature": "Attribute",
                "predictor_type": "Macro Group",
                "mean_abs_shap": f"{city} Mean |SHAP|",
            }
        )
        city_table = city_table.loc[:, ["Macro Group", "Group", "Attribute", f"{city} Mean |SHAP|"]]
        if merged is None:
            merged = city_table
        else:
            merged = merged.merge(city_table, on=["Macro Group", "Group", "Attribute"], how="outer")

    if merged is None or merged.empty:
        raise ValueError("No SHAP rows available for global importance comparison.")

    group_order = [
        *cfg.ATTRIBUTE_SHAP_GROUP_ORDER,
        *cfg.QUESTIONNAIRE_SHAP_GROUP_ORDER,
    ]
    merged["Group"] = pd.Categorical(merged["Group"], categories=group_order, ordered=True)
    row_order = _comparison_table_row_order(
        {city: combined_by_city[city].rename(columns={"feature": "feature"}) for city in cities},
        group_order=group_order,
    )
    # Row order uses (group, feature) from attribute table helper — rebuild from combined
    order_map: dict[tuple[str, str], int] = {}
    idx = 0
    for predictor in ("Streetscape attribute", "Questionnaire"):
        subset = merged.loc[merged["Macro Group"] == predictor]
        for group in group_order:
            group_rows = subset.loc[subset["Group"] == group, "Attribute"]
            for attribute in group_rows:
                order_map[(str(group), str(attribute))] = idx
                idx += 1
    merged["_row_order"] = [
        order_map.get((str(g), str(a)), idx)
        for g, a in zip(merged["Group"].astype(str), merged["Attribute"].astype(str))
    ]
    shap_cols = [f"{city} Mean |SHAP|" for city in cities]
    return (
        merged.sort_values(["Macro Group", "Group", "_row_order", "Attribute"], kind="stable")
        .drop(columns="_row_order")
        .loc[:, ["Macro Group", "Group", "Attribute", *shap_cols]]
        .reset_index(drop=True)
    )


def summarize_shap_importance_by_predictor(
    attr_table: pd.DataFrame,
    questionnaire_table: pd.DataFrame,
) -> pd.DataFrame:
    """Block-level summary: mean |SHAP| by predictor group."""
    records = [
        {
            "Group": "Streetscape attribute",
            "Mean |SHAP|": float(attr_table["mean_abs_shap"].mean()),
        },
        {
            "Group": "Questionnaire",
            "Mean |SHAP|": float(questionnaire_table["mean_abs_shap"].mean()),
        },
    ]
    return pd.DataFrame(records)


def plot_shap_global_importance_figure(
    tables_by_city: dict[str, pd.DataFrame],
    *,
    title: str,
    cities: list[str] | None = None,
) -> plt.Figure:
    """Side-by-side mean |SHAP| bars for attributes and questionnaire variables."""
    cities = list(cities or cfg.CITIES)
    predictor_colors = {
        "Streetscape attribute": cfg.ATTRIBUTE_LAYER_BAND_COLORS["Layer 3a — SVI Pixel %"],
        "Questionnaire": cfg.SHAP_GROUP_COLORS["Demographics"],
    }

    fig, axes = plt.subplots(1, len(cities), figsize=(8.5 * len(cities), 9.0), squeeze=False)
    axes = axes[0]

    for ax, city in zip(axes, cities):
        plot_df = tables_by_city[city].sort_values("mean_abs_shap", ascending=True)
        colors = [predictor_colors.get(p, "#888888") for p in plot_df["predictor_type"]]
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color=colors, alpha=0.92)
        ax.set_title(city, fontweight="semibold", fontsize=11)
        ax.set_xlabel("Mean |SHAP|")
        sns.despine(ax=ax)

    handles = [
        Patch(color=color, label=label, alpha=0.92)
        for label, color in predictor_colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontweight="semibold", y=1.01)
    fig.tight_layout()
    return fig


combine_shap_global_importance_tables = _combine_attr_and_questionnaire_shap_tables


def _shap_row_order_from_table(
    table: pd.DataFrame,
    *,
    group_order: list[str],
    label_col: str = "feature",
    value_col: str = "mean_shap",
) -> list[tuple[str, str]]:
    """Return (group, feature) pairs in plot order for one SHAP table."""
    row_order: list[tuple[str, str]] = []
    for group_name in group_order:
        group_df = (
            table.loc[table["group"] == group_name, [label_col, value_col, "group"]]
            .sort_values(value_col, ascending=False)
        )
        for record in group_df.itertuples(index=False):
            row_order.append((group_name, str(getattr(record, label_col))))
    return row_order


def _shap_grouped_bar_plot_data(
    table: pd.DataFrame,
    *,
    group_order: list[str],
    label_col: str = "feature",
    value_col: str = "mean_shap",
    group_gap: float = 1.0,
    row_order: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, float, float]]]:
    """Prepare row order, colours metadata, and group spans for grouped SHAP bars."""
    rows: list[dict[str, object]] = []
    if row_order is None:
        row_order = _shap_row_order_from_table(
            table,
            group_order=group_order,
            label_col=label_col,
            value_col=value_col,
        )

    for group_name, feature in row_order:
        group_df = table.loc[table["group"] == group_name]
        if group_df.empty:
            continue
        match = group_df.loc[group_df[label_col].astype(str) == feature]
        if match.empty:
            continue
        record = match.iloc[0]
        rows.append(
            {
                "group": group_name,
                "feature": str(record[label_col]),
                "shap_value": float(record[value_col]),
            }
        )
    if not rows:
        raise ValueError("No rows in table match the requested SHAP groups.")

    plot_df = pd.DataFrame(rows)
    y_positions: list[float] = []
    group_spans: list[tuple[str, float, float]] = []
    y = float(len(plot_df) - 1)
    for group_name in group_order:
        group_rows = plot_df.loc[plot_df["group"] == group_name]
        if group_rows.empty:
            continue
        start_y = y
        for _ in group_rows.itertuples():
            y_positions.append(y)
            y -= 1.0
        group_spans.append((group_name, y + 1.0, start_y))
        y -= group_gap

    return plot_df.assign(y=y_positions), group_spans


def _plot_shap_grouped_importance_on_ax(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    group_order: list[str],
    group_colors: dict[str, str],
    group_gap: float = 1.0,
    label_col: str = "feature",
    value_col: str = "mean_shap",
    panel_title: str | None = None,
    row_order: list[tuple[str, str]] | None = None,
    show_ylabels: bool = True,
) -> list[tuple[str, float, float]]:
    """Draw grouped signed SHAP bars on one axes; return group spans for side labels."""
    try:
        plot_df, group_spans = _shap_grouped_bar_plot_data(
            table,
            group_order=group_order,
            label_col=label_col,
            value_col=value_col,
            group_gap=group_gap,
            row_order=row_order,
        )
    except ValueError:
        ax.text(0.5, 0.5, "No SHAP values", ha="center", va="center", transform=ax.transAxes)
        if panel_title:
            ax.set_title(panel_title, fontweight="semibold", fontsize=11)
        sns.despine(ax=ax, left=False)
        return []

    bar_colors = [group_colors.get(g, "#888888") for g in plot_df["group"]]
    labels = [str(v) for v in plot_df["feature"]]
    ax.barh(plot_df["y"], plot_df["shap_value"], height=0.72, color=bar_colors, alpha=0.92)
    ax.axvline(0, color="#333333", linewidth=0.8, zorder=1)
    ax.set_yticks(plot_df["y"])
    if show_ylabels:
        ax.set_yticklabels(labels, ha="right")
        ax.tick_params(axis="y", pad=4, labelsize=8, labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
    y_positions = plot_df["y"].tolist()
    ax.set_ylim(min(y_positions) - 0.6, max(y_positions) + 0.6)
    if panel_title:
        ax.set_title(panel_title, fontweight="semibold", fontsize=11)
    sns.despine(ax=ax, left=False)
    return group_spans


def _shap_group_label_x(
    fig: plt.Figure,
    ax: plt.Axes,
    label_texts: list[str],
    *,
    fontsize: float = 10,
    gap: float = 0.018,
) -> float:
    """Figure-x for vertical group labels, left of y tick labels without overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_bboxes = [
        label.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for label in ax.get_yticklabels()
        if label.get_text().strip()
    ]
    tick_left = min(bb.x0 for bb in tick_bboxes) if tick_bboxes else ax.get_position().x0

    max_half_width = 0.0
    for text in label_texts:
        probe = fig.text(0, 0, text, fontsize=fontsize, rotation=90, ha="center", va="center")
        probe_bb = probe.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        max_half_width = max(max_half_width, (probe_bb.x1 - probe_bb.x0) / 2.0)
        probe.remove()

    return tick_left - gap - max_half_width


def _add_shap_top_level_group_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    top_level_groups: dict[str, list[str]],
    group_spans: list[tuple[str, float, float]],
    fontsize: float = 10,
    gap: float = 0.018,
) -> None:
    """Place vertical top-level group labels just left of the y-axis."""
    from matplotlib.transforms import blended_transform_factory

    span_lookup = {name: (y_min, y_max) for name, y_min, y_max in group_spans}
    label_texts = [name for name, child_groups in top_level_groups.items() if child_groups]
    if not label_texts:
        return

    label_x = _shap_group_label_x(
        fig,
        ax,
        label_texts,
        fontsize=fontsize,
        gap=gap,
    )
    label_trans = blended_transform_factory(fig.transFigure, ax.transData)
    for top_name, child_groups in top_level_groups.items():
        child_spans = [span_lookup[g] for g in child_groups if g in span_lookup]
        if not child_spans:
            continue
        y_min = min(span[0] for span in child_spans)
        y_max = max(span[1] for span in child_spans)
        fig.text(
            label_x,
            (y_min + y_max) / 2.0,
            top_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="semibold",
            color="#1a1a1a",
            transform=label_trans,
        )


def plot_shap_grouped_importance_bar(
    table: pd.DataFrame,
    *,
    group_order: list[str],
    group_colors: dict[str, str],
    title: str,
    top_level_groups: dict[str, list[str]] | None = None,
    group_gap: float = 1.0,
    label_col: str = "feature",
    value_col: str = "mean_shap",
) -> plt.Figure:
    """Horizontal grouped SHAP bars with vertical labels and per-group colours."""
    plot_df, group_spans = _shap_grouped_bar_plot_data(
        table,
        group_order=group_order,
        label_col=label_col,
        value_col=value_col,
        group_gap=group_gap,
    )
    height = max(5.0, 0.32 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, height))
    _plot_shap_grouped_importance_on_ax(
        ax,
        table,
        group_order=group_order,
        group_colors=group_colors,
        group_gap=group_gap,
        label_col=label_col,
        value_col=value_col,
    )

    x_label = "Mean |SHAP value|" if value_col == "mean_abs_shap" else "Mean SHAP value"
    ax.set_xlabel(x_label)
    ax.set_title(title, fontweight="semibold")
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.98])

    if top_level_groups and group_spans:
        _add_shap_top_level_group_labels(
            fig,
            ax,
            top_level_groups=top_level_groups,
            group_spans=group_spans,
        )

    return fig


def plot_shap_city_comparison_figure(
    tables_by_city: dict[str, pd.DataFrame],
    *,
    group_order: list[str],
    group_colors: dict[str, str],
    title: str,
    cities: list[str] | None = None,
    group_filter: str | None = None,
    top_level_groups: dict[str, list[str]] | None = None,
    value_col: str = "mean_shap",
    reference_city: str | None = None,
) -> plt.Figure:
    """Side-by-side SHAP bar panels (e.g. Detmold | Turin) for direct comparison."""
    cities = list(cities or cfg.CITIES)
    if group_filter:
        tables_by_city = {
            city: table.loc[table["group"] == group_filter].copy()
            for city, table in tables_by_city.items()
        }
        group_order = [group_filter]

    reference_city = reference_city or cities[0]
    if reference_city not in tables_by_city:
        raise KeyError(f"Missing SHAP table for reference_city={reference_city!r}.")
    row_order = _shap_row_order_from_table(
        tables_by_city[reference_city],
        group_order=group_order,
        value_col=value_col,
    )

    plot_df, _ = _shap_grouped_bar_plot_data(
        tables_by_city[reference_city],
        group_order=group_order,
        value_col=value_col,
        row_order=row_order,
    )
    height = max(5.0, 0.30 * len(plot_df) + 2.0)

    fig, axes = plt.subplots(
        1,
        len(cities),
        figsize=(8.5 * len(cities), height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    x_label = "Mean |SHAP value|" if value_col == "mean_abs_shap" else "Mean SHAP value"
    left_spans: list[tuple[str, float, float]] = []
    for ax, city in zip(axes, cities):
        spans = _plot_shap_grouped_importance_on_ax(
            ax,
            tables_by_city[city],
            group_order=group_order,
            group_colors=group_colors,
            value_col=value_col,
            panel_title=city,
            row_order=row_order,
            show_ylabels=(ax is axes[0]),
        )
        if not left_spans:
            left_spans = spans
        if ax is axes[-1]:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel("")

    ref_plot_df, _ = _shap_grouped_bar_plot_data(
        tables_by_city[reference_city],
        group_order=group_order,
        value_col=value_col,
        row_order=row_order,
    )
    axes[0].set_yticks(ref_plot_df["y"])
    axes[0].set_yticklabels([str(v) for v in ref_plot_df["feature"]], ha="right")
    axes[0].tick_params(axis="y", pad=4, labelsize=8, labelleft=True)

    fig.suptitle(title, fontweight="semibold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.96])

    if top_level_groups and left_spans:
        _add_shap_top_level_group_labels(
            fig,
            axes[0],
            top_level_groups=top_level_groups,
            group_spans=left_spans,
        )

    return fig


def plot_shap_dual_city_figure(
    tables_by_city: dict[str, pd.DataFrame],
    *,
    group_order: list[str],
    title: str,
    cities: list[str] | None = None,
    group_filter: str | None = None,
    top_level_groups: dict[str, list[str]] | None = None,
    value_col: str = "mean_shap",
    reference_city: str | None = None,
    bar_height: float = 0.30,
) -> plt.Figure:
    """Single-panel horizontal SHAP bars with one bar per city per feature."""
    cities = list(cities or cfg.CITIES)
    if group_filter:
        tables_by_city = {
            city: table.loc[table["group"] == group_filter].copy()
            for city, table in tables_by_city.items()
        }
        group_order = [group_filter]

    reference_city = reference_city or cities[0]
    if reference_city not in tables_by_city:
        raise KeyError(f"Missing SHAP table for reference_city={reference_city!r}.")
    row_order = _shap_row_order_from_table(
        tables_by_city[reference_city],
        group_order=group_order,
        value_col=value_col,
    )
    ref_plot_df, group_spans = _shap_grouped_bar_plot_data(
        tables_by_city[reference_city],
        group_order=group_order,
        value_col=value_col,
        row_order=row_order,
    )

    height = max(5.0, 0.32 * len(ref_plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(9.0, height))
    x_label = "Mean |SHAP value|" if value_col == "mean_abs_shap" else "Mean SHAP value"

    for record in ref_plot_df.itertuples(index=False):
        y_center = float(record.y)
        group_name = str(record.group)
        feature_name = str(record.feature)
        for city_idx, city in enumerate(cities):
            city_table = tables_by_city[city]
            match = city_table.loc[
                (city_table["group"] == group_name)
                & (city_table["feature"].astype(str) == feature_name)
            ]
            if match.empty:
                continue
            shap_value = float(match.iloc[0][value_col])
            y_bar = y_center + ((len(cities) - 1) / 2 - city_idx) * bar_height
            ax.barh(
                y_bar,
                shap_value,
                height=bar_height * 0.92,
                color=cfg.CITY_COLORS.get(city, "#888888"),
                alpha=0.92,
            )

    ax.axvline(0, color="#333333", linewidth=0.8, zorder=1)
    ax.set_yticks(ref_plot_df["y"])
    ax.set_yticklabels([str(v) for v in ref_plot_df["feature"]], ha="right")
    ax.tick_params(axis="y", pad=4, labelsize=8)
    y_positions = ref_plot_df["y"].tolist()
    ax.set_ylim(min(y_positions) - 0.6, max(y_positions) + 0.6)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontweight="semibold")
    sns.despine(ax=ax, left=False)

    handles = [
        Patch(color=cfg.CITY_COLORS.get(city, "#888888"), label=city, alpha=0.92)
        for city in cities
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.98])

    if top_level_groups and group_spans:
        _add_shap_top_level_group_labels(
            fig,
            ax,
            top_level_groups=top_level_groups,
            group_spans=group_spans,
        )

    return fig


def plot_shap_importance_bar(
    table: pd.DataFrame,
    *,
    label_col: str,
    title: str,
    top_n: int | None = None,
    color: str = "#3D405B",
) -> plt.Figure:
    """Horizontal bar chart of mean |SHAP| at the source-variable level."""
    plot_df = table.head(top_n) if top_n else table
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.38 * len(plot_df) + 1.2)))
    labels = [str(v).replace("_", " ") for v in plot_df[label_col]]
    ax.barh(labels[::-1], plot_df["mean_abs_shap"][::-1], color=color, alpha=0.92)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, fontweight="semibold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_shap_signed_bar(
    table: pd.DataFrame,
    *,
    label_col: str,
    title: str,
    top_n: int | None = None,
) -> plt.Figure:
    """Signed mean SHAP — direction of association with predicted comfort."""
    plot_df = table.head(top_n) if top_n else table
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in plot_df["mean_shap"]]
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.38 * len(plot_df) + 1.2)))
    labels = [str(v).replace("_", " ") for v in plot_df[label_col]]
    ax.barh(labels[::-1], plot_df["mean_shap"][::-1], color=colors[::-1], alpha=0.92)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Mean SHAP value")
    ax.set_title(title, fontweight="semibold")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
