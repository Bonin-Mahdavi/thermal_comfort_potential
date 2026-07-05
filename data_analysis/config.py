"""Configuration for RQ1/RQ2 thermal-comfort analysis."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

QUESTIONNAIRE_SHEET_ID = "1NvDIcb-0e1GEwScNdMEFXUeZX_hNLSKyTeo_xyrlreQ"

# ---------------------------------------------------------------------------
# Attribute table (Google Sheet)
# ---------------------------------------------------------------------------

SHEET_ID = "16zxu6Oz2DrUN01AWmOPVsZ7c-1Ija7mv"
SHEET_TAB = "Attribute Data"
GROUP_HEADER_ROW = 0  # 0-based row index of merged layer labels
HEADER_ROW = 1  # 0-based row index of column names

# ---------------------------------------------------------------------------
# Column groups (exact names as in the Google Sheet, row 2)
# ---------------------------------------------------------------------------

POINT_IDENTIFIERS = [
    "Point ID",
    "LCZ Class",
    "LATITUDE",
    "LONGITUDE",
]

# Sheet metadata — not streetscape predictors; excluded before screening and modelling.
ATTRIBUTE_METADATA_COLUMNS = frozenset({
    "Point ID",
    "LCZ Class",
    "LATITUDE",
    "LONGITUDE",
})

LAYER_1_RS_GIS = [
    "LST _mean(°C)",
    "NDVI_mean",
    "Albedo",
    "MNDWI_mean",
]

LAYER_3A_SVI_PIXEL_PCT = [
    "Road (%)",
    "Sidewalk (%)",
    "Building (%)",
    "Wall (%)",
    "Fence (%)",
    "Pole (%)",
    "Traffic light (%)",
    "Traffic sign (%)",
    "Vegetation (%)",
    "Terrain (%)",
    "Sky (%)",
    "Person (%)",
    "Rider (%)",
    "Car (%)",
    "Truck (%)",
    "Bus (%)",
    "Train (%)",
    "Motorcycle (%)",
    "Bicycle (%)",
]

# Not present in the current attribute sheet; kept for optional legacy columns.
LAYER_2_LCZ_MORPHOLOGY: list[str] = []

LAYER_3B_CONTEXTUAL_ATTRIBUTES: list[str] = []

MEAN_Q13_COLUMN = "Mean Q13 rating"

# Collinearity exploratory screening: μ ± k·σ on Σ|ρ| / Σρ distributions.
SCREENING_STD_MULTIPLIER = 1.0

# Variation screening — constant and near-absent rules (Zuur et al., 2010; Zhu et al., 2026).
ENTROPY_MIN_UNIQUE_VALUES = 2
ENTROPY_MIN_NONZERO_POINTS = 3
# Per-city attribute points (n = 10): stricter near-absent rule (≥ 3 non-zero points).
ENTROPY_MIN_NONZERO_POINTS_CITY = 3
# Normalized Shannon entropy floor for retained attributes (H_norm ∈ [0, 1]).
ATTRIBUTE_MIN_H_NORM = 0.50

# Questionnaire variation screening (Zuur et al., 2010; Field, 2018).
QUESTIONNAIRE_MIN_UNIQUE_VALUES = 2
# Modal category share at or above this value flags near-constant factors (Field, 2018).
QUESTIONNAIRE_MAX_DOMINANT_CATEGORY_SHARE = 0.95
# Normalized Shannon entropy floor for retained questionnaire fields.
QUESTIONNAIRE_MIN_H_NORM = 0.40

# Ambiguous occupation labels in the raw sheet → canonical English (applied in load_cleaned_data).
QUESTIONNAIRE_OCCUPATION_LABEL_ALIASES: dict[str, str] = {
    "Employed": "Employed full-time",
}

# Questionnaire collinearity (§2.2): stricter flag than attributes — see Preprocessing §2.
QUESTIONNAIRE_COLLINEARITY_FLAGGED_ABS_RHO_THRESHOLD = 0.8
# Substantive drops after per-city |ρ| review when pairs exceed the §2.2 threshold.
QUESTIONNAIRE_COLLINEARITY_MANUAL_DROPS: dict[str, list[str]] = {
    "Detmold": [],
    "Turin": [],
}

# Pairwise Spearman screening: |ρ| above this value flags redundant pairs (Dormann et al., 2013).
COLLINEARITY_PAIRWISE_ABS_RHO_THRESHOLD = 0.5
# Strong-correlation flag for streetscape attributes in §2.1 pair tables (|ρ| > threshold).
COLLINEARITY_PAIRWISE_FLAGGED_ABS_RHO_THRESHOLD = 0.7

# Substantive drops after iterative |ρ| > 0.7 review (Preprocessing §2.1).
COLLINEARITY_PAIRWISE_MANUAL_DROPS: list[str] = [
    "NDVI",
    "Building (%)",
    "Fence (%)",
    "Traffic sign (%)",
    "Albedo",
]

# Display order for flagged-pair tables (Albedo first, then Vegetation, then others).
COLLINEARITY_HUB_PRIORITY: list[str] = [
    "Albedo",
    "Vegetation (%)",
]

# PCA input: Layer 1–3a numeric attributes + Layer 3b contextual (one-hot).
PCA_NUMERIC_COLUMNS = [
    *LAYER_1_RS_GIS,
    *LAYER_2_LCZ_MORPHOLOGY,
    *LAYER_3A_SVI_PIXEL_PCT,
]

PCA_CATEGORICAL_COLUMNS = LAYER_3B_CONTEXTUAL_ATTRIBUTES

# SVI shares are stored as proportions in the sheet; morphology fractions are 0–100.
SVI_PROPORTION_COLUMNS = LAYER_3A_SVI_PIXEL_PCT

COLUMN_GROUPS: dict[str, list[str]] = {
    "Point Identifiers": POINT_IDENTIFIERS,
    "Layer 1 — RS / GIS": LAYER_1_RS_GIS,
    "Layer 3a — SVI Pixel %": LAYER_3A_SVI_PIXEL_PCT,
}

ALL_COLUMNS: list[str] = [
    *POINT_IDENTIFIERS,
    *LAYER_1_RS_GIS,
    *LAYER_3A_SVI_PIXEL_PCT,
]

# ---------------------------------------------------------------------------
# Cities and survey points (10 per city: 1–5 LCZ-2, 6–10 LCZ-5)
# ---------------------------------------------------------------------------

CITIES = ("Detmold", "Turin")
CITY_PREFIX = {"Detmold": "D", "Turin": "T"}
CITY_COLORS = {"Detmold": "#004B87", "Turin": "#F17A00"}
LCZ_COLORS = {"LCZ-2": "#2E86AB", "LCZ-5": "#A23B72"}
LCZ2_POINT_NUMBERS = (1, 2, 3, 4, 5)
LCZ5_POINT_NUMBERS = (6, 7, 8, 9, 10)


def point_id(city: str, number: int) -> str:
    """Return sheet Point ID, e.g. Detmold point 3 → ``D-3``."""
    return f"{CITY_PREFIX[city]}-{number}"


def city_point_ids(city: str) -> list[str]:
    """All ten point IDs for a city in survey order (1 … 10)."""
    return [point_id(city, n) for n in range(1, 11)]


# ---------------------------------------------------------------------------
# Questionnaire cleaning (Google Sheets)
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "gender",
    "age",
    "birthplace",
    "education",
    "occupation",
    "daily_activity",
    "city_relationship",
    "time_in_city",
    "summer_visit",
    "summer_description",
    "heat_adaptation",
    "modify_routes_heat",
    "transport_walking",
    "transport_cycling",
    "transport_public",
    "transport_car",
    "point_1",
    "point_2",
    "point_3",
    "point_4",
    "point_5",
    "point_6",
    "point_7",
    "point_8",
    "point_9",
    "point_10",
]

SHEET_COLUMN_MAPS: dict[str, dict[str, str]] = {
    "Detmold-English": {
        "Timestamp": "timestamp",
        "Consent question ": "consent",
        "Gender": "gender",
        "Age  ": "age",
        " Where were you born?  ": "birthplace",
        "Highest level of education completed  ": "education",
        "Current occupation  ": "occupation",
        "Your daily activity mainly take place": "daily_activity",
        "Relationship to Detmold  ": "city_relationship",
        "How long have you lived or regularly visited Detmold?": "time_in_city",
        "Have you ever visited or spent time in Detmold during summer?": "summer_visit",
        "How would you describe the summers in Detmold?  ": "summer_description",
        "How well adapted do you feel to summer heat in Detmold?  ": "heat_adaptation",
        "In summer, do you modify your outdoor routes to avoid heat?  ": "modify_routes_heat",
        "In summer, how often do you use the following modes of transport outdoors in Detmold?   [Walking]": "transport_walking",
        "In summer, how often do you use the following modes of transport outdoors in Detmold?   [Cycling]": "transport_cycling",
        "In summer, how often do you use the following modes of transport outdoors in Detmold?   [Public transport]": "transport_public",
        "In summer, how often do you use the following modes of transport outdoors in Detmold?   [Private car]": "transport_car",
        "Point 1.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_1",
        "Point 2. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?   ": "point_2",
        "Point 3. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_3",
        "Point 4.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_4",
        "Point 5.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_5",
        "Point 6.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_6",
        "Point 7.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_7",
        "Point 8.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_8",
        "Point 9. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_9",
        "Point 10. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?   ": "point_10",
        "If you have any related comments, please write in this section.": "comments",
    },
    "Detmold-German": {
        "Timestamp": "timestamp",
        "Einverständnis": "consent",
        "Geschlecht": "gender",
        "Alter": "age",
        "Wo wurden Sie geboren?": "birthplace",
        "Höchster Bildungsabschluss": "education",
        "Derzeitige Tätigkeit": "occupation",
        "Ihre tägliche Hauptaktivität findet überwiegend statt": "daily_activity",
        "Beziehung zu Detmold": "city_relationship",
        "Wie lange leben Sie bereits in Detmold oder besuchen die Stadt regelmäßig?": "time_in_city",
        "Haben Sie im Sommer schon einmal Zeit in Detmold verbracht?": "summer_visit",
        "Wie würden Sie die Sommer in Detmold beschreiben?": "summer_description",
        "Wie gut fühlen Sie sich an sommerliche Hitze in Detmold angepasst?": "heat_adaptation",
        "Ändern Sie im Sommer Ihre Wege im Freien, um Hitze zu vermeiden?": "modify_routes_heat",
        "Wie häufig nutzen Sie im Sommer in Detmold folgende Verkehrsmittel? [Zu Fuß]": "transport_walking",
        "Wie häufig nutzen Sie im Sommer in Detmold folgende Verkehrsmittel? [Fahrrad]": "transport_cycling",
        "Wie häufig nutzen Sie im Sommer in Detmold folgende Verkehrsmittel? [Öffentliche Verkehrsmittel]": "transport_public",
        "Wie häufig nutzen Sie im Sommer in Detmold folgende Verkehrsmittel? [Privates Auto]": "transport_car",
        "Punkt 1. Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_1",
        "Punkt 2.  Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_2",
        "Point 3.  Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_3",
        "Punkt 4. Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_4",
        "Punkt 5.  Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_5",
        "Punkt 6. Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_6",
        "Punkt 7.  Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_7",
        "Punkt 8.  Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_8",
        "Punkt 9.  Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_9",
        "Punkt 10.  Stellen Sie sich vor, Sie gehen an einem heißen Sommertag durch die auf dem Bild gezeigte Straße. Wie würden Sie Ihr thermisches Befinden dabei voraussichtlich bewerten?": "point_10",
        "Wenn Sie weitere Anmerkungen haben, schreiben Sie diese bitte hier.": "comments",
    },
    "Turin-English": {
        "Timestamp": "timestamp",
        "Consent question ": "consent",
        "Gender": "gender",
        "Age  ": "age",
        " Where were you born?  ": "birthplace",
        "Highest level of education completed  ": "education",
        "Current occupation  ": "occupation",
        "Your daily activity mainly take place": "daily_activity",
        "Relationship to Turin  ": "city_relationship",
        "How long have you lived or regularly visited Turin?": "time_in_city",
        "Have you ever visited or spent time in Turin during summer?": "summer_visit",
        "How would you describe the summers in Turin?  ": "summer_description",
        "How well adapted do you feel to summer heat in Turin?  ": "heat_adaptation",
        "In summer, do you modify your outdoor routes to avoid heat?  ": "modify_routes_heat",
        "In summer, how often do you use the following modes of transport outdoors in Turin?   [Walking]": "transport_walking",
        "In summer, how often do you use the following modes of transport outdoors in Turin?   [Cycling]": "transport_cycling",
        "In summer, how often do you use the following modes of transport outdoors in Turin?   [Public transport]": "transport_public",
        "In summer, how often do you use the following modes of transport outdoors in Turin?   [Private car]": "transport_car",
        "Point 1. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_1",
        "Point 2.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?   ": "point_2",
        "Point 3.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_3",
        "Point 4.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_4",
        "Point 5. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_5",
        "Point 6.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_6",
        "Point 7.  Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_7",
        "Point 8. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_8",
        "Point 9. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_9",
        "Point 10. Imagine walking through the street shown in this image on a hot summer day. How thermally comfortable or uncomfortable would you expect to feel?": "point_10",
        "If you have any related comments, please write in this section.": "comments",
    },
    "Turin-Italian": {
        "Timestamp": "timestamp",
        "Domanda di Conseso": "consent",
        "Genere": "gender",
        "Età": "age",
        "Dove è nato/a?": "birthplace",
        "Titolo di studio più elevato conseguito": "education",
        "Occupazione attuale": "occupation",
        "Le Sue attività quotidiane si svolgono prevalentemente": "daily_activity",
        "Rapporto con la città di Torino": "city_relationship",
        "Da quanto tempo abita o frequenta regolarmente Torino?": "time_in_city",
        "Ha mai visitato o soggiornato a Torino durante la stagione estiva?": "summer_visit",
        "Come descriverebbe le estati a Torino?": "summer_description",
        "Quanto si ritiene adattato/a al caldo estivo di Torino?": "heat_adaptation",
        "Durante l'estate, modifica i propri percorsi all'aperto per evitare il caldo?": "modify_routes_heat",
        "In estate, con quale frequenza utilizza i seguenti modalità di trasporto a Torino? [A piedi]": "transport_walking",
        "In estate, con quale frequenza utilizza i seguenti modalità di trasporto a Torino? [Bicicletta]": "transport_cycling",
        "In estate, con quale frequenza utilizza i seguenti modalità di trasporto a Torino? [Mezzi pubblici]": "transport_public",
        "In estate, con quale frequenza utilizza i seguenti modalità di trasporto a Torino? [Auto privata]": "transport_car",
        "Punto 1. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_1",
        "Punto 2. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_2",
        "Punto 3. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_3",
        "Punto 4. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_4",
        "Punto 5. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_5",
        "Punto 6. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_6",
        "Punto 7. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_7",
        "Punto 8. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_8",
        "Punto 9. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_9",
        "Punto 10. Immagini di percorrere la strada mostrata in questa immagine in una giornata estiva calda. Quanto si aspetterebbe di sentirsi termicamente confortevole o scomodo/a?": "point_10",
        "Se ha commenti in merito, la preghiamo di scriverli in questa sezione.": "comments",
    },
}


SHEET_METADATA: dict[str, dict[str, str]] = {
    "Detmold-English": {"questionnaire_city": "Detmold", "language": "en"},
    "Detmold-German": {"questionnaire_city": "Detmold", "language": "de"},
    "Turin-English": {"questionnaire_city": "Turin", "language": "en"},
    "Turin-Italian": {"questionnaire_city": "Turin", "language": "it"},
}

MERGED_COLUMNS = ["questionnaire_city", "language", *CANONICAL_COLUMNS]

COMFORT_SCALE_MAP = {
    "1 — Sehr unangenehm": "1 — Very uncomfortable",
    "2 — Unangenehm": "2 — Uncomfortable",
    "3 — Neutral": "3 — Neutral",
    "4 — Angenehm": "4 — Comfortable",
    "5 — Sehr angenehm": "5 — Very comfortable",
    "1 — Molto Scomodo/a": "1 — Very uncomfortable",
    "2 — Scomodo/a": "2 — Uncomfortable",
    "3 — Neutrale": "3 — Neutral",
    "4 — Confortevole": "4 — Comfortable",
    "5 — Molto Confortevole": "5 — Very comfortable",
}

FREQUENCY_MAP = {
    "Immer": "Always",
    "Manchmal": "Sometimes",
    "Nie": "Never",
    "Täglich": "Daily",
    "Sempre": "Always",
    "A volte": "Sometimes",
    "Mai": "Never",
    "Quotidianamente": "Daily",
}

# birthplace → in_city | in_region | in_country | other_place
BIRTHPLACE_MAP: dict[str, str] = {
    # in_city — born in the survey city
    "In Detmold": "in_city",
    "In Turin": "in_city",
    "A Torino": "in_city",
    # in_region — born elsewhere in the survey region (NRW / Piedmont)
    "Elsewhere in North Rhine-Westphalia": "in_region",
    "Anderswo in Nordrhein-Westfalen": "in_region",
    "Elsewhere in Piedmonte": "in_region",
    "Altrove in Piemonte": "in_region",
    # in_country — born elsewhere in the same country
    "Elsewhere in Germany": "in_country",
    "Elsewhere in Italy": "in_country",
    "Anderswo in Deutschland": "in_country",
    "Altrove in Italia": "in_country",
    # other_place — born abroad (countries, foreign cities, open-text abroad)
    "Abudhabi": "other_place",
    "Afghanistan": "other_place",
    "Ardabil, Iran": "other_place",
    "Australia": "other_place",
    "Baku": "other_place",
    "Baku, Azerbaijan": "other_place",
    "Bangladesh": "other_place",
    "Bogotá-Colombia": "other_place",
    "brasilien": "other_place",
    "Bulgaria": "other_place",
    "Canada": "other_place",
    "China": "other_place",
    "Colombia": "other_place",
    "Danzig, Polen": "other_place",
    "Greece": "other_place",
    "In Finland": "other_place",
    "Indonesia": "other_place",
    "Iran": "other_place",
    "IRAN": "other_place",
    "iran": "other_place",
    "Iran, Tehran": "other_place",
    "Iran,Tehran": "other_place",
    "Kyiv, Ukraine": "other_place",
    "Nairobi": "other_place",
    "Outside italy": "other_place",
    "Pakistan": "other_place",
    "Perú": "other_place",
    "Polen": "other_place",
    "Romania": "other_place",
    "Russia": "other_place",
    "Russland": "other_place",
    "Sari": "other_place",
    "Slovakia": "other_place",
    "Syria": "other_place",
    "Syrien": "other_place",
    "Syrisch": "other_place",
    "Tehran": "other_place",
    "Tehran, Iran": "other_place",
    "Tehran-Iran": "other_place",
    "The Netherlands": "other_place",
    "in den Niederlanden": "other_place",
    "Turkey": "other_place",
    "Turkiye": "other_place",
    "USA": "other_place",
    "United Kingdom (London)": "other_place",
    "norway": "other_place",
}

VALUE_MAPS: dict[str, dict[str, str]] = {
    "gender": {
        "Frau": "Female",
        "Mann": "Male",
        "Nicht-Binär": "Non-binary",
        "Donna": "Female",
        "Uomo": "Male",
    },
    "age": {
        "Più di 65": "Over 65",
    },
    "education": {
        "Bachelor oder gleichwertig": "Bachelor's degree or equivalent",
        "Master oder gleichwertig": "Master's degree or equivalent",
        "Kein höherer Abschluss als Schulabschluss": "Not higher than high school",
        "Berufsausbildung oder gleichwertig (z. B. Ausbildung)": "Vocational training or equivalent",
        "Abitur": "Not higher than high school",
        "Abitur, Berufsausbildung": "Vocational training or equivalent",
        "Laurea triennale o equivalente": "Bachelor's degree or equivalent",
        "Laurea magistrale o equivalente": "Master's degree or equivalent",
        "Dottorato di ricerca (PhD) o superiore": "Doctoral degree (PhD) or more",
        "Diploma di scuola superiore o inferiore": "Not higher than high school",
        "Qualifica professionale o equivalente": "Vocational training or equivalent",
        "Scuola specializzazione e master II livello": "Master's degree or equivalent",
        "Master post laurea": "Master's degree or equivalent",
    },
    "occupation": {
        "Im Ruhestand": "Retired",
        "Student/in": "Student",
        "Teilzeit beschäftigt": "Employed part-time",
        "Vollzeit beschäftigt": "Employed full-time",
        "Dipendente a tempo parziale (part-time)": "Employed part-time",
        "Dipendente a tempo pieno (full-time)": "Employed full-time",
        "Disoccupato/a": "Unemployed",
        "Libero professionista / Lavoratore autonomo": "Self-employed",
        "Pensionato/a": "Retired",
        "Studente / Studentessa": "Student",
    },
    "daily_activity": {
        "Gemischt (innen und außen)": "Mixed (both indoors and outdoors)",
        "Im Freien (z. B. Bau, Landwirtschaft, Feldarbeit)": "Outdoors (construction, agriculture, fieldwork, etc.)",
        "Innenräumen (Büro, Klassenraum, Zuhause)": "Indoors (office, classroom, home)",
        "Al chiuso (ufficio, aula, casa)": "Indoors (office, classroom, home)",
        "In modo misto (sia al chiuso che all'aperto)": "Mixed (both indoors and outdoors)",
    },
    "city_relationship": {
        "Ich wohne in Detmold": "I live in the city",
        "Ich pendle regelmäßig nach Detmold": "I commute to the city regularly",
        "Abito a Torino": "I live in the city",
        "I live in Detmold": "I live in the city",
        "I live in Turin": "I live in the city",
        "I commute to Detmold regularly": "I commute to the city regularly",
        "I commute to Turin regularly": "I commute to the city regularly",
        "Faccio il pendolare verso Torino regolarmente": "I commute to the city regularly",
    },
    "time_in_city": {
        "Weniger als 1 Jahr": "Less than 1 year",
        "1–3 Jahre": "1–3 years",
        "4–10 Jahre": "4–10 years",
        "Mehr als 10 Jahre": "More than 10 years",
        "Seit Geburt": "Since birth",
        "Meno di 1 anno": "Less than 1 year",
        "1–3 anni": "1–3 years",
        "4–10 anni": "4–10 years",
        "Più di 10 anni": "More than 10 years",
        "Dalla nascita": "Since birth",
    },
    "summer_visit": {
        "Ja": "Yes",
        "Nein": "No",
        "Sì": "Yes",
    },
    "summer_description": {
        "Heiß": "Hot",
        "Kühl": "Mild",
        "Calde": "Warm",
        "Molto calde": "Very hot",
        "Troppo calde": "Very hot",
    },
    "heat_adaptation": {
        "Gar nicht angepasst": "Not at all adapted",
        "Gut angepasst": "Well adapted",
        "Leicht angepasst": "Slightly adapted",
        "Mäßig angepasst": "Moderately adapted",
        "Per nulla adattato/a": "Not at all adapted",
        "Ben adattato/a": "Well adapted",
        "Poco adattato/a": "Slightly adapted",
        "Moderatamente adattato/a": "Moderately adapted",
    },
    "modify_routes_heat": FREQUENCY_MAP,
    "transport_walking": FREQUENCY_MAP,
    "transport_cycling": FREQUENCY_MAP,
    "transport_public": FREQUENCY_MAP,
    "transport_car": FREQUENCY_MAP,
}

# English summer-climate labels; occasionally entered in summer_visit by mistake.
SUMMER_DESCRIPTION_EN_LABELS = frozenset({"Hot", "Mild", "Warm", "Very hot"})
# Assume yes when a respondent describes summers but left summer_visit blank/wrong field.
SUMMER_VISIT_WHEN_DESCRIPTION_MISPLACED = "Yes"

POINT_COLUMNS = [f"point_{i}" for i in range(1, 11)]

# Shared keyword maps (English label → keyword)
FREQUENCY_KEYWORDS = {
    "Never": "never",
    "Sometimes": "sometimes",
    "Always": "always",
    "Daily": "daily",
    "1-3 times/month": "monthly",
    "1-3 times/week": "weekly",
}

COMFORT_KEYWORDS = {
    "1 — Very uncomfortable": 1,
    "2 — Uncomfortable": 2,
    "3 — Neutral": 3,
    "4 — Comfortable": 4,
    "5 — Very comfortable": 5,
}

# Per-column English label → keyword (birthplace handled by BIRTHPLACE_MAP)
KEYWORD_MAPS: dict[str, dict[str, str]] = {
    "gender": {
        "Female": "f",
        "Male": "m",
        "Non-binary": "nb",
    },
    "age": {
        "18–25": "18_25",
        "26–35": "26_35",
        "36–45": "36_45",
        "46–55": "46_55",
        "56–65": "56_65",
        "Over 65": "65_plus",
    },
    "education": {
        "Not higher than high school": "high_school",
        "Vocational training or equivalent": "vocational",
        "Bachelor's degree or equivalent": "bachelors",
        "Master's degree or equivalent": "masters",
        "Doctoral degree (PhD) or more": "phd",
    },
    "occupation": {
        "Student": "student",
        "Employed full-time": "full_time",
        "Employed part-time": "part_time",
        "Self-employed": "self_employed",
        "Unemployed": "unemployed",
        "Retired": "retired",
    },
    "daily_activity": {
        "Indoors (office, classroom, home)": "indoors",
        "Mixed (both indoors and outdoors)": "mix",
        "Outdoors (construction, agriculture, fieldwork, etc.)": "outdoors",
    },
    "city_relationship": {
        "I live in the city": "resident",
        "I commute to the city regularly": "commuter",
    },
    "time_in_city": {
        "Less than 1 year": "lt_1y",
        "1–3 years": "y_1_3",
        "4–10 years": "y_4_10",
        "More than 10 years": "gt_10y",
        "Since birth": "since_birth",
    },
    "summer_visit": {
        "Yes": "y",
        "No": "n",
    },
    "summer_description": {
        "Mild": "mild",
        "Warm": "warm",
        "Hot": "hot",
        "Very hot": "very_hot",
    },
    "heat_adaptation": {
        "Not at all adapted": "none",
        "Slightly adapted": "slight",
        "Moderately adapted": "moderate",
        "Well adapted": "well",
    },
    "modify_routes_heat": FREQUENCY_KEYWORDS,
    "transport_walking": FREQUENCY_KEYWORDS,
    "transport_cycling": FREQUENCY_KEYWORDS,
    "transport_public": FREQUENCY_KEYWORDS,
    "transport_car": FREQUENCY_KEYWORDS,
}

# Raw numeric age entries → keyword (survey otherwise uses age bands)
AGE_NUMERIC_KEYWORDS: dict[int, str] = {
    31: "26_35",
}


# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------

RATING_ORDER = [1, 2, 3, 4, 5]
KDE_FILL_ALPHA = 0.22
KDE_BANDWIDTH = 0.8
BOXPLOT_WIDTH = 0.42

ATTRIBUTE_LAYERS_MAIN = {
    "Layer 1 — RS / GIS": LAYER_1_RS_GIS,
    "Layer 3a — SVI Pixel %": LAYER_3A_SVI_PIXEL_PCT,
}

# Fallback when layer group headers cannot be read from the sheet.
ATTRIBUTE_LAYERS_ALL = dict(ATTRIBUTE_LAYERS_MAIN)

# Side-band colours for layer-grouped attribute figures.
ATTRIBUTE_LAYER_BAND_COLORS = {
    "Layer 1 — RS / GIS": "#3D5A80",
    "Layer 3a — SVI Pixel %": "#A23B72",
}

NON_PREDICTOR_ATTRIBUTE_COLUMNS = frozenset({
    *ATTRIBUTE_METADATA_COLUMNS,
    MEAN_Q13_COLUMN,
})

SECTION_2_COLUMNS = [
    "gender",
    "age",
    "birthplace",
    "education",
    "occupation",
    "daily_activity",
    "city_relationship",
    "time_in_city",
]

SECTION_3_COLUMNS = [
    "summer_description",
    "heat_adaptation",
    "modify_routes_heat",
    "transport_walking",
    "transport_cycling",
    "transport_public",
    "transport_car",
]

# Questionnaire SHAP figure — section group labels (left axis) and member variables
QUESTIONNAIRE_SHAP_GROUPS: dict[str, list[str]] = {
    "Demographics": list(SECTION_2_COLUMNS),
    "Summer & adaptation": [
        "summer_visit",
        "summer_description",
        "heat_adaptation",
        "modify_routes_heat",
    ],
    "Travel behaviour": [
        "transport_walking",
        "transport_cycling",
        "transport_public",
        "transport_car",
    ],
}

SHAP_TABLE_GROUP_ORDER = [
    "RS / GIS",
    "Streetscape",
    *QUESTIONNAIRE_SHAP_GROUPS.keys(),
]

ATTRIBUTE_SHAP_GROUP_ORDER = ["RS / GIS", "Streetscape"]

QUESTIONNAIRE_SHAP_GROUP_ORDER = list(QUESTIONNAIRE_SHAP_GROUPS.keys())

QUESTIONNAIRE_VARIABLE_LABELS: dict[str, str] = {
    "gender": "Gender",
    "age": "Age",
    "birthplace": "Birthplace",
    "education": "Education",
    "occupation": "Occupation",
    "daily_activity": "Daily activity",
    "city_relationship": "City relationship",
    "time_in_city": "Time in city",
    "summer_visit": "Summer visit",
    "summer_description": "Summer description",
    "heat_adaptation": "Heat adaptation",
    "modify_routes_heat": "Modify routes for heat",
    "transport_walking": "Transport (walking)",
    "transport_cycling": "Transport (cycling)",
    "transport_public": "Transport (public)",
    "transport_car": "Transport (car)",
}

BIRTHPLACE_KEYWORD_LABELS: dict[str, str] = {
    "in_city": "In survey city",
    "in_region": "Elsewhere in region",
    "in_country": "Elsewhere in country",
    "other_place": "Abroad / other",
}

SHAP_GROUP_COLORS = {
    "RS / GIS": ATTRIBUTE_LAYER_BAND_COLORS["Layer 1 — RS / GIS"],
    "Streetscape": ATTRIBUTE_LAYER_BAND_COLORS["Layer 3a — SVI Pixel %"],
    "Demographics": "#6B705C",
    "Summer & adaptation": "#BC6C25",
    "Travel behaviour": "#7D6B5D",
}

ATTRIBUTE_SHAP_GROUP_COLORS = {
    group: SHAP_GROUP_COLORS[group] for group in ATTRIBUTE_SHAP_GROUP_ORDER
}

QUESTIONNAIRE_SHAP_GROUP_COLORS = {
    group: SHAP_GROUP_COLORS[group] for group in QUESTIONNAIRE_SHAP_GROUP_ORDER
}

# Nested figure label: parent spans these questionnaire subsection groups.
SHAP_PARENT_GROUPS: dict[str, list[str]] = {
    "Respondent Profile": list(QUESTIONNAIRE_SHAP_GROUPS.keys()),
}

# Top-level figure labels — aligned on one vertical column.
ATTRIBUTE_SHAP_TOP_LEVEL_GROUPS: dict[str, list[str]] = {
    "RS / GIS": ["RS / GIS"],
    "Streetscape": ["Streetscape"],
}

SHAP_TOP_LEVEL_GROUPS: dict[str, list[str]] = {
    **ATTRIBUTE_SHAP_TOP_LEVEL_GROUPS,
    **SHAP_PARENT_GROUPS,
}

RQ5_CATEGORY_COLUMNS = [
    *SECTION_2_COLUMNS,
    "summer_visit",
    *SECTION_3_COLUMNS,
]

COVARIATE_CATEGORY_ORDERS = {
    "age": ("18–25", "26–35", "36–45", "Over 46"),
    "education": (
        "High school / vocational",
        "Bachelor's",
        "Master's / PhD",
        "unknown",
    ),
    "heat_adaptation": ("none", "slight", "moderate", "well"),
    "summer_description": ("mild", "warm", "hot", "very_hot"),
    "time_in_city": ("< 3 years", "4–10 years", "10+ years / since birth"),
    "modify_routes_heat": ("never", "monthly", "weekly", "sometimes", "daily", "always"),
    "transport_walking": ("never", "monthly", "weekly", "sometimes", "daily", "always"),
    "transport_cycling": ("never", "monthly", "weekly", "sometimes", "daily", "always"),
    "transport_public": ("never", "monthly", "weekly", "sometimes", "daily", "always"),
    "transport_car": ("never", "monthly", "weekly", "sometimes", "daily", "always"),
    "daily_activity": ("Indoors", "Mix / outdoors"),
    "gender": ("f", "m"),
    "birthplace": ("In city / region", "In country", "Other place"),
    "occupation": ("Student", "Employed", "Not employed"),
    "city_relationship": ("resident", "commuter", "unknown"),
    "summer_visit": ("y", "n", "unknown"),
}

# RQ1 profile analysis: a priori category collapse (survey review, survey-results.docx).
QUESTIONNAIRE_PROFILE_CATEGORY_GROUPS: dict[str, list[list[str]]] = {
    "age": [
        ["18_25"],
        ["26_35"],
        ["36_45"],
        ["46_55", "56_65", "65_plus"],
    ],
    "birthplace": [
        ["in_city", "in_region"],
        ["in_country"],
        ["other_place"],
    ],
    "education": [
        ["high_school", "vocational"],
        ["bachelors"],
        ["masters", "phd"],
    ],
    "occupation": [
        ["student"],
        ["full_time", "part_time", "self_employed"],
        ["retired", "unemployed"],
    ],
    "daily_activity": [
        ["indoors"],
        ["mix", "outdoors"],
    ],
    "time_in_city": [
        ["lt_1y", "y_1_3"],
        ["y_4_10"],
        ["gt_10y", "since_birth"],
    ],
}

QUESTIONNAIRE_PROFILE_GROUP_LABELS: dict[str, list[str]] = {
    "age": ["18–25", "26–35", "36–45", "Over 46"],
    "birthplace": ["In city / region", "In country", "Other place"],
    "education": ["High school / vocational", "Bachelor's", "Master's / PhD"],
    "occupation": ["Student", "Employed", "Not employed"],
    "daily_activity": ["Indoors", "Mix / outdoors"],
    "time_in_city": ["< 3 years", "4–10 years", "10+ years / since birth"],
}

# RQ1: exclude non-binary gender from respondent-profile sections.
RQ1_EXCLUDED_GENDER_CATEGORIES = ("nb",)

# RQ1 category-comparison step: interpret |diff from city mean| on the 1–5 scale.
RQ1_CATEGORY_DIFF_NEGLIGIBLE = 0.10   # |diff| below → not substantively different
RQ1_CATEGORY_DIFF_SUBSTANTIVE = 0.25  # |diff| at or above → pronounced separation

# Legacy (unused after category-comparison refactor).
RQ1_CATEGORY_INFLUENCE_MARGIN = 0.2

# ML preparation
ML_EXCLUDED_FEATURE_COLUMNS = frozenset({
    "respondent_id",
    "point_id",
    "language__de",
    "language__en",
    "language__it",
    "questionnaire_city__Detmold",
    "questionnaire_city__Turin",
    *ATTRIBUTE_METADATA_COLUMNS,
})

QUESTIONNAIRE_CATEGORICAL_COLUMNS = [
    "gender",
    "age",
    "birthplace",
    "education",
    "occupation",
    "daily_activity",
    "city_relationship",
    "time_in_city",
    "summer_visit",
    "summer_description",
    "heat_adaptation",
    "modify_routes_heat",
    "transport_walking",
    "transport_cycling",
    "transport_public",
    "transport_car",
]

ATTRIBUTE_CATEGORICAL_COLUMNS = list(LAYER_3B_CONTEXTUAL_ATTRIBUTES)
ATTRIBUTE_EXCLUDE_COLUMNS = [*ATTRIBUTE_METADATA_COLUMNS, MEAN_Q13_COLUMN]
RATING_COLUMNS = list(POINT_COLUMNS)

# RQ2 — model comparison (Yang et al., 2025 VATA Table 3 + OLS baseline)
RQ2_TUNED_MODELS = (
    "decision_tree",
    "knn",
    "svr",
    "elastic_net",
    "lightgbm",
    "random_forest",
    "xgboost",
    "catboost",
    "ols",
)
RQ2_TUNING_RANDOM_STATE = 42
RQ2_TUNING_N_ITER = 50
# Respondent-level held-out fraction per city (75% train / 25% test).
RQ2_TEST_SIZE = 0.25
# When True, models use RQ2_HYPERPARAMETERS (no random search). Set False to re-tune.
RQ2_USE_FIXED_HYPERPARAMETERS = True
# Selected via random search (n_iter=50, lowest train MAE); seed 42. Re-tune if RQ2_TEST_SIZE changes.
RQ2_HYPERPARAMETERS: dict[str, dict[str, dict[str, object]]] = {
    "Detmold": {
        "decision_tree": {
            "max_depth": 20,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
        },
        "knn": {"n_neighbors": 3, "p": 1, "weights": "distance"},
        "svr": {"C": 10.0, "epsilon": 0.01, "gamma": 0.1},
        "elastic_net": {"alpha": 0.00018329807108324357, "l1_ratio": 1.0},
        "lightgbm": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": -1,
            "num_leaves": 127,
            "min_child_samples": 10,
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 16,
            "min_samples_leaf": 1,
            "max_features": 0.7,
        },
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.2,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
        "catboost": {
            "iterations": 500,
            "depth": 10,
            "learning_rate": 0.2,
            "l2_leaf_reg": 1,
        },
        "ols": {},
    },
    "Turin": {
        "decision_tree": {
            "max_depth": 20,
            "min_samples_leaf": 2,
            "min_samples_split": 2,
        },
        "knn": {"n_neighbors": 3, "p": 1, "weights": "distance"},
        "svr": {"C": 10.0, "epsilon": 0.01, "gamma": 0.1},
        "elastic_net": {"alpha": 0.00018329807108324357, "l1_ratio": 1.0},
        "lightgbm": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": -1,
            "num_leaves": 127,
            "min_child_samples": 10,
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 16,
            "min_samples_leaf": 1,
            "max_features": 0.7,
        },
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.2,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
        "catboost": {
            "iterations": 500,
            "depth": 10,
            "learning_rate": 0.2,
            "l2_leaf_reg": 1,
        },
        "ols": {},
    },
}
RQ2_MODEL_DISPLAY_NAMES = {
    "decision_tree": "Decision tree",
    "knn": "KNN regression",
    "svr": "Support vector regression",
    "elastic_net": "Elastic net (ENRM)",
    "lightgbm": "LightGBM",
    "random_forest": "Random forest",
    "xgboost": "Extreme gradient boosting (XGBoost)",
    "catboost": "CatBoost",
    "ols": "OLS / linear regression",
}
# Display labels for per-city model performance tables (test-set MAE, MSE, RMSE, adjusted R²).
RQ2_MODEL_REPORT_NAMES = {
    "decision_tree": "Decision tree",
    "knn": "KNN regression",
    "svr": "Support vector regression",
    "elastic_net": "ENRM (Lasso + Ridge)",
    "lightgbm": "LightGBM",
    "random_forest": "Random forest",
    "xgboost": "Extreme gradient boosting",
    "catboost": "CatBoost",
    "ols": "OLS / linear regression",
}
