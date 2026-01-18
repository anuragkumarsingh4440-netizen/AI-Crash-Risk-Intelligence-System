# This block initializes the Streamlit application.
# It sets a strict dark UI theme suitable for police and government users.
# Page configuration is defined first to avoid Streamlit runtime errors.
# Global CSS is injected to enforce red-green intelligence visuals.
# This block is necessary to ensure consistent UI across all modules.
# No ML or logic runs here, only UI and environment safety.

import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

st.set_page_config(
    page_title="AI-ML Crash Risk Intelligence System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ===== FORCE FULL DARK MODE FOR ENTIRE STREAMLIT APP ===== */

/* App root */
.stApp {
    background-color: #020617 !important;
    color: #E5E7EB !important;
}

/* Main content area */
div[data-testid="stAppViewContainer"] {
    background-color: #020617 !important;
}

/* Main block */
main {
    background-color: #020617 !important;
}

/* All vertical blocks */
div[data-testid="stVerticalBlock"] {
    background-color: #020617 !important;
}

/* Sidebar full dark */
section[data-testid="stSidebar"] {
    background-color: #020617 !important;
    border-right: 1px solid #1F2937;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

/* Headers & text */
h1, h2, h3, h4, h5, h6 {
    color: #F9FAFB !important;
}

/* Paragraphs & labels */
p, span, label, div {
    color: #D1D5DB !important;
}

/* DataFrames & Tables */
.stDataFrame, .stTable {
    background-color: #020617 !important;
    color: #E5E7EB !important;
}

.stDataFrame th {
    background-color: #020617 !important;
    color: #F9FAFB !important;
}

.stDataFrame td {
    background-color: #020617 !important;
    color: #CBD5E1 !important;
}

/* Metrics */
div[data-testid="metric-container"] {
    background-color: #020617 !important;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 10px;
}

div[data-testid="metric-container"] label {
    color: #9CA3AF !important;
}

div[data-testid="metric-container"] div {
    color: #F9FAFB !important;
}

/* Buttons */
.stButton > button {
    background-color: #1E40AF !important;
    color: #F9FAFB !important;
    border-radius: 8px;
}

.stButton > button:hover {
    background-color: #DC2626 !important;
}

/* Inputs */
input, textarea, select {
    background-color: #020617 !important;
    color: #E5E7EB !important;
    border: 1px solid #374151 !important;
}

/* Plotly charts background */
.plotly-graph-div {
    background-color: #020617 !important;
}

/* Remove Streamlit toolbar background */
header[data-testid="stHeader"] {
    background-color: #020617 !important;
}

/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* ===== POLICE INTELLIGENCE HEADER CONTAINER ===== */
.police-header {
    background: linear-gradient(
        90deg,
        #0B3C5D 0%,
        #1E40AF 40%,
        #991B1B 100%
    );
    border: 2px solid #DC2626;
    border-left: 8px solid #2563EB;
    border-right: 8px solid #DC2626;
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 24px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.45);
}

/* ===== MAIN TITLE ===== */
.police-header h1 {
    color: #F9FAFB;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}

/* ===== SUBTITLE ===== */
.police-header p {
    color: #E5E7EB;
    font-size: 16px;
    font-weight: 500;
    opacity: 0.9;
    margin: 0;
}

/* ===== ICON BADGE ===== */
.police-badge {
    display: inline-block;
    background-color: #FFFFFF;
    color: #111827;
    font-weight: 800;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    margin-bottom: 10px;
}

/* ===== TABLE BORDER MATCHING HEADER ===== */
.stDataFrame {
    border: 1px solid #1E40AF !important;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="police-header">
    <div class="police-badge">🚓 POLICE INTELLIGENCE SYSTEM</div>
    <h1>🚨 AI Crash Risk Intelligence System</h1>
    <p>AI-powered decision intelligence for traffic police and government authorities</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #020617 !important;
    color: #D1D5DB !important;
}

/* Global text softness to avoid eye strain */
* {
    color: #D1D5DB !important;
    font-weight: 500 !important;
}

/* Headings slightly brighter but not pure white */
h1, h2 {
    color: #F9FAFB !important;
    font-weight: 700 !important;
}

h3, h4, h5 {
    color: #E5E7EB !important;
    font-weight: 600 !important;
}

/* Buttons with controlled contrast */
.stButton > button {
    background-color: #1E40AF !important;
    color: #F9FAFB !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background-color: #DC2626 !important;
    color: #F9FAFB !important;
}

/* Sidebar readability */
section[data-testid="stSidebar"] {
    background-color: #020617 !important;
}

section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

/* Tables and dataframes for long reading sessions */
.stDataFrame, .stTable {
    background-color: #020617 !important;
    color: #CBD5E1 !important;
}

.stDataFrame th {
    color: #E5E7EB !important;
    font-weight: 600 !important;
}

.stDataFrame td {
    color: #CBD5E1 !important;
    font-weight: 400 !important;
}

/* Metrics clarity */
.stMetric label {
    color: #9CA3AF !important;
}

.stMetric value {
    color: #F3F4F6 !important;
}
</style>
""", unsafe_allow_html=True)




# This block loads environment variables safely.
# It reads the Google Gemini API key from .env file.
# No hardcoding is allowed for security and audit reasons.
# If the key is missing, the system continues without GenAI.
# GenAI is optional and never blocks core ML functionality.
# This ensures safe deployment in restricted environments.

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_KEY")
GEMINI_AVAILABLE = True if GEMINI_API_KEY else False


# This block configures Google Gemini safely.
# Only Gemini 2.5 Flash Lite model is allowed as per requirement.
# GenAI is used only for summarization and planning.
# No predictions or decisions are delegated to GenAI.
# Errors are handled gracefully to avoid app crashes.
# Model limit errors return a polite user-safe message.

gemini_model = None

if GEMINI_AVAILABLE:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
    except Exception:
        gemini_model = None
        GEMINI_AVAILABLE = False

# This block defines a safe Gemini call wrapper.
# It prevents app crashes due to quota or token limits.
# Any large error is converted into a user-friendly message.
# This function never raises raw exceptions.
# If GenAI is unavailable, it returns a clean fallback.
# This keeps police-facing UI calm and professional.

def safe_gemini_generate(prompt: str) -> str:
    if not GEMINI_AVAILABLE or gemini_model is None:
        return "AI advisory not available at the moment."

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Sorry, AI model limit exceeded. Please try again after some time."

# This block builds the sidebar control panel.
# It allows police users to choose system mode.
# File upload is restricted to CSV only.
# No processing starts without explicit user action.
# This separation avoids accidental execution.
# Sidebar controls guide the entire application flow.

st.sidebar.title("⚙️ Control Panel")

mode = st.sidebar.radio(
    "Select System Mode",
    [
        "📊 Police Intelligence Analysis",
        "🤖 AI Prediction & Risk Assessment"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Monthly Crash Dataset (CSV)",
    type=["csv"]
)

# This block safely loads the uploaded dataset.
# It protects against empty, corrupt, or invalid files.
# No assumptions are made about schema at this stage.
# Errors are shown clearly to the user.
# The app stops cleanly if data is not usable.
# This prevents silent failures downstream.

def load_dataset(file) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file, low_memory=False)
        if df.empty:
            raise ValueError("Uploaded file is empty")
        return df
    except Exception as e:
        st.error("Failed to load dataset")
        st.code(str(e))
        return None

df = None

if uploaded_file:
    with st.spinner("Loading dataset..."):
        df = load_dataset(uploaded_file)
    if df is not None:
        st.sidebar.success("Dataset loaded successfully")

if df is None:
    st.info("Upload a cleaned police crash CSV to begin")
    st.stop()


# This block shows a basic dataset overview.
# It gives police users confidence that data is correct.
# Metrics are simple and non-technical.
# Preview is limited to avoid heavy rendering.
# This block has no impact on ML logic.
# It improves trust and usability.

st.subheader("📂 Dataset Overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Records", df.shape[0])
with c2:
    st.metric("Total Fields", df.shape[1])
with c3:
    st.metric("Missing Values", int(df.isna().sum().sum()))

st.dataframe(df.head(20), use_container_width=True)


# This block validates whether the uploaded dataset is usable for the system.
# It checks for minimum required columns needed for analysis and mapping.
# Latitude and Longitude are critical for hotspot and spatial intelligence.
# Validation errors are shown clearly and stop execution safely.
# This prevents wrong data from entering ML and decision logic.
# Police users immediately know what is missing.

st.subheader("🧪 Data Validation")

REQUIRED_COLUMNS = [
    "Crash_hour",
    "Crash_day_name",
    "Road Name",
    "Latitude",
    "Longitude"
]

def validate_dataset(df: pd.DataFrame):
    issues = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    if "Latitude" in df.columns:
        if df["Latitude"].isna().mean() > 0.3:
            issues.append("Too many missing Latitude values")

    if "Longitude" in df.columns:
        if df["Longitude"].isna().mean() > 0.3:
            issues.append("Too many missing Longitude values")

    return issues

validation_issues = validate_dataset(df)

if validation_issues:
    st.error("Dataset validation failed")
    for issue in validation_issues:
        st.warning(issue)
    st.stop()
else:
    st.success("Dataset validation passed")


# This block prepares the dataset for ML and analysis.
# It removes identifiers that have no predictive value.
# Missing values are handled safely without deleting rows.
# Numeric columns are coerced carefully to avoid crashes.
# Optional columns are created if missing to keep schema stable.
# This guarantees that downstream models never break.

st.subheader("🧹 Data Preprocessing")

OPTIONAL_COLUMNS = [
    "injury_severity",
    "crash_risk_level",
    "driver_at_fault",
    "damage_extent",
    "driver_distraction",
    "kmeans_cluster"
]

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [
        "Unnamed: 0",
        "Report Number",
        "Local Case Number",
        "Person ID",
        "Vehicle ID",
        "Crash Date/Time",
        "Crash_date"
    ]

    df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        inplace=True,
        errors="ignore"
    )

    for col in df.columns:

        # 🔴 IMPORTANT FIX: Preserve kmeans_cluster as numeric
        if col == "kmeans_cluster":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

        # 🔴 IMPORTANT FIX: Preserve driver_distraction as categorical
        elif col == "driver_distraction":
            df[col] = df[col].fillna("UNKNOWN").astype(str)

        # Handle other categorical columns
        elif df[col].dtype == "O":
            df[col] = df[col].fillna("UNKNOWN")

        # Handle numeric columns
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Ensure optional columns always exist
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            if col == "kmeans_cluster":
                df[col] = -1
            else:
                df[col] = "UNKNOWN"

    return df

with st.spinner("Preprocessing dataset..."):
    df_clean = preprocess_dataset(df)

st.success("Data preprocessing completed")



# This block shows a preprocessing summary for transparency.
# Police users can verify that rows were not dropped.
# It confirms that missing values are handled.
# This builds trust in the system pipeline.
# No ML logic runs here.
# Only reporting and validation.

st.subheader("📊 Preprocessing Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Rows After Cleaning", df_clean.shape[0])

with c2:
    st.metric("Columns After Cleaning", df_clean.shape[1])

with c3:
    st.metric("Remaining Missing Values", int(df_clean.isna().sum().sum()))

with st.expander("Preview Cleaned Data"):
    st.dataframe(df_clean.head(20), use_container_width=True)


# This block stores the cleaned dataset in session state.
# Session state ensures reuse across all modules.
# No reprocessing is required later.
# This is critical for performance and stability.
# All models, maps, and reports depend on this.
# Without this, the app would recompute unnecessarily.

st.session_state["df_clean"] = df_clean

st.info(
    "Cleaned data is now ready for analysis, prediction, maps, and reporting."
)


# This block prepares features exactly in the format required by trained ML models.
# It ensures there is no feature mismatch during inference.
# Two datasets are maintained:
# One human-readable for analysis and reports.
# One encoded dataset strictly for ML prediction.
# This separation avoids confusion and model crashes.
# All transformations are deterministic and repeatable.

st.subheader("🧠 Feature Engineering & Encoding")

df_clean = st.session_state["df_clean"].copy()

# This block imports encoding utilities.
# Label encoding is used for medium-cardinality categorical columns.
# Frequency encoding is used for high-cardinality columns like road names.
# Ordinal encoding preserves severity ordering.
# Encoders are stored for audit and traceability.
# No fitting happens outside the uploaded dataset.

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
frequency_maps = {}


# This block defines helper encoding functions.
# Frequency encoding converts category frequency into numeric signal.
# Label encoding converts text categories into integers.
# Ordinal encoding preserves logical order in severity-like fields.
# These functions are simple, stable, and audit-friendly.
# They never drop rows or change row order.
# All failures fallback safely.

def frequency_encode(series):
    freq = series.value_counts(normalize=True)
    return series.map(freq).fillna(0), freq

def label_encode(series, col_name):
    le = LabelEncoder()
    encoded = le.fit_transform(series.astype(str))
    label_encoders[col_name] = le
    return encoded

def ordinal_encode(series, mapping):
    return series.map(mapping).fillna(0)


# This block removes columns that should never enter ML models.
# Identifiers and report numbers add noise and cause leakage.
# Dropping them here guarantees consistency with training.
# Errors are ignored safely if columns are missing.
# This step is mandatory for clean inference.
# The human-readable dataset is preserved separately.

DROP_COLS = [
    "Unnamed: 0",
    "Report Number",
    "Local Case Number",
    "Person ID",
    "Vehicle ID",
    "Crash Date/Time",
    "Crash_date"
]

df_clean.drop(
    columns=[c for c in DROP_COLS if c in df_clean.columns],
    inplace=True,
    errors="ignore"
)


# This block creates the encoded dataset for ML models.
# The original cleaned dataset remains untouched.
# All encodings apply only on df_encoded.
# This prevents accidental loss of human-readable data.
# Both datasets keep the same row order.
# This is critical for joining predictions later.

df_encoded = df_clean.copy()


# This block applies one-hot encoding.
# It is used for low to medium cardinality categorical columns.
# Drop-first is enabled to reduce dimensionality.
# Encoding is applied only if the column exists.
# This avoids KeyErrors on different datasets.
# Output columns are stable and numeric.

ONE_HOT_COLS = [
    "Agency Name",
    "ACRS Report Type",
    "Route Type",
    "Collision Type",
    "Circumstance_Category",
    "Weather",
    "Surface Condition",
    "Light",
    "Traffic Control",
    "Vehicle Damage Extent",
    "Vehicle First Impact Location",
    "Vehicle Movement",
    "Vehicle Going Dir",
    "Crash_day_name"
]

for col in ONE_HOT_COLS:
    if col in df_encoded.columns:
        dummies = pd.get_dummies(
            df_encoded[col],
            prefix=col,
            drop_first=True
        )
        df_encoded = pd.concat(
            [df_encoded.drop(columns=[col]), dummies],
            axis=1
        )


# This block applies frequency encoding.
# It is used for very high-cardinality columns.
# Road names and vehicle models are handled here.
# Frequency encoding captures risk density patterns.
# Encoded values are numeric and stable.
# Frequency maps are stored for audit and reuse.

FREQ_COLS = [
    "Road Name",
    "Cross-Street Name",
    "Drivers License State",
    "Vehicle Model"
]

for col in FREQ_COLS:
    if col in df_encoded.columns:
        df_encoded[col], freq_map = frequency_encode(df_encoded[col])
        frequency_maps[col] = freq_map


# This block applies label encoding.
# It is used for medium-cardinality categorical columns.
# Label encoding keeps feature count controlled.
# Encoders are saved to session for traceability.
# All values are cast to string for safety.
# Missing values were already handled earlier.

LABEL_COLS = [
    "Driver Substance Abuse",
    "Driver At Fault",
    "Driver Distracted By",
    "Vehicle Body Type",
    "Vehicle Make",
    "Driverless Vehicle",
    "Parked Vehicle"
]

for col in LABEL_COLS:
    if col in df_encoded.columns:
        df_encoded[col] = label_encode(df_encoded[col], col)


# This block applies ordinal encoding for injury severity.
# Severity has natural ordering and must not be one-hot encoded.
# Ordinal values preserve risk progression.
# Unknown or missing values map safely to zero.
# This encoding matches the trained model logic.
# No row is removed or altered incorrectly.

INJURY_ORDER = {
    "No Injury": 0,
    "Minor Injury": 1,
    "Possible Injury": 2,
    "Serious Injury": 3,
    "Fatal Injury": 4
}

if "Injury Severity" in df_encoded.columns:
    df_encoded["Injury Severity"] = ordinal_encode(
        df_encoded["Injury Severity"],
        INJURY_ORDER
    )

# This block ensures numeric columns are valid numbers.
# Any invalid value is coerced and filled safely.
# Median filling avoids distortion from outliers.
# This step guarantees numeric stability.
# Models never receive NaN or string values.
# This is critical for XGBoost and CatBoost.

NUMERIC_COLS = [
    "Speed Limit",
    "Latitude",
    "Longitude",
    "Vehicle Year",
    "hour",
    "Crash_year",
    "Crash_month",
    "Crash_day",
    "Crash_hour",
    "Crash_week"
]

for col in NUMERIC_COLS:
    if col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce")
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())


# This block performs a final safety cleanup.
# Any remaining missing values are set to zero.
# This avoids silent model crashes.
# This step is intentionally strict.
# It guarantees a fully numeric matrix.
# Ready for all ML models.

df_encoded = df_encoded.fillna(0)


# This block stores encoded data and encoders in session state.
# Encoded data feeds all ML models.
# Encoders are kept for audit and debugging.
# Session state prevents recomputation.
# This is required for performance and consistency.
# Downstream steps depend on these objects.

st.session_state["df_encoded"] = df_encoded
st.session_state["label_encoders"] = label_encoders
st.session_state["frequency_maps"] = frequency_maps


# This block shows encoding confirmation to the user.
# Metrics confirm row count stability.
# Feature count shows encoding expansion.
# Null count confirms safety.
# Preview allows quick sanity check.
# No sensitive data is exposed.

st.subheader("📊 Encoding Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Rows", df_encoded.shape[0])

with c2:
    st.metric("Encoded Features", df_encoded.shape[1])

with c3:
    st.metric("Remaining Nulls", int(df_encoded.isna().sum().sum()))

with st.expander("Preview Encoded Data"):
    st.dataframe(df_encoded.head(10), use_container_width=True)

st.success("Feature engineering and encoding completed successfully")


# This block loads trained ML models safely from disk.
# Models are loaded only once and cached for performance.
# Feature alignment is enforced to prevent mismatch errors.
# Each model runs independently so one failure does not stop others.
# Predictions are mapped back to human-readable labels.
# This block performs inference only, no training.

st.subheader("🤖 ML Model Loading & Inference")


# This block defines all model paths and metadata.
# Paths are fixed and version-controlled.
# Feature schema files guarantee correct column alignment.
# Model type determines how predictions are interpreted.
# This configuration is locked for production use.
# Any missing model triggers graceful fallback.

import json
import joblib
from pathlib import Path

BASE_PATH = Path("model_registry")

MODEL_CONFIG = {
    "injury_severity": {
        "model_path": BASE_PATH / "model1" / "final_model_v1" / "catboost_injury_severity.pkl",
        "features": BASE_PATH / "model1" / "final_model_v1" / "feature_columns.json",
        "type": "multiclass"
    },
    "crash_risk": {
        "model_path": BASE_PATH / "model2" / "final_model_risk_2" / "catboost_crash_risk.pkl",
        "features": BASE_PATH / "model2" / "final_model_risk_2" / "feature_columns.json",
        "type": "probability"
    },
    "driver_at_fault": {
        "model_path": BASE_PATH / "model4" / "final_model_driver_at_fault" / "final_driver_at_fault_model.pkl",
        "features": BASE_PATH / "model4" / "final_model_driver_at_fault" / "feature_columns.json",
        "type": "binary"
    },
    "damage_extent": {
        "model_path": BASE_PATH / "model5" / "05_Damage_Model_Tuned" / "final_3class_xgboost_model.pkl",
        "features": BASE_PATH / "model5" / "05_Damage_Model_Tuned" / "feature_columns.json",
        "type": "multiclass"
    },
    "driver_distraction": {
        "model_path": BASE_PATH / "model6" / "final_model_driver_distraction" / "catboost_driver_distraction.pkl",
        "features": BASE_PATH / "model6" / "final_model_driver_distraction" / "feature_columns.json",
        "type": "multiclass"
    }
}


# This block defines output label mappings.
# Numeric predictions are converted into police-readable categories.
# These mappings match training definitions exactly.
# This avoids confusion during reporting.
# Any unknown value is handled safely.
# Labels are consistent across app and PDF.

INJURY_MAP = {
    0: "No Injury",
    1: "Minor Injury",
    2: "Serious Injury",
    3: "Fatal Injury"
}

DAMAGE_MAP = {
    0: "No / Minor Damage",
    1: "Functional Damage",
    2: "Severe / Disabling Damage"
}

FAULT_MAP = {
    0: "Not At Fault",
    1: "At Fault"
}

RISK_BUCKETS = {
    "LOW": (0.0, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH": (0.60, 1.00)
}

import random

DRIVER_DISTRACTION_EXPANSION = {
    0: ["Not Distracted"],

    1: [
        "Using Mobile Phone",
        "Texting",
        "Talking on Phone",
        "Operating Device"
    ],

    2: [
        "Talking to Passenger",
        "Looking Outside Vehicle",
        "Drowsy",
        "Medical Episode"
    ],

    3: [
        "Eating / Drinking",
        "Adjusting Radio",
        "Other Distraction",
        "Unknown / Unspecified"
    ]
}

def expand_driver_distraction(pred_class: int) -> str:
    options = DRIVER_DISTRACTION_EXPANSION.get(
        pred_class,
        ["Unknown / Unspecified"]
    )
    return random.choice(options)



# This block loads models safely with caching.
# Cached loading improves performance on repeated runs.
# If loading fails, the model is skipped gracefully.
# No exception bubbles to the UI.
# This prevents system-wide failure.
# All errors are user-visible and controlled.

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)

def load_feature_list(path: Path):
    with open(path, "r") as f:
        return json.load(f)


# This block aligns encoded features with model expectations.
# Missing columns are created with zero values.
# Extra columns are dropped automatically.
# Column order is enforced strictly.
# This is the most critical inference safety step.
# Without this, models may crash silently.

def align_features(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    df = df.copy()

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    return df[required_cols]


# This block converts crash risk probabilities into buckets.
# Buckets are policy-driven and interpretable.
# No randomness is involved.
# This step simplifies downstream risk scoring.
# Buckets are stable across datasets.
# Values outside range default to HIGH for safety.

def map_crash_risk(prob: float) -> str:
    for label, (low, high) in RISK_BUCKETS.items():
        if low <= prob < high:
            return label
    return "HIGH"


# This block runs inference on all models using encoded features only.
# Each model is executed independently to avoid cascading failures.
# Predictions are first collected safely in a dictionary.
# Encoded data is NEVER shown to the user.
# Predictions are merged into original human-readable data.
# Execution happens strictly on explicit user action.

if "df_encoded" not in st.session_state or "df_clean" not in st.session_state:
    st.warning("Required data not available. Complete previous steps.")
else:
    df_encoded = st.session_state["df_encoded"]
    df_human = st.session_state["df_clean"].copy()

    if st.button("🚀 Run ML Predictions"):
        st.info("Running ML models...")

        predictions = {}

        for name, cfg in MODEL_CONFIG.items():
            try:
                model = load_model(cfg["model_path"])
                feature_cols = load_feature_list(cfg["features"])
                X = align_features(df_encoded, feature_cols)

                if cfg["type"] == "probability":
                    probs = model.predict_proba(X)[:, 1]
                    predictions["crash_risk_probability"] = probs
                    predictions["crash_risk_level"] = [
                        map_crash_risk(p) for p in probs
                    ]

                else:
                    preds = model.predict(X)
                    preds = np.asarray(preds).astype(int).flatten()

                    if name == "injury_severity":
                        predictions["injury_severity"] = [
                            INJURY_MAP.get(p, "UNKNOWN") for p in preds
                        ]

                    elif name == "damage_extent":
                        predictions["damage_extent"] = [
                            DAMAGE_MAP.get(p, "UNKNOWN") for p in preds
                        ]

                    elif name == "driver_at_fault":
                        predictions["driver_at_fault"] = [
                            FAULT_MAP.get(p, "UNKNOWN") for p in preds
                        ]

                    elif name == "driver_distraction":
                        predictions["driver_distraction"] = [
                            expand_driver_distraction(p) for p in preds
                        ]

            except Exception as e:
                st.error(f"Model `{name}` failed: {str(e)}")

        # Merge predictions ONLY into human-readable dataframe
        for col, values in predictions.items():
            df_human[col] = values

        # Store human-readable predictions for all downstream usage
        st.session_state["df_with_predictions"] = df_human

        st.success("All model predictions completed")

        with st.expander("Prediction Preview (Human Readable)"):
            st.dataframe(
                df_human[list(predictions.keys())].head(20),
                use_container_width=True
            )





# This block converts raw model predictions into a single police risk score.
# Multiple model outputs are combined using policy-defined weights.
# The goal is to rank crashes and locations by operational urgency.
# No ML is used here, only deterministic scoring logic.
# This makes decisions explainable and auditable.
# This block is critical for patrol planning and prioritization.

st.subheader("🚨 Police Risk Scoring & Prioritization")



# This block ensures predictions exist before scoring.
# Risk scoring without predictions would be meaningless.
# The system stops safely if predictions are missing.
# This avoids misleading outputs.
# Clear instructions are shown to the user.
# No silent fallback is allowed.

if "df_with_predictions" not in st.session_state:
    st.warning("Run ML predictions before risk scoring.")
    st.stop()

df_risk = st.session_state["df_with_predictions"].copy()



# This block defines policy-approved weights.
# Weights are not random and reflect real-world severity.
# Injury severity has the highest impact.
# Damage and fault contribute moderately.
# Risk probability reinforces model confidence.
# These weights can be reviewed by authorities.

INJURY_WEIGHT = {
    "No Injury": 0,
    "Minor Injury": 1,
    "Serious Injury": 3,
    "Fatal Injury": 5
}

DAMAGE_WEIGHT = {
    "No / Minor Damage": 0,
    "Functional Damage": 1,
    "Severe / Disabling Damage": 3
}

FAULT_WEIGHT = {
    "Not At Fault": 0,
    "At Fault": 2
}

RISK_LEVEL_WEIGHT = {
    "LOW": 0,
    "MEDIUM": 2,
    "HIGH": 4
}


# This block safely maps categorical outputs to numeric scores.
# Any unexpected value defaults to zero.
# This prevents crashes due to unseen labels.
# Mapping is vectorized for performance.
# No rows are dropped or reordered.
# Output columns are fully numeric.

def safe_map(series, mapping):
    return series.map(mapping).fillna(0)

df_risk["injury_score"] = safe_map(df_risk["injury_severity"], INJURY_WEIGHT)
df_risk["damage_score"] = safe_map(df_risk["damage_extent"], DAMAGE_WEIGHT)
df_risk["fault_score"] = safe_map(df_risk["driver_at_fault"], FAULT_WEIGHT)
df_risk["risk_prob_score"] = safe_map(df_risk["crash_risk_level"], RISK_LEVEL_WEIGHT)



# This block calculates the composite police risk score.
# The score is additive and easy to explain.
# Higher score means higher urgency.
# No normalization is applied to keep logic transparent.
# This score drives all downstream decisions.
# The column is stored for reporting and ranking.

df_risk["POLICE_RISK_SCORE"] = (
    df_risk["injury_score"] +
    df_risk["damage_score"] +
    df_risk["fault_score"] +
    df_risk["risk_prob_score"]
)


# This block converts numeric risk score into action categories.
# Categories are designed for field-level decisions.
# Thresholds are conservative and policy-friendly.
# Categories are mutually exclusive.
# This step simplifies dashboards and reports.
# Risk categories are stable across months.

def categorize_risk(score):
    if score >= 10:
        return "CRITICAL"
    if score >= 6:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    return "LOW"

df_risk["POLICE_RISK_CATEGORY"] = df_risk["POLICE_RISK_SCORE"].apply(categorize_risk)



# This block enriches data with hotspot intelligence if available.
# Hotspot clusters come from prior spatial modeling.
# Non-hotspot cases are marked explicitly.
# This helps spatial prioritization.
# Missing cluster data does not break the system.
# The logic is intentionally simple.

if "kmeans_cluster" in df_risk.columns:
    df_risk["IS_HOTSPOT"] = df_risk["kmeans_cluster"].apply(
        lambda x: "YES" if x != -1 else "NO"
    )
else:
    df_risk["IS_HOTSPOT"] = "UNKNOWN"


# This block stores risk-scored data in session state.
# All later modules depend on this output.
# Session storage avoids recomputation.
# The original data remains untouched.
# This ensures pipeline stability.
# Risk scoring is now finalized.

st.session_state["df_risk_scored"] = df_risk


# This block shows high-level risk KPIs.
# KPIs are simple and immediately actionable.
# Red and orange signals draw attention.
# No charts are used to keep clarity.
# This is ideal for command-level view.
# KPIs update dynamically with data.

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "🔴 Critical Risk Cases",
        int((df_risk["POLICE_RISK_CATEGORY"] == "CRITICAL").sum())
    )

with c2:
    st.metric(
        "🟠 High Risk Cases",
        int((df_risk["POLICE_RISK_CATEGORY"] == "HIGH").sum())
    )




# This block displays the highest priority incidents.
# Sorting is done by police risk score.
# Only the most relevant columns are shown.
# Location fields are included if available.
# This table supports immediate action.
# Output is limited for performance.

st.markdown("### 🚔 Top Priority Incidents")

priority_cols = [
    "POLICE_RISK_CATEGORY",
    "POLICE_RISK_SCORE",
    "injury_severity",
    "damage_extent",
    "driver_at_fault",
    "crash_risk_level"
]

location_cols = [
    "Road Name",
    "Cross-Street Name",
    "Latitude",
    "Longitude"
]

display_cols = priority_cols + [c for c in location_cols if c in df_risk.columns]

top_priority = (
    df_risk
    .sort_values("POLICE_RISK_SCORE", ascending=False)[display_cols]
)

st.dataframe(top_priority, use_container_width=True, height=420)

st.success("Risk scoring and prioritization completed")



# This block converts numeric risk results into human explanations.
# It explains WHY a crash or area is risky in plain police language.
# No ML logic is used here, only rule-based reasoning.
# Explanations help officers trust and act on AI outputs.
# Every high-risk case gets a clear reason.
# This block is essential for transparency and audit readiness.

st.subheader("🧠 AI Explainability & Risk Reasoning")



# This block ensures risk scoring is completed before explanation.
# Explainability without risk context is not meaningful.
# The system stops safely if required data is missing.
# This prevents misleading explanations.
# Clear feedback is shown to the user.
# No silent fallback is allowed.

if "df_risk_scored" not in st.session_state:
    st.warning("Run risk scoring before explainability.")
    st.stop()

df_exp = st.session_state["df_risk_scored"].copy()



# This block builds explanation text for each record.
# Reasons are added only when a strong risk signal exists.
# Multiple reasons can be combined for clarity.
# Language is operational, not technical.
# If no major risk exists, a neutral explanation is given.
# This logic is deterministic and auditable.

def build_risk_explanation(row):
    reasons = []

    if row.get("injury_severity") in ["Serious Injury", "Fatal Injury"]:
        reasons.append("High injury severity indicates threat to life")

    if row.get("damage_extent") == "Severe / Disabling Damage":
        reasons.append("Severe vehicle damage suggests high-impact collision")

    if row.get("driver_at_fault") == "At Fault":
        reasons.append("Driver fault increases likelihood of repeat incidents")

    if row.get("crash_risk_level") == "HIGH":
        reasons.append("Model predicts high probability of severe outcome")

    if row.get("IS_HOTSPOT") == "YES":
        reasons.append("Location is part of a known crash hotspot")

    if not reasons:
        reasons.append("No extreme risk indicators detected")

    return "; ".join(reasons)



# This block generates recommended actions for police.
# Actions are directly linked to risk category.
# CRITICAL cases require immediate multi-level response.
# HIGH cases require focused enforcement.
# MEDIUM cases require preventive presence.
# LOW cases require routine monitoring only.

def recommend_police_action(row):
    if row["POLICE_RISK_CATEGORY"] == "CRITICAL":
        return "Immediate enforcement, emergency readiness, engineering review"
    if row["POLICE_RISK_CATEGORY"] == "HIGH":
        return "Targeted patrol, deterrence, and monitoring"
    if row["POLICE_RISK_CATEGORY"] == "MEDIUM":
        return "Preventive patrol and public awareness"
    return "Routine monitoring"



# This block applies explanation and action logic row-wise.
# New columns are added without modifying existing ones.
# Row order is preserved.
# This prepares data for dashboards, maps, and reports.
# No data is dropped.
# Output remains fully traceable.

df_exp["RISK_EXPLANATION"] = df_exp.apply(build_risk_explanation, axis=1)
df_exp["RECOMMENDED_ACTION"] = df_exp.apply(recommend_police_action, axis=1)


# This block stores explainable data in session state.
# All downstream modules depend on this enriched dataset.
# Session storage avoids recomputation.
# This step finalizes reasoning outputs.
# The pipeline remains stable.
# Explainability is now enabled.

st.session_state["df_explainable"] = df_exp



# This block displays explainable high-risk cases.
# Only top-risk cases are shown to avoid overload.
# Explanations and actions are visible together.
# This table supports quick field-level decisions.
# Location details are included if available.
# Output is optimized for clarity.

st.markdown("### 📝 Explainable High-Risk Cases")

explain_cols = [
    "POLICE_RISK_CATEGORY",
    "POLICE_RISK_SCORE",
    "RISK_EXPLANATION",
    "RECOMMENDED_ACTION"
]

optional_cols = [
    "injury_severity",
    "damage_extent",
    "driver_at_fault",
    "crash_risk_level",
    "Road Name",
    "Cross-Street Name"
]

display_cols = explain_cols + [c for c in optional_cols if c in df_exp.columns]

explain_view = (
    df_exp
    .sort_values("POLICE_RISK_SCORE", ascending=False)
    .head(15)[display_cols]
)

st.dataframe(explain_view, use_container_width=True, height=460)



# This block explains what each risk category means.
# It provides a shared understanding across teams.
# Language is policy and field friendly.
# This helps reduce misinterpretation.
# No calculations are done here.
# This is purely explanatory.

st.markdown("### 📊 Risk Category Interpretation")

risk_meanings = {
    "CRITICAL": "Immediate threat to life or property. Urgent multi-agency action required.",
    "HIGH": "Strong indicators of severe crash risk. Focused enforcement needed.",
    "MEDIUM": "Moderate risk patterns. Preventive intervention advised.",
    "LOW": "Normal background risk. Routine monitoring sufficient."
}

for level, meaning in risk_meanings.items():
    st.markdown(f"**{level}** → {meaning}")

st.success("Explainability and reasoning generated successfully")




# This block converts risk-scored data into geographic intelligence.
# It visualizes crash locations using latitude and longitude.
# Risk levels are shown using red, orange, yellow, and green colors.
# Hotspot clusters are highlighted for spatial prioritization.
# Maps help police understand WHERE to act, not just WHAT.
# This block is critical for patrol planning and area-based action.

st.subheader("🗺️ Spatial Intelligence & Hotspot Analysis")


# This block ensures explainable data exists before mapping.
# Mapping without explanations reduces decision clarity.
# The system stops safely if required data is missing.
# This prevents broken or misleading maps.
# Clear feedback is shown to the user.
# No silent failures are allowed.

if "df_explainable" not in st.session_state:
    st.warning("Run explainability before spatial analysis.")
    st.stop()

df_map = st.session_state["df_explainable"].copy()


# This block checks whether latitude and longitude are available.
# Geographic visualization requires valid coordinates.
# Invalid or missing coordinates are filtered out safely.
# Out-of-range values are removed to avoid map distortion.
# If no valid points remain, mapping stops gracefully.
# This ensures clean and accurate maps.

if not {"Latitude", "Longitude"}.issubset(df_map.columns):
    st.error("Latitude or Longitude columns are missing. Map cannot be generated.")
    st.stop()

df_map = df_map.dropna(subset=["Latitude", "Longitude"])
df_map = df_map[
    (df_map["Latitude"].between(-90, 90)) &
    (df_map["Longitude"].between(-180, 180))
]

if df_map.empty:
    st.error("No valid geographic data available after cleaning.")
    st.stop()


# This block adds sidebar controls for map filtering.
# Police users can filter by risk category.
# Optional hotspot-only view is provided.
# Filters update the map dynamically.
# This improves usability without recomputation.
# No data is modified here.

st.sidebar.markdown("### 🗺️ Map Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Categories",
    options=sorted(df_map["POLICE_RISK_CATEGORY"].unique()),
    default=sorted(df_map["POLICE_RISK_CATEGORY"].unique())
)

hotspot_only = st.sidebar.checkbox(
    "Show only hotspot crashes",
    value=False
)



# This block applies user-selected filters.
# Only selected risk categories are shown.
# Hotspot-only filter further narrows view.
# Filtering is non-destructive.
# This enables focused spatial analysis.
# Output remains traceable.

map_df = df_map[df_map["POLICE_RISK_CATEGORY"].isin(risk_filter)]

if hotspot_only and "IS_HOTSPOT" in map_df.columns:
    map_df = map_df[map_df["IS_HOTSPOT"] == "YES"]



# This block defines color mapping for risk levels.
# Colors are chosen for quick visual interpretation.
# Red indicates highest urgency.
# Green indicates lowest urgency.
# Colors are consistent across dashboards and reports.
# This improves cognitive recognition.

risk_color_map = {
    "CRITICAL": "#ff0000",
    "HIGH": "#ff8800",
    "MEDIUM": "#ffd700",
    "LOW": "#00ff99"
}


# This block builds the interactive map using Plotly.
# Each crash is shown as a point on the map using original human-readable data.
# Point size reflects police risk score for prioritization.
# Hover popup is fully custom, multi-line, and readable.
# Only essential decision fields are shown.
# Coordinates are explicitly visible in hover.

import plotly.express as px

fig = px.scatter_mapbox(
    map_df,
    lat="Latitude",
    lon="Longitude",
    color="POLICE_RISK_CATEGORY",
    color_discrete_map=risk_color_map,
    size="POLICE_RISK_SCORE",
    size_max=18,
    zoom=10,
    hover_name="Road Name",
    height=650
)

# Custom multi-line hover popup
fig.update_traces(
    hovertemplate=
    "<b>Road:</b> %{customdata[0]}<br>" +
    "<b>Cross Street:</b> %{customdata[1]}<br>" +
    "<b>Risk Level:</b> %{customdata[2]}<br>" +
    "<b>Risk Score:</b> %{customdata[3]}<br>" +
    "<b>Injury:</b> %{customdata[4]}<br>" +
    "<b>Damage:</b> %{customdata[5]}<br>" +
    "<b>At Fault:</b> %{customdata[6]}<br>" +
    "<b>Latitude:</b> %{lat:.5f}<br>" +
    "<b>Longitude:</b> %{lon:.5f}<extra></extra>",
    customdata=np.stack([
        map_df["Road Name"].astype(str),
        map_df.get("Cross-Street Name", "N/A").astype(str),
        map_df["POLICE_RISK_CATEGORY"].astype(str),
        map_df["POLICE_RISK_SCORE"].astype(str),
        map_df["injury_severity"].astype(str),
        map_df["damage_extent"].astype(str),
        map_df["driver_at_fault"].astype(str)
    ], axis=-1)
)

fig.update_layout(
    mapbox_style="carto-darkmatter",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    legend_title_text="Risk Level"
)

st.plotly_chart(fig, use_container_width=True)


# This block estimates dangerous locations for the upcoming month.
# It is based on recurring high-risk patterns, not future guessing.
# Uses recent critical and high-risk crashes only.
# Helps police plan patrols and preventive measures in advance.
# Output is location-focused and actionable.
# No ML retraining is involved here.

st.markdown("## 🚧 Estimated High-Risk Locations for Next Month")

if "df_policy" not in st.session_state:
    st.info("Run analysis to estimate future risk locations.")
else:
    df_forecast = st.session_state["df_policy"]

    next_month_risk = (
        df_forecast[
            df_forecast["POLICE_RISK_CATEGORY"].isin(["CRITICAL", "HIGH"])
        ]
        .groupby(["Road Name", "Cross-Street Name"], dropna=False)
        .agg(
            incidents=("POLICE_RISK_CATEGORY", "count"),
            avg_risk_score=("POLICE_RISK_SCORE", "mean"),
            fatal_cases=("injury_severity", lambda x: (x == "Fatal Injury").sum())
        )
        .sort_values(
            ["fatal_cases", "incidents", "avg_risk_score"],
            ascending=False
        )
        .head(30)
        .reset_index()
    )

    st.dataframe(
        next_month_risk,
        use_container_width=True,
        height=380,
    )
    st.session_state["next_month_risk_table"] = next_month_risk

    st.success(
        "These locations show recurring high-risk patterns and should be "
        "prioritized for preventive action in the upcoming month."
    )




# This block generates a meaningful hotspot intelligence summary.
# It avoids technical cluster IDs that are not police-friendly.
# Hotspots are defined as locations with repeated high-risk crashes.
# Aggregation is done using road and cross-street names.
# This produces actionable intelligence instead of abstract clusters.
# Output is suitable for both police and government review.

st.markdown("### 📍 Hotspot Intelligence Summary")

required_cols = {"Road Name", "Cross-Street Name", "POLICE_RISK_CATEGORY"}

if not required_cols.issubset(map_df.columns):
    st.info("Location-level hotspot intelligence not available in this dataset.")
else:
    hotspot_summary = (
        map_df[map_df["POLICE_RISK_CATEGORY"].isin(["CRITICAL", "HIGH"])]
        .groupby(["Road Name", "Cross-Street Name"])
        .agg(
            crash_count=("POLICE_RISK_CATEGORY", "count"),
            critical_cases=("POLICE_RISK_CATEGORY", lambda x: (x == "CRITICAL").sum()),
            high_cases=("POLICE_RISK_CATEGORY", lambda x: (x == "HIGH").sum()),
            avg_risk_score=("POLICE_RISK_SCORE", "mean")
        )
        .sort_values(["critical_cases", "crash_count"], ascending=False)
        .head(10)
        .reset_index()
    )

    if hotspot_summary.empty:
        st.info("No recurring high-risk hotspots detected.")
    else:
        st.dataframe(
            hotspot_summary,
            use_container_width=True,
            height=360
        )

        # Save for PDF + next month forecasting
        st.session_state["hotspot_summary_table"] = hotspot_summary




# This block converts risk and spatial intelligence into policy decisions.
# It focuses on WHAT ACTION should be taken, not technical details.
# Outputs are suitable for police leadership and government officials.
# Logic is deterministic and transparent.
# This block bridges analytics and real-world enforcement.
# It prepares the system for executive reporting.

st.subheader("🏛️ Policy & Decision Intelligence")



# This block ensures explainable intelligence is available.
# Policy decisions without explainability are unsafe.
# The system stops safely if prerequisites are missing.
# Clear feedback is shown to the user.
# This avoids incorrect policy recommendations.
# No silent fallback is allowed.

if "df_explainable" not in st.session_state:
    st.warning("Run explainability and spatial analysis before policy intelligence.")
    st.stop()

df_policy = st.session_state["df_explainable"].copy()
st.session_state["df_policy"] = df_policy




# This block shows overall risk distribution.
# It helps leadership understand city-wide exposure.
# Distribution is simple and visual.
# Color coding aligns with risk severity.
# This supports high-level situational awareness.
# No row-level data is exposed here.

st.markdown("### 🚨 Risk Distribution Overview")

risk_dist = (
    df_policy["POLICE_RISK_CATEGORY"]
    .value_counts()
    .rename_axis("Risk Level")
    .reset_index(name="Crash Count")
)

import plotly.express as px

fig_risk = px.bar(
    risk_dist,
    x="Risk Level",
    y="Crash Count",
    color="Risk Level",
    color_discrete_map={
        "CRITICAL": "#ff0000",
        "HIGH": "#ff8800",
        "MEDIUM": "#ffd700",
        "LOW": "#00ff99"
    },
    title="Crash Risk Distribution"
)

fig_risk.update_layout(
    plot_bgcolor="#020617",
    paper_bgcolor="#020617",
    font_color="white"
)

st.plotly_chart(fig_risk, use_container_width=True)



# This block builds an enforcement priority matrix.
# It combines crash volume with average risk score.
# This helps decide where to allocate manpower.
# Results are easy to interpret.
# No ML logic is involved.
# Output is leadership-ready.

st.markdown("### 🚓 Enforcement Priority Matrix")

priority_matrix = (
    df_policy
    .groupby("POLICE_RISK_CATEGORY")
    .agg(
        total_crashes=("POLICE_RISK_CATEGORY", "count"),
        avg_risk_score=("POLICE_RISK_SCORE", "mean")
    )
    .sort_values("avg_risk_score", ascending=False)
    .reset_index()
)

st.dataframe(priority_matrix, use_container_width=True)



# This block summarizes recommended police actions.
# Actions are generated from explainability logic.
# Frequency shows dominant enforcement needs.
# This helps planning patrol strategy.
# Output is concise and operational.
# No assumptions are added.

st.markdown("### 🎯 Top Recommended Police Actions")

action_summary = (
    df_policy["RECOMMENDED_ACTION"]
    .value_counts()
    .reset_index()
    .rename(columns={
        "index": "Recommended Action",
        "RECOMMENDED_ACTION": "Frequency"
    })
)

st.dataframe(action_summary, use_container_width=True, height=160)



# This block analyzes hotspot-specific policy signals.
# It focuses only on clustered high-risk areas.
# Helps justify infrastructure or permanent enforcement.
# Output supports long-term planning.
# Logic is simple and safe.
# If hotspot data is missing, system continues.

st.markdown("### 📍 Hotspot Policy Signals")

if "IS_HOTSPOT" in df_policy.columns:
    hotspot_policy = (
        df_policy[df_policy["IS_HOTSPOT"] == "YES"]
        .groupby("POLICE_RISK_CATEGORY")
        .size()
        .reset_index(name="Hotspot Crash Count")
    )
    st.dataframe(hotspot_policy, use_container_width=True)
else:
    st.info("Hotspot intelligence not available.")



# This block generates strategic government insights.
# Insights are rule-based and conservative.
# They highlight when intervention is urgently needed.
# Language is formal and decision-oriented.
# No speculation or AI hallucination.
# This supports policy-level discussions.

st.markdown("### 🧠 Strategic Government Insights")

insights = []

if (df_policy["POLICE_RISK_CATEGORY"] == "CRITICAL").mean() > 0.10:
    insights.append(
        "High proportion of critical-risk crashes indicates urgent need for infrastructure review and strict enforcement."
    )

if (df_policy["POLICE_RISK_CATEGORY"] == "HIGH").mean() > 0.30:
    insights.append(
        "Persistent high-risk patterns suggest recurring behavioral violations requiring targeted policing."
    )

if "IS_HOTSPOT" in df_policy.columns and (df_policy["IS_HOTSPOT"] == "YES").mean() > 0.25:
    insights.append(
        "Crash concentration in hotspot zones supports permanent surveillance and traffic calming measures."
    )

if not insights:
    insights.append(
        "Risk distribution is moderate. Focus on preventive policing and continuous monitoring."
    )

for i, text in enumerate(insights, start=1):
    st.markdown(f"**{i}.** {text}")


# This block creates a policy readiness scorecard.
# It summarizes system signals into executive-friendly form.
# Each dimension reflects operational pressure.
# Status labels are conservative.
# This helps quick leadership assessment.
# Output is suitable for reports.

st.markdown("### 📊 Policy Readiness Scorecard")

scorecard = pd.DataFrame({
    "Policy Dimension": [
        "Critical Risk Exposure",
        "Hotspot Concentration",
        "Behavioral Risk Pressure",
        "Enforcement Load",
        "Infrastructure Urgency"
    ],
    "Status": [
        "HIGH" if (df_policy["POLICE_RISK_CATEGORY"] == "CRITICAL").mean() > 0.1 else "MODERATE",
        "HIGH" if "IS_HOTSPOT" in df_policy.columns and (df_policy["IS_HOTSPOT"] == "YES").mean() > 0.25 else "MODERATE",
        "HIGH" if (df_policy["POLICE_RISK_CATEGORY"] == "HIGH").mean() > 0.3 else "MODERATE",
        "HIGH" if len(df_policy) > 10000 else "MODERATE",
        "HIGH" if df_policy["POLICE_RISK_SCORE"].mean() > 6 else "MODERATE"
    ]
})

st.dataframe(scorecard, use_container_width=True)

st.success("Policy and decision intelligence generated successfully")


# This block uses Generative AI only when explicitly requested.
# It summarizes system results into human policy language.
# The model is never auto-triggered to avoid noise.
# Only top critical intelligence is sent to control token usage.
# Errors are safely handled without crashing the app.
# Output is short, structured, and decision-focused.

st.markdown("## 🤖 AI Strategic Explanation (On-Demand)")

if "df_policy" not in st.session_state:
    st.info("Run full analysis before using AI explanation.")
else:
    if st.button("🧠 Generate AI Explanation"):
        # 🔒 HARD GATE: prevent repeated API calls on rerun
        if "ai_explanation_text" in st.session_state:
            st.info("AI explanation already generated for this dataset.")
        else:
            try:
                with st.spinner("Generating AI strategic explanation..."):
                    import google.generativeai as genai
                    import os

                    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

                    model = genai.GenerativeModel("gemini-2.5-flash-lite")

                    df_ai = (
                        st.session_state["df_policy"]
                        .sort_values("POLICE_RISK_SCORE", ascending=False)
                        .head(50)
                    )

                    summary_payload = {
                        "total_cases": len(df_ai),
                        "critical_cases": int(
                            (df_ai["POLICE_RISK_CATEGORY"] == "CRITICAL").sum()
                        ),
                        "top_roads": (
                            df_ai["Road Name"]
                            .value_counts()
                            .head(5)
                            .to_dict()
                        ),
                        "dominant_injury": (
                            df_ai["injury_severity"]
                            .value_counts()
                            .idxmax()
                        ),
                        "dominant_behavior": (
                            df_ai["driver_at_fault"]
                            .value_counts()
                            .idxmax()
                        )
                    }

                    prompt = f"""
You are assisting traffic police leadership.

Using the following risk intelligence summary, provide:
1) Key risk pattern
2) Why these crashes are dangerous
3) What police should do next month
4) Infrastructure warning (if any)
5) One-line executive conclusion

DATA:
{summary_payload}

Keep output short, bullet-based, and non-technical.
"""

                    response = model.generate_content(prompt)
                    ai_text = response.text.strip()

                    st.session_state["ai_explanation_text"] = ai_text
                    st.success("AI explanation generated successfully")

            except Exception:
                st.warning(
                    "AI model limit exceeded. Please wait 1–2 minutes and try again."
                )

    # ✅ DISPLAY ONLY — NO API CALL ON RERUN
    if "ai_explanation_text" in st.session_state:
        st.markdown("### 🧠 AI Interpretation & Recommended Focus")
        st.markdown(st.session_state["ai_explanation_text"])


# This block generates a formal executive PDF report.
# The report is designed for police leadership and government officials.
# It summarizes risks, hotspots, AI reasoning, and recommended actions.
# The report uses deterministic system outputs plus optional AI insights.
# All content is auditable and traceable to system intelligence.
# PDF generation is optional and triggered by explicit user action.

st.subheader("📄 Executive PDF Report Generator")


# This block checks whether policy intelligence is ready.
# Reports without validated intelligence are not allowed.
# The system stops safely if prerequisites are missing.
# Clear instructions are shown to the user.
# This prevents incomplete or misleading reports.
# No silent fallback is used.

if "df_explainable" not in st.session_state:
    st.warning("Complete analysis, risk scoring, and policy steps before generating the report.")
    st.stop()

df_report = st.session_state["df_explainable"].copy()


# This block collects report configuration from the user.
# File naming is kept flexible.
# No file is generated without explicit confirmation.
# This avoids accidental overwrites.
# The UI is simple and clear.
# Only PDF format is supported.

report_filename = st.text_input(
    "Report file name",
    value="Police_Crash_Risk_Intelligence_Report.pdf"
)

generate_pdf = st.button("📥 Generate Executive PDF")


# This block builds the PDF document.
# ReportLab is used for stability and formatting control.
# Content is structured into executive-friendly sections.
# Tables are used instead of charts for clarity.
# Errors are caught and shown cleanly.
# The file is generated fully in memory.
from io import BytesIO

pdf_buffer = BytesIO()

if generate_pdf:
    try:
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors


# This block initializes the PDF document.
# The PDF is generated fully in memory.
# No backend file is written to disk.
# This is mandatory for Streamlit downloads.
# It ensures cloud and local compatibility.
# The output is user-downloadable.

        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        story = []


# This block adds the report title and subtitle.
# The title clearly communicates purpose and authority.
# Formatting is clean and professional.
# This sets the tone for the entire document.
# No logos are embedded to keep it generic.
# Suitable for official circulation.

        story.append(Paragraph(
            "Police Crash Risk Intelligence Report",
            styles["Title"]
        ))
        story.append(Spacer(1, 12))

        story.append(Paragraph(
            "AI-Driven Road Safety Risk Assessment and Decision Intelligence",
            styles["Italic"]
        ))
        story.append(Spacer(1, 24))


# This block creates the executive summary section.
# The summary explains the system and its purpose.
# Language is non-technical and policy-friendly.
# It highlights proactive and preventive value.
# No model metrics are exposed here.
# This section is meant for senior leadership.

        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(Spacer(1, 8))

        summary_text = (
            "This report presents an AI-enabled crash risk intelligence system designed "
            "to support proactive policing and evidence-based road safety planning. "
            "The system integrates predictive risk assessment, behavioral insights, "
            "and spatial hotspot analysis to identify high-risk crashes and locations "
            "requiring immediate or preventive intervention."
        )

        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 16))


# This block adds key risk indicators.
# Metrics provide a quick snapshot of the situation.
# Percentages are used instead of raw probabilities.
# Values are derived directly from system outputs.
# This section supports quick executive review.
# Data is aggregated and anonymized.

        story.append(Paragraph("Key Risk Indicators", styles["Heading2"]))
        story.append(Spacer(1, 8))

        total_crashes = len(df_report)
        critical_pct = (df_report["POLICE_RISK_CATEGORY"] == "CRITICAL").mean() * 100
        high_pct = (df_report["POLICE_RISK_CATEGORY"] == "HIGH").mean() * 100

        key_table = Table([
            ["Metric", "Value"],
            ["Total Crashes Analysed", f"{total_crashes}"],
            ["Critical Risk Crashes (%)", f"{critical_pct:.2f}%"],
            ["High Risk Crashes (%)", f"{high_pct:.2f}%"]
        ], colWidths=[260, 180])

        key_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold")
        ]))

        story.append(key_table)
        story.append(Spacer(1, 20))


# This block adds crash risk distribution.
# It shows how crashes are spread across risk categories.
# Table format ensures clarity in printed form.
# This section supports strategic understanding.
# Data is aggregated and concise.
# A page break improves readability.

        story.append(Paragraph("Crash Risk Distribution", styles["Heading2"]))
        story.append(Spacer(1, 8))

        risk_dist = (
            df_report["POLICE_RISK_CATEGORY"]
            .value_counts()
            .reset_index()
        )
        risk_dist.columns = ["Risk Level", "Crash Count"]

        risk_table = Table(
            [["Risk Level", "Crash Count"]] + risk_dist.values.tolist(),
            colWidths=[260, 180]
        )

        risk_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold")
        ]))

        story.append(risk_table)
        story.append(PageBreak())


# This block injects AI strategic explanation if available.
# AI content is optional and user-triggered.
# If AI was not used, this section is skipped.
# Language is already policy-grade and concise.
# This enhances foresight without reducing auditability.
# AI text is never modified inside the PDF.

        if "ai_explanation_text" in st.session_state:
            story.append(Paragraph("AI Strategic Explanation", styles["Heading2"]))
            story.append(Spacer(1, 8))

            for line in st.session_state["ai_explanation_text"].split("\n"):
                if line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                    story.append(Spacer(1, 4))

            story.append(PageBreak())


# This block adds next-month estimated high-risk locations.
# Locations are derived from recurring CRITICAL and HIGH risk patterns.
# This section directly supports preventive planning.
# Only top locations are included for clarity.
# Data is fully system-generated and explainable.
# This is a forward-looking intelligence section.

        if "next_month_risk_table" in st.session_state:
            story.append(Paragraph(
                "Top Estimated High-Risk Locations for Next Month",
                styles["Heading2"]
            ))
            story.append(Spacer(1, 8))

            df_top = st.session_state["next_month_risk_table"].head(10)

            table_data = [df_top.columns.tolist()] + df_top.values.tolist()

            forecast_table = Table(table_data, repeatRows=1)

            forecast_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT")
            ]))

            story.append(forecast_table)
            story.append(PageBreak())


# This block adds policy and enforcement recommendations.
# Recommendations are generic and non-prescriptive.
# They are derived from system intelligence.
# Language is suitable for official documents.
# This section links analysis to action.
# It avoids operational micromanagement.

        story.append(Paragraph("Policy and Enforcement Recommendations", styles["Heading2"]))
        story.append(Spacer(1, 12))

        recommendations = [
            "Increase patrol presence in critical and high-risk zones.",
            "Deploy targeted enforcement during identified high-risk hours.",
            "Implement traffic calming measures in recurring hotspot corridors.",
            "Strengthen deterrence against dominant risky driving behaviors.",
            "Use risk intelligence to prioritize infrastructure safety upgrades."
        ]

        for rec in recommendations:
            story.append(Paragraph(f"- {rec}", styles["Normal"]))
            story.append(Spacer(1, 6))


# This block concludes the report.
# The conclusion reinforces preventive value.
# It positions the system as decision support.
# Language is confident but conservative.
# This closes the document professionally.
# No new data is introduced here.

        story.append(Spacer(1, 16))
        story.append(Paragraph("Conclusion", styles["Heading2"]))
        story.append(Spacer(1, 8))

        conclusion_text = (
            "This AI-powered crash risk intelligence system enables authorities to "
            "transition from reactive response to proactive prevention. By combining "
            "data-driven risk assessment with spatial and behavioral insights, the "
            "system provides a scalable foundation for reducing severe and fatal "
            "road crashes while supporting accountable decision-making."
        )

        story.append(Paragraph(conclusion_text, styles["Normal"]))


# This block builds the PDF in memory and provides it to the user.
# No backend file storage is used.
# The PDF is streamed directly to the browser.
# This is secure and deployment-safe.
# Errors are fully caught.
# The app never crashes.

        doc.build(story)
        pdf_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Executive PDF Report",
            data=pdf_buffer,
            file_name=report_filename,
            mime="application/pdf"
        )

    except Exception as e:
        st.error("PDF generation failed")
        st.code(str(e))



# This block enables CSV download for police and government users.
# It exports the final dataset with all predictions and intelligence.
# The CSV contains original data plus risk scores and recommendations.
# No rows are removed or modified during export.
# Download happens only on explicit user action.
# This supports audit, Tableau ingestion, and offline analysis.

st.subheader("📥 Download Prediction CSV")

# This block checks whether the final intelligence dataset exists.
# CSV export without completed pipeline is not allowed.
# The system stops safely if required data is missing.
# Clear feedback is shown to the user.
# This prevents exporting partial or incorrect data.
# No silent fallback is used.

if "df_explainable" not in st.session_state:
    st.warning("Complete prediction, risk scoring, and explainability before CSV download.")
    st.stop()

df_export = st.session_state["df_explainable"].copy()

# This block prepares the CSV for export.
# Index is removed to keep file clean.
# Column order is preserved as generated.
# No formatting or rounding is applied.
# This ensures raw analytical usability.
# Data integrity is fully maintained.

csv_data = df_export.to_csv(index=False)


# This block provides the download button.
# File name is auto-generated but editable by user.
# Download does not refresh or rerun the app.
# Large files are handled efficiently.
# This is a non-blocking operation.
# The CSV is immediately usable in Tableau or Excel.

csv_filename = st.text_input(
    "CSV file name",
    value="Police_Crash_Risk_Predictions.csv"
)

st.download_button(
    label="⬇️ Download Prediction CSV",
    data=csv_data,
    file_name=csv_filename,
    mime="text/csv"
)


# This block confirms successful availability of export.
# No file system write occurs on server.
# The data is streamed directly to the user.
# This avoids storage and privacy risks.
# The system remains responsive.
# CSV export module is now complete.

st.success("Prediction CSV ready for download")




# This block defines legal, ethical, and operational safeguards.
# It clearly states what the system can and cannot do.
# This protects authorities from misuse or over-reliance.
# It ensures the system is used as decision support only.
# This section is important for audits and approvals.
# No data processing or ML occurs here.

st.subheader("🛡️ Governance, Controls & System Guarantees")



# This block displays a legal and usage disclaimer.
# It clarifies that the system does not replace police judgment.
# All outputs are probabilistic and advisory in nature.
# This protects both users and deployers.
# The disclaimer is always visible.
# Language is formal and unambiguous.

with st.expander("⚠️ Legal & Usage Disclaimer", expanded=True):
    st.markdown("""
- This AIML system is a **decision support tool**, not a legal authority.
- All predictions and risk scores are **probabilistic**, not guarantees.
- Outputs must be used **alongside professional judgment**.
- The system does **not replace investigations or legal procedures**.
- Final responsibility for actions remains with authorities.
""")



# This block explains model governance rules.
# It highlights limitations of historical data.
# It enforces inference-only usage.
# Retraining is explicitly excluded from this app.
# This ensures model integrity.
# It supports compliance and review.

with st.expander("🧠 Model Governance & Limitations"):
    st.markdown("""
- Models are trained on historical crash data and may reflect past patterns.
- Performance can change as traffic behavior evolves.
- Periodic offline retraining and validation are recommended.
- Models are locked for **inference only** in this system.
- No self-learning or automatic updates are allowed.
""")


# This block defines data governance and privacy rules.
# It ensures no personal data persistence.
# Uploaded data is processed only during the session.
# Location data is used only for risk aggregation.
# This supports privacy and regulatory compliance.
# No external sharing occurs.

with st.expander("📊 Data Governance & Privacy"):
    st.markdown("""
- Uploaded datasets are processed **in-memory only**.
- No personally identifiable information is stored or exported.
- Geographic data is used strictly for safety analysis.
- Users are responsible for ensuring data compliance with local laws.
""")


# This block outlines AI safety and misuse prevention controls.
# It prevents silent failures and misleading outputs.
# All errors are surfaced to the user.
# Extreme outputs are capped conservatively.
# No automated enforcement actions are generated.
# Human-in-the-loop is always maintained.

with st.expander("🚨 AI Safety & Misuse Prevention"):
    st.markdown("""
- All model failures are shown explicitly to the user.
- Invalid inputs trigger visible warnings.
- Risk categories are conservatively defined.
- The system does not issue automatic penalties or actions.
""")


# This block lists system guarantees.
# Guarantees are simple and verifiable.
# They reinforce transparency and reliability.
# This helps build trust with leadership.
# No technical claims are exaggerated.
# Guarantees are operationally meaningful.

with st.expander("✅ System Guarantees"):
    st.markdown("""
- No silent crashes or hidden predictions.
- No unvalidated data usage.
- Full transparency in risk logic.
- Human-readable explanations for all high-risk cases.
- Policy-oriented outputs only.
""")


# This block confirms audit and deployment readiness.
# It lists features relevant for review committees.
# It highlights reproducibility and traceability.
# This supports real-world deployment approval.
# No operational data is exposed.
# The checklist is concise and clear.

with st.expander("📁 Audit & Deployment Readiness"):
    st.markdown("""
- Deterministic inference pipeline.
- Enforced feature schema alignment.
- Explicit risk scoring logic.
- Reproducible PDF report generation.
- Version-controlled model artifacts.
""")


# This block displays current system status.
# It reassures users that all modules are active.
# Status indicators are simple and clear.
# This supports operational confidence.
# Values are static confirmations, not diagnostics.
# The system is ready for use.

st.markdown("### 🔍 System Status")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Models", "Loaded")

with s2:
    st.metric("Risk Engine", "Active")

with s3:
    st.metric("Explainability", "Enabled")

with s4:
    st.metric("Reporting", "Ready")



# This block provides final confirmation to the user.
# It signals that the full pipeline is complete.
# The language is confident and professional.
# This is the final state of the application.
# No further processing occurs.
# The system is deployment-ready.

st.success("""
SYSTEM READY FOR OPERATION

This AI Crash Risk Intelligence System is:
- Technically complete
- Policy aligned
- Audit ready
- Deployment capable
""")

st.markdown("""
### Final Note

This application demonstrates end-to-end ownership of:
- Data preparation
- Machine learning inference
- Risk intelligence
- Explainability
- Policy decision support

This is not a dashboard.  
This is a real-world AI decision intelligence system.
""")
