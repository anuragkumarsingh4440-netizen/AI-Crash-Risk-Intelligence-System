# 🚨 AIML Based Crash Risk Intelligence System  (CAPSTONE)
### End-to-End Data Science, Risk Modeling & Decision Intelligence Platform

🔗 **Live Application (Streamlit):** <https://crash-risk-system.streamlit.app/>  
📊 **Live Tableau Dashboard:** <https://shorturl.at/pbxhj>  
📹 **System Demo Video:** <https://tinyurl.com/35jk7esr>  
📧 **Contact:** anuragkumarsingh4440@gmail.com  
##  **Data Source - https://catalog.data.gov/dataset/crash-reporting-drivers-data**

---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cb35d870-a704-42c4-9af3-e0e1dd7315c0" />
---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e0e190e7-30f9-4a84-b1f1-56123e5c2fca" />
---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/066088e4-7891-4754-b5af-b009ccf130c0" />
---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7668ceae-6c76-4010-8604-90b1aa13f76a" />
---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ccfb0b78-163d-447e-bc48-618a8f2bcc2e" />
---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/972c244a-4813-41fc-96e8-8c1dade7faf2" />

## 👤 Ownership, Role & Development Approach

This project is **fully designed, owned, and led by me as a Data Scientist**.  
I am responsible for:

- Problem formulation
- Data cleaning strategy
- Statistical & probabilistic analysis
- Feature engineering
- Model selection & validation
- Risk scoring logic
- Explainability framework
- Business & policy alignment

### Application Layer Disclosure (Transparent & Professional)

The **Streamlit application (`app.py`) was developed with the help of ChatGPT as an engineering agent**.

This decision was intentional:
- My primary expertise is **data science, modeling, and analytics**
- The application layer is **large-scale**, multi-model, deployment-oriented
- AI was used as an **accelerator**, similar to collaborating with a platform engineer

👉 **All analytical logic, modeling decisions, risk definitions, and outputs are entirely mine.**
<img width="445" height="478" alt="image" src="https://github.com/user-attachments/assets/402ae76f-19d1-4629-91d7-539cf5c76125" />
<img width="743" height="487" alt="image" src="https://github.com/user-attachments/assets/26ffd1dc-c748-4e4e-b0af-0a2043a87d28" />
<img width="509" height="387" alt="image" src="https://github.com/user-attachments/assets/43da16ab-3de0-43fd-ae5b-40664dab6c4d" />




This reflects **real-world industry practice**, not dependency.

---

## 🎯 Problem Statement

Road crash data exists in abundance, but authorities struggle with:

- Reactive accident response
- Manual hotspot identification
- No behavioral attribution
- No severity forecasting
- No explainability
- Disconnected analysis tools

**Goal:**  
Transform raw crash records into **proactive, explainable, operational intelligence** that helps authorities **prevent crashes before they happen**.

---

## 🧠 What This System Delivers

- Multi-model crash risk prediction
- Injury & damage severity estimation
- Driver fault & distraction analysis
- Spatial hotspot intelligence
- Composite police risk scoring
- Explainable AI reasoning
- Future high-risk location forecasting
- Executive-ready PDF & CSV outputs

This is **not a dashboard**.  
This is a **decision intelligence system**.

---

## 🧱 Complete Data Science Workflow (Strictly Followed)

Below is the **exact end-to-end workflow** followed **before any model training**.

---

### 1. Data Ingestion & Validation
- Safe CSV loading
- Schema enforcement
- Column consistency checks
- No silent failures

---

### 2. Initial Data Understanding
- Structural analysis (`shape`, `info`)
- Feature type assessment
- Domain sanity checks

---

### 3. Duplicate & Record Integrity Analysis
- Row-level duplicates
- Logical duplicates
- Temporal duplication patterns

---

### 4. Anomaly Detection & Data Sanitization
Instead of naïve cleaning, the following were handled:

- Misclassified categories
- Junk tokens (`?`, `--`, `/`, mixed labels)
- Inconsistent categorical semantics

All anomalies were **converted to true null states**, not forced values.

---

### 5. Advanced Missing Value Strategy (NO Mean/Median/Mode Bias)

❌ Mean / Median / Mode were **NOT** the primary strategy.

✅ Instead, the following were used:

- **Probabilistic frequency-based imputation**
- **Context-aware logical imputation**
- **Temporal forward/backward reasoning**
- **Cross-feature dependency filling**
- **NLP-inspired normalization for text categories**
- Distribution-preserving fills (no distortion)

Rules:
- Columns with >40% nulls → dropped
- Rows dropped only if <5% total loss

---

### 6. Outlier Handling (Conservative & Risk-Safe)
- IQR & Z-score used for detection
- Outliers removed **only if <5%**
- Extreme values preserved when they represented **real crash risk**

---

### 7. Feature Engineering
- Temporal extraction (year, month, week, hour, weekday)
- Behavioral indicators
- Spatial readiness for clustering
- Zero target leakage

---

### 8. Statistical Understanding
- 5-point summary
- Distribution shape analysis
- Skewness & kurtosis interpretation
- Data behavior understood **before modeling**

---

### 9. Deep Exploratory Data Analysis

#### Univariate
- Numeric: distribution, tails, density
- Categorical: dominance, rarity, imbalance

#### Bivariate
- Numeric–Numeric correlations
- Categorical–Numeric impact analysis
- Categorical–Categorical interaction patterns

#### Multivariate
- Heatmaps
- Interaction plots
- Risk pattern convergence

---

### 10. Inferential Statistics
- Hypothesis-driven checks
- Pattern validation
- Insights documented **before training**

---

### 11. Encoding Strategy (Model-Specific)
- Ordinal encoding (true order only)
- Label encoding (controlled)
- One-hot (N-1 rule)
- Frequency encoding for high-cardinality features

---

### 12. Transformation & Scaling
- Applied **only when statistically justified**
- Box-Cox / Yeo-Johnson
- Robust, Standard, MinMax scaling aligned with models

---

## 🤖 Models Used (Final Production Models)

| Model | Objective | Algorithm |
|------|----------|----------|
Injury Severity | Predict life impact | CatBoost (Multiclass) |
Crash Risk Score | Probability of severe crash | CatBoost (Probabilistic) |
Driver At Fault | Behavioral culpability | CatBoost (Binary) |
Vehicle Damage Extent | Impact intensity | XGBoost (3-Class) |
Driver Distraction Cause | Behavioral cause | CatBoost (Multiclass) |
Crash Hotspots | Spatial clustering | KMeans |

### Why Tree-Based Models?
- Tabular dominance
- Handles non-linearity
- Robust to noise
- Interpretable
- Industry-proven for risk systems

---

## 🚨 Unified Police Risk Scoring

A composite **POLICE_RISK_SCORE** is generated using:

- Injury severity weight
- Damage severity weight
- Driver fault weight
- Crash risk probability bucket

This produces:
- LOW / MEDIUM / HIGH / CRITICAL categories
- Ranked incidents
- Action prioritization

---

## 🧠 Explainable AI Layer

For every high-risk case, the system generates:

- Why this case is risky
- Which factors contributed
- What action is recommended

Human language.  
No ML jargon.  
No black box.

---

## 🗺️ Spatial Intelligence

- Interactive dark-theme maps
- Risk-colored points
- Clean, readable popups
- Hotspot cluster summaries
- Coordinate-level clarity

---

## 🔮 Next-Month Risk Estimation

Based on:
- Recent crash density
- Severity dominance
- Risk score trends

Authorities receive a **future risk location list** for preventive action.

---

## 📊 Visualization Strategy

### Tableau
- Instant descriptive analysis
- Historical pattern exploration
- Management-friendly views

### Streamlit Application
- Predictive intelligence
- Risk scoring
- Explainability
- Downloads (CSV & PDF)
- Deployment-ready system

---

## 📦 Outputs

- **Human-readable CSV**
- **Executive PDF report**
- **Interactive risk maps**
- **Policy-ready summaries**

---

## 🛡️ Governance & Ethics

- Decision support only
- No automated enforcement
- Transparent logic
- Audit-ready
- No personal data storage

---

## 🧠 Final Note

This project demonstrates **full data science **, not isolated modeling.

Every decision — from anomaly handling to deployment — was made with:
- Real-world constraints
- Safety implications
- Interpretability
- Operational usability

📧 **Contact:** anuragkumarsingh4440@gmail.com
