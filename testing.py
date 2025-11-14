import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import plotly.express as px

# ---------------------------
# Streamlit app: Internship Recommender (Full Rewrite)
# - Uploadable CSVs
# - Apple-inspired UI
# - Fuzzy + Baseline scoring
# ---------------------------

st.set_page_config(page_title="Internship Recommender", layout="wide")

APPLE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600&display=swap');

body {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #f5f5f7;
    color: #1d1d1f;
}

.stApp {
    background: linear-gradient(135deg, #000000 0%, #1c1c1e 100%);
    border-radius: 0px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    padding: 0px;
    margin: 0px;
}

.card {
    background: #1d1d1f;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 10px 15px;
    margin: 5px 0;
}

.stButton button {
    background: #007aff;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    transition: background 0.3s ease;
}

.stButton button:hover {
    background: #0056cc;
}

.stSelectbox, .stTextInput {
    border-radius: 8px;
    border: 0px solid #d1d1d6;
}

h1, h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

h1 {
    white-space: nowrap;
    text-align: center;
}

.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}
</style>
"""

st.markdown(APPLE_CSS, unsafe_allow_html=True)

# ---------------------------
# Utility: sample templates for download
# ---------------------------

@st.cache_data
def sample_students_df():
    df = pd.DataFrame(
        {
            "Student_ID": ["S001", "S002"],
            "Name": ["Alice Tan", "Bob Lee"],
            "Field": ["Software", "Data"],
            "CGPA": [3.6, 2.9],
            "Skill_Match_Percent": [85, 60],
        }
    )
    return df

@st.cache_data
def sample_internships_df():
    df = pd.DataFrame(
        {
            "Internship_ID": ["I001", "I002"],
            "Company": ["Acme Tech", "DataWorks"],
            "Role": ["Backend Intern", "Data Analyst Intern"],
            "Field": ["Software", "Data"],
            "Location": ["Kuala Lumpur", "Penang"],
            "Min_CGPA": [3.0, 2.5],
        }
    )
    return df

# ---------------------------
# CSV Upload / Load
# ---------------------------

st.title("Internship Recommendation System — Group 6")
st.caption("Powered by Fuzzy Logic — Upload your CSVs to get personalized internship matches")

with st.expander("Upload or download sample CSVs"):
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download student CSV template",
            data=sample_students_df().to_csv(index=False),
            file_name="students_template.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download internship CSV template",
            data=sample_internships_df().to_csv(index=False),
            file_name="internships_template.csv",
            mime="text/csv",
        )

st.markdown("---")

uploaded_students = st.file_uploader("Upload Student CSV", type=["csv"], key="students_uploader")
uploaded_internships = st.file_uploader("Upload Internship CSV", type=["csv"], key="internships_uploader")

use_local_files = st.checkbox("Use local CSV files instead (student_data.csv & internship_data.csv)")

@st.cache_data
def load_local_files():
    try:
        students_df = pd.read_csv("student_data.csv")
        internships_df = pd.read_csv("internship_data.csv")
        return students_df, internships_df
    except Exception:
        return None, None


def load_data_from_uploads(stu_file, int_file, fallback_local=False):
    # Priority: uploaded files > local files (if fallback_local True)
    if stu_file is not None and int_file is not None:
        try:
            s = pd.read_csv(stu_file)
            i = pd.read_csv(int_file)
            st.success("CSV files uploaded and read successfully.")
            return s, i
        except Exception as e:
            st.error(f"Error reading uploaded files: {e}")
            return None, None

    if fallback_local:
        s, i = load_local_files()
        if s is not None and i is not None:
            st.info("Loaded local CSV files from project folder.")
            return s, i
        else:
            st.warning("Local CSV files not found or unreadable.")
            return None, None

    st.warning("Please upload BOTH student and internship CSV files (or enable local files).")
    return None, None

students, internships = load_data_from_uploads(uploaded_students, uploaded_internships, fallback_local=use_local_files)

if students is None or internships is None:
    st.stop()

# Basic validation
required_student_cols = {"Student_ID", "Name", "Field", "CGPA", "Skill_Match_Percent"}
required_intern_cols = {"Internship_ID", "Company", "Role", "Field", "Location", "Min_CGPA"}

if not required_student_cols.issubset(set(students.columns)):
    st.error(f"Student CSV is missing columns: {required_student_cols - set(students.columns)}")
    st.stop()

if not required_intern_cols.issubset(set(internships.columns)):
    st.error(f"Internship CSV is missing columns: {required_intern_cols - set(internships.columns)}")
    st.stop()

# Normalize column types
students = students.copy()
internships = internships.copy()

students["CGPA"] = pd.to_numeric(students["CGPA"], errors="coerce").fillna(0.0)
students["Skill_Match_Percent"] = pd.to_numeric(students["Skill_Match_Percent"], errors="coerce").fillna(0.0)
internships["Min_CGPA"] = pd.to_numeric(internships["Min_CGPA"], errors="coerce").fillna(0.0)

# Show small preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Uploaded Data Preview")
colA, colB = st.columns([1, 1])
with colA:
    st.write("**Students (first 5 rows)**")
    st.dataframe(students.head())
with colB:
    st.write("**Internships (first 5 rows)**")
    st.dataframe(internships.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------

def field_alignment(student_field, job_field):
    if pd.isna(student_field) or pd.isna(job_field):
        return 0.0, "Not Match"
    s = str(student_field).strip().lower()
    j = str(job_field).strip().lower()
    if s == j:
        return 1.0, "Match"
    related = {
        "software": {"data", "networks"},
        "data": {"software", "networks"},
        "networks": {"software", "data"},
    }
    if s in related and j in related[s]:
        return 0.5, "Partial Match"
    return 0.0, "Not Match"


def compute_features(student_row, internship_row):
    fa_num, fa_label = field_alignment(student_row["Field"], internship_row["Field"])
    return {
        "cgpa": float(student_row["CGPA"]),
        "skill_match_pct": float(student_row["Skill_Match_Percent"]),
        "field_align_num": float(fa_num),
        "field_align_label": fa_label,
        "min_cgpa": float(internship_row["Min_CGPA"]),
    }

# ---------------------------
# Fuzzy system builder (returns a fresh simulation each time to avoid state issues)
# ---------------------------

def build_fuzzy_simulation():
    cgpa_u = np.arange(0, 4.01, 0.01)
    skill_u = np.arange(0, 101, 1)
    field_u = np.arange(0, 1.01, 0.01)
    rec_u = np.arange(0, 101, 1)

    CGPA = ctrl.Antecedent(cgpa_u, 'CGPA')
    Skill = ctrl.Antecedent(skill_u, 'Skill')
    Field = ctrl.Antecedent(field_u, 'Field')
    Recommendation = ctrl.Consequent(rec_u, 'Recommendation')

    CGPA['low'] = fuzz.trimf(CGPA.universe, [0, 0, 2.5])
    CGPA['medium'] = fuzz.trimf(CGPA.universe, [2.0, 2.75, 3.5])
    CGPA['high'] = fuzz.trimf(CGPA.universe, [3.0, 4.0, 4.0])

    Skill['poor'] = fuzz.trimf(Skill.universe, [0, 0, 50])
    Skill['moderate'] = fuzz.trimf(Skill.universe, [30, 55, 80])
    Skill['excellent'] = fuzz.trimf(Skill.universe, [70, 100, 100])

    Field['notmatch'] = fuzz.trimf(Field.universe, [0, 0, 0.3])
    Field['partial'] = fuzz.trimf(Field.universe, [0.2, 0.5, 0.8])
    Field['match'] = fuzz.trimf(Field.universe, [0.7, 1.0, 1.0])

    Recommendation['weak'] = fuzz.trimf(Recommendation.universe, [0, 0, 50])
    Recommendation['fair'] = fuzz.trimf(Recommendation.universe, [30, 55, 80])
    Recommendation['strong'] = fuzz.trimf(Recommendation.universe, [70, 100, 100])

    # Rules
    rule1 = ctrl.Rule(CGPA['high'] & Skill['excellent'] & Field['match'], Recommendation['strong'])
    rule2 = ctrl.Rule(CGPA['medium'] & Skill['excellent'] & Field['match'], Recommendation['strong'])
    rule3 = ctrl.Rule(CGPA['high'] & Skill['moderate'] & Field['match'], Recommendation['strong'])
    rule4 = ctrl.Rule(CGPA['medium'] & Skill['moderate'] & Field['partial'], Recommendation['fair'])
    rule5 = ctrl.Rule(CGPA['low'] & Skill['poor'], Recommendation['weak'])
    rule6 = ctrl.Rule(Field['notmatch'], Recommendation['weak'])
    rule7 = ctrl.Rule(Skill['excellent'] & Field['partial'], Recommendation['fair'])
    rule8 = ctrl.Rule(CGPA['high'] & Field['partial'], Recommendation['fair'])
    rule9 = ctrl.Rule(Skill['moderate'] & Field['match'], Recommendation['fair'])

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    sim = ctrl.ControlSystemSimulation(system)
    return sim


def fuzzy_score(features):
    sim = build_fuzzy_simulation()
    sim.input['CGPA'] = features['cgpa']
    sim.input['Skill'] = features['skill_match_pct']
    sim.input['Field'] = features['field_align_num']
    try:
        sim.compute()
        return float(sim.output['Recommendation'])
    except Exception:
        # When fuzzy fails for edge inputs, return 0
        return 0.0

# ---------------------------
# Baseline scoring (explainable weighted formula)
# ---------------------------

def baseline_score(features):
    # cgpa_fit: 1 if meets minimum, otherwise proportionally smaller
    if features['min_cgpa'] <= 0:
        cgpa_fit_val = 1.0 if features['cgpa'] > 0 else 0.0
    else:
        cgpa_fit_val = 1.0 if features['cgpa'] >= features['min_cgpa'] else max(0.0, (features['cgpa'] / max(1e-6, features['min_cgpa'])) * 0.5)

    skill01 = features['skill_match_pct'] / 100.0
    field01 = features['field_align_num']

    score01 = 0.5 * cgpa_fit_val + 0.3 * skill01 + 0.2 * field01
    return float(score01 * 100.0)

# ---------------------------
# Ranking and UI Controls
# ---------------------------

def rank_for_student(student_id, top_n=5):
    srows = students[students['Student_ID'].astype(str) == str(student_id)]
    if srows.empty:
        # also allow selection by name
        srows = students[students['Name'].astype(str).str.contains(str(student_id))]
        if srows.empty:
            return pd.DataFrame()
    srow = srows.iloc[0]

    rows = []
    for _, irow in internships.iterrows():
        feats = compute_features(srow, irow)
        fz = fuzzy_score(feats)
        bl = baseline_score(feats)
        rows.append(
            {
                "Internship_ID": irow["Internship_ID"],
                "Company": irow["Company"],
                "Role": irow["Role"],
                "Field": irow["Field"],
                "Location": irow["Location"],
                "Fuzzy_Score": round(fz, 2),
                "Baseline_Score": round(bl, 2),
                "Field_Align": feats["field_align_label"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Fuzzy_Score", "Baseline_Score"], ascending=[False, False]).reset_index(drop=True)
    return out.head(top_n)

# Controls: Choose student & number of results
student_options = (students['Student_ID'].astype(str) + " - " + students['Name'].astype(str)).tolist()
selected_student = st.selectbox("Choose a student:", student_options)
student_id = selected_student.split(" - ")[0]

top_n = st.slider("Number of top recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    results = rank_for_student(student_id, top_n)
    if results.empty:
        st.error("No matches found for the selected student. Check Student ID or uploaded data.")
    else:
        st.success(f"Top {len(results)} Internship Matches for {student_id}")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Recommendations")
        st.dataframe(results[["Company", "Role", "Fuzzy_Score", "Baseline_Score", "Field_Align"]])
        st.markdown('</div>', unsafe_allow_html=True)

        # Fuzzy chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Fuzzy Scores Chart")
        fig_fuzzy = px.bar(results, x='Role', y='Fuzzy_Score', color='Company', title=f'Fuzzy Scores for {student_id}')
        fig_fuzzy.update_layout(font_family="SF Pro Display", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_fuzzy.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_fuzzy, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Baseline chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Baseline Scores Chart")
        fig_baseline = px.bar(results, x='Role', y='Baseline_Score', color='Company', title=f'Baseline Scores for {student_id}')
        fig_baseline.update_layout(font_family="SF Pro Display", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_baseline.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_baseline, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

st.caption("Made with ❤️ — ask me if you want additional features: sliders for rule weights, export results, or add more fuzzy rules.")
