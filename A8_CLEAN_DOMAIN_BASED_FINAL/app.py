
import streamlit as st
import pandas as pd
import plotly.express as px
from data_utils import *

st.set_page_config(page_title="A8 Clean Resume Intelligence", layout="wide")

st.title("A8 Resume Screening Intelligence (CLEAN vFINAL)")
st.caption("Domain-based hiring intelligence system")

# ---------------- INPUTS ----------------
st.sidebar.header("Controls")

n = st.sidebar.slider("Synthetic Candidates",200,800,400)

uploaded = st.sidebar.file_uploader("Upload Resume CSV", type=["csv"])

role_list = ROLES["role"].tolist()
selected_role = st.sidebar.selectbox("Select Role (Optional)", ["All"] + role_list)

jd_text = st.text_area("Job Description (Optional - No Classification)", "python sql data analysis communication leadership")

# ---------------- DATA ----------------
resumes = generate(n)

if uploaded:
    try:
        resumes = pd.read_csv(uploaded)
    except:
        pass

role_filter = None if selected_role == "All" else selected_role

df = build(resumes, selected_role=role_filter)

top1 = df[df["rank"] == 1]

# ---------------- JD MATCH ----------------
jd_vector = jd_to_vector(jd_text)

role_scores = []
for _, r in ROLES.iterrows():
    sim = compute_similarity(jd_vector, r[SKILLS])
    role_scores.append({
        "role": r["role"],
        "domain": r["domain"],
        "jd_match_score": sim * 100
    })

role_match_df = pd.DataFrame(role_scores).sort_values("jd_match_score", ascending=False)

best_role = role_match_df.iloc[0]["role"]

# ---------------- TABS ----------------
t1, t2, t3, t4 = st.tabs([
    "Overview",
    "Role Explorer",
    "JD Matching",
    "Explainability"
])

# ---------------- OVERVIEW ----------------
with t1:
    st.metric("Candidates", len(resumes))
    st.metric("Best JD Role Match", best_role)

    st.plotly_chart(px.histogram(df, x="score", nbins=30, title="Score Distribution"))

# ---------------- ROLE EXPLORER ----------------
with t2:
    st.dataframe(df)
    st.plotly_chart(px.bar(df.groupby("role")["score"].mean().reset_index(),
                            x="role", y="score"))

# ---------------- JD MATCHING ----------------
with t3:
    st.subheader("JD → Role Similarity (NO CLASSIFICATION)")

    st.dataframe(role_match_df)

    st.plotly_chart(px.bar(role_match_df, x="role", y="jd_match_score"))

# ---------------- EXPLAINABILITY ----------------
with t4:
    c = st.selectbox("Candidate", resumes["candidate_id"].unique())

    view = df[df["candidate_id"] == c]

    view["explanation"] = view.apply(explain, axis=1)

    st.dataframe(view)

    st.plotly_chart(px.bar(view, x="role", y="score"))
