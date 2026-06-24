
import streamlit as st
import pandas as pd
import plotly.express as px
from data_utils import *

st.set_page_config(page_title="A8 Hiring Intelligence Studio", layout="wide")

st.title("A8 Resume Screening Intelligence (v3 - NLP + Explainability)")

# ---------------- Inputs ----------------
st.sidebar.header("Inputs")

n = st.sidebar.slider("Synthetic Candidates",200,800,400)

jd_text = st.text_area("Job Description Input (NLP Parser)", 
"Python ML Data Analysis Communication Leadership")

uploaded = st.sidebar.file_uploader("Upload Resume CSV",type=["csv"])

# ---------------- Data ----------------
resumes = generate(n)

if uploaded:
    try:
        up = pd.read_csv(uploaded)
        resumes = up
    except:
        pass

df = build(resumes)

# ---------------- NLP JD vector ----------------
jd_vector = parse_job_description(jd_text)

df["jd_alignment"] = (
    df["match_score"]*0.6 +
    df["similarity"]*0.4
)

df["explanation"] = df.apply(explain, axis=1)

top1 = df[df["rank"]==1]

# ---------------- Tabs ----------------
t1,t2,t3,t4,t5 = st.tabs([
"Overview","Resume NLP","Job Matching","Explainability","Shortlist"
])

# ---------------- Overview ----------------
with t1:
    st.metric("Candidates", len(resumes))
    st.metric("Avg Suitability", round(df["suitability"].mean(),2))

    st.plotly_chart(px.histogram(df,x="suitability",nbins=30))

# ---------------- Resume NLP ----------------
with t2:
    st.subheader("Resume Text Parser (Simulated NLP)")
    if "resume_text" in resumes.columns:
        st.dataframe(resumes[["candidate_id","resume_text"]].head())
    else:
        st.warning("No resume text found")

# ---------------- Job Matching ----------------
with t3:
    st.subheader("JD Alignment Score")
    st.write(jd_vector)

# ---------------- Explainability ----------------
with t4:
    cand = st.selectbox("Candidate", resumes["candidate_id"].unique())
    view = df[df["candidate_id"]==cand]
    st.dataframe(view[["role","suitability","explanation"]])
    st.plotly_chart(px.bar(view,x="role",y="suitability"))

# ---------------- Shortlist ----------------
with t5:
    st.subheader("Top Candidates")
    st.dataframe(df[df["rank"]==1].sort_values("suitability",ascending=False).head(20))
