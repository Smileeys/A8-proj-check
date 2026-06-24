
import streamlit as st
import pandas as pd
import plotly.express as px
from data_utils import *

st.set_page_config(page_title="A8 Hiring Intelligence v4", layout="wide")

st.title("A8 Resume Screening Intelligence (v4 - FIXED JD ROLE DETECTION)")

# ---------------- Inputs ----------------
st.sidebar.header("Controls")

n = st.sidebar.slider("Candidates",200,800,400)
uploaded = st.sidebar.file_uploader("Upload Resume CSV",type=["csv"])

jd = st.text_area("Job Description", "Python ML model data analysis neural networks deployment")

# ---------------- Role Detection (FIXED CORE) ----------------
role_detected, confidence, scores = detect_role(jd)

st.success(f"Detected Role: {role_detected} | Confidence: {round(confidence*100,2)}%")

st.json(scores)

# ---------------- Data ----------------
resumes = generate(n)

if uploaded:
    try:
        resumes = pd.read_csv(uploaded)
    except:
        pass

df = build(resumes, selected_role=role_detected)

top1 = df[df["rank"]==1]

# ---------------- Tabs ----------------
t1,t2,t3,t4 = st.tabs([
"Overview","Role View","Explainability","Shortlist"
])

with t1:
    st.metric("Detected Role", role_detected)
    st.metric("Avg Suitability", round(df["suitability"].mean(),2))

    st.plotly_chart(px.histogram(df,x="suitability",nbins=30))

with t2:
    st.dataframe(df)
    st.plotly_chart(px.bar(df,x="candidate_id",y="suitability"))

with t3:
    c = st.selectbox("Candidate", df["candidate_id"].unique())
    view = df[df["candidate_id"]==c]
    view["explanation"] = view.apply(lambda r: explain(r, role_detected), axis=1)
    st.dataframe(view)

with t4:
    st.dataframe(top1.sort_values("suitability",ascending=False).head(20))
