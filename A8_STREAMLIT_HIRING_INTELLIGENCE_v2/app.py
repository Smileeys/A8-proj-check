
import streamlit as st
import pandas as pd
import plotly.express as px
from data_utils import *

st.set_page_config(page_title="A8 Hiring Intelligence Studio", layout="wide")

st.title("A8 — Resume Screening Intelligence Platform (v2 FIXED)")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Controls")

n = st.sidebar.slider("Synthetic Candidates",200,1000,600)
uploaded = st.sidebar.file_uploader("Upload Resume CSV", type=["csv"])

role_filter = st.sidebar.multiselect("Filter Roles", [r for r in ROLES["role"]], default=list(ROLES["role"]))
readiness_filter = st.sidebar.multiselect("Readiness", ["High","Medium","Low"], default=["High","Medium","Low"])
min_score = st.sidebar.slider("Min Suitability Score",0,100,0)

# ---------------- Data ----------------
if uploaded:
    resumes = load_csv(uploaded)
else:
    resumes = generate(n)

df = build(resumes)

df = df[df["role"].isin(role_filter)]
df = df[df["readiness"].isin(readiness_filter)]
df = df[df["suitability"]>=min_score]

top1 = df[df["rank"]==1]

# ---------------- Tabs ----------------
t1,t2,t3,t4,t5 = st.tabs(["Overview","Candidates","Roles","Gaps","Shortlist"])

with t1:
    st.metric("Candidates", len(resumes))
    st.metric("Avg Suitability", round(df["suitability"].mean(),2))

    fig = px.histogram(df,x="suitability",nbins=30)
    st.plotly_chart(fig,use_container_width=True)

with t2:
    c = st.selectbox("Candidate", resumes["candidate_id"].unique())
    view = df[df["candidate_id"]==c]
    st.dataframe(view)
    st.plotly_chart(px.bar(view,x="role",y="suitability"))

with t3:
    st.dataframe(role_summary(df))
    st.plotly_chart(px.bar(role_summary(df),x="role",y="avg_suitability"))

with t4:
    st.dataframe(df.groupby("role")["skill_gap"].mean().reset_index())

with t5:
    st.dataframe(top_per_role(df))
