
import streamlit as st
import plotly.express as px
import pandas as pd
from data_utils import *

st.set_page_config(page_title="A8 Hiring Intelligence Studio", layout="wide")

st.title("A8 — Resume Screening Intelligence Platform (FINAL)")
st.caption("Fully aligned Hiring Intelligence System")

# ---------------------------
# Data
# ---------------------------
n = st.sidebar.slider("Candidates",200,1000,600)

resumes = generate_resumes(n)

df = build_matches(resumes)
df = add_intelligence(df)

top1 = top_candidates(df)

# ---------------------------
# Tabs
# ---------------------------
t1,t2,t3,t4,t5 = st.tabs([
"Overview","Candidate","Roles","Skill Gaps","Shortlist"
])

# ---------------------------
# Overview
# ---------------------------
with t1:
    st.metric("Candidates", len(resumes))
    st.metric("Avg Suitability", round(df["suitability"].mean(),2))

    fig = px.histogram(df,x="suitability",nbins=30,title="Suitability Distribution")
    st.plotly_chart(fig,use_container_width=True)

# ---------------------------
# Candidate
# ---------------------------
with t2:
    c = st.selectbox("Candidate", resumes["candidate_id"].unique())

    view = df[df["candidate_id"]==c].sort_values("suitability",ascending=False)

    st.dataframe(view)

    fig = px.bar(view,x="role",y="suitability",title="Role Fit")
    st.plotly_chart(fig,use_container_width=True)

# ---------------------------
# Roles
# ---------------------------
with t3:
    rs = role_summary(df).reset_index()
    st.dataframe(rs)

    fig = px.bar(rs,x="role",y="avg_suitability",title="Role Intelligence")
    st.plotly_chart(fig,use_container_width=True)

# ---------------------------
# Skill Gaps
# ---------------------------
with t4:
    gap = df.groupby("role")["skill_gap"].mean().reset_index()

    fig = px.bar(gap,x="role",y="skill_gap",title="Skill Gap by Role")
    st.plotly_chart(fig,use_container_width=True)

# ---------------------------
# Shortlist
# ---------------------------
with t5:
    st.subheader("Top Candidates per Role")

    st.dataframe(top_per_role(df))
