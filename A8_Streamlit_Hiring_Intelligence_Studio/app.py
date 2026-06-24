
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="A8 Hiring Intelligence Studio", layout="wide")

st.title("A8 — Resume Screening Intelligence Platform")
st.caption("Hiring Intelligence Studio: Resume matching, suitability scoring, skill gaps, and candidate ranking")

# ---------------------------
# Job Roles
# ---------------------------
roles = pd.DataFrame([
    {"role":"Data Scientist","technical":90,"programming":90,"analytics":90,"business":70,"communication":70,"leadership":60,"experience":2},
    {"role":"Data Analyst","technical":75,"programming":70,"analytics":85,"business":75,"communication":75,"leadership":50,"experience":1},
    {"role":"Business Analyst","technical":60,"programming":50,"analytics":70,"business":90,"communication":85,"leadership":65,"experience":1},
    {"role":"ML Engineer","technical":95,"programming":95,"analytics":85,"business":60,"communication":60,"leadership":55,"experience":3}
])

skills = ["technical","programming","analytics","business","communication","leadership"]

# ---------------------------
# Data
# ---------------------------
st.sidebar.header("Data Input")

uploaded = st.sidebar.file_uploader("Upload Resume CSV", type=["csv"])

n = st.sidebar.slider("Synthetic candidates", 200, 1000, 600)

if uploaded:
    resumes = pd.read_csv(uploaded)
else:
    resumes = pd.DataFrame({
        "candidate_id":[f"CAND-{i}" for i in range(n)],
        "experience":np.random.randint(0,8,n),
        "technical":np.random.randint(30,100,n),
        "programming":np.random.randint(30,100,n),
        "analytics":np.random.randint(30,100,n),
        "business":np.random.randint(30,100,n),
        "communication":np.random.randint(30,100,n),
        "leadership":np.random.randint(20,95,n)
    })

# ---------------------------
# Scoring
# ---------------------------
def compute(r,j):
    sim = cosine_similarity(r[skills].values.reshape(1,-1),
                            j[skills].values.reshape(1,-1))[0][0]

    exp = min(r["experience"]/max(j["experience"],1),1)*100

    gap = np.maximum(j[skills].values - r[skills].values,0).mean()

    match = (sim*100*0.6) + (exp*0.25) + ((100-gap)*0.15)

    return sim*100, exp, gap, match

rows=[]

for _,r in resumes.iterrows():
    for _,j in roles.iterrows():
        sim,exp,gap,match = compute(r,j)

        rows.append({
            "candidate_id":r["candidate_id"],
            "role":j["role"],
            "similarity":sim,
            "experience_score":exp,
            "skill_gap":gap,
            "match_score":match
        })

df = pd.DataFrame(rows)

df["suitability"] = (
    0.5*df["match_score"] +
    0.3*df["similarity"] +
    0.2*df["experience_score"]
)

df["rank"] = df.groupby("candidate_id")["suitability"].rank(ascending=False)

top1 = df[df["rank"]==1]

def readiness(x):
    if x>80: return "High"
    elif x>60: return "Medium"
    else: return "Low"

df["readiness"] = df["suitability"].apply(readiness)

# ---------------------------
# Tabs
# ---------------------------
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Hiring Workspace",
    "Candidate Analysis",
    "Role Intelligence",
    "Skill Gaps",
    "Shortlisting"
])

# ---------------------------
# TAB 1
# ---------------------------
with tab1:
    st.subheader("Overview Dashboard")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Candidates", len(resumes))
    c2.metric("Roles", len(roles))
    c3.metric("Avg Suitability", round(df["suitability"].mean(),2))
    c4.metric("High Readiness %", round((df["readiness"]=="High").mean()*100,2))

    fig = px.histogram(df, x="suitability", nbins=30, title="Suitability Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 2
# ---------------------------
with tab2:
    st.subheader("Candidate Deep Dive")

    cand = st.selectbox("Select Candidate", resumes["candidate_id"].unique())

    view = df[df["candidate_id"]==cand].sort_values("suitability",ascending=False)

    st.dataframe(view)

    fig = px.bar(view, x="role", y="suitability", title="Role Fit")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 3
# ---------------------------
with tab3:
    st.subheader("Role Intelligence")

    role_summary = df.groupby("role").agg(
        avg_match=("match_score","mean"),
        avg_suitability=("suitability","mean"),
        avg_gap=("skill_gap","mean")
    ).reset_index()

    st.dataframe(role_summary)

    fig = px.bar(role_summary, x="role", y="avg_suitability", title="Role Suitability")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 4
# ---------------------------
with tab4:
    st.subheader("Skill Gap Analysis")

    gap = df.groupby("role")["skill_gap"].mean().reset_index()

    fig = px.bar(gap, x="role", y="skill_gap", title="Average Skill Gap")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 5
# ---------------------------
with tab5:
    st.subheader("Top Candidate Shortlisting")

    st.dataframe(top1.sort_values("suitability",ascending=False).head(20))

