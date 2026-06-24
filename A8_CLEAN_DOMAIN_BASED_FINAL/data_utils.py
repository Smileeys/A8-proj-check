
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 3 DOMAIN ROLE SYSTEM (CLEAN)
# -------------------------------

ROLES = pd.DataFrame([
    # Engineering
    {"role":"Data Scientist","domain":"Engineering","technical":90,"programming":90,"analytics":90,"business":70,"communication":70,"leadership":60},
    {"role":"ML Engineer","domain":"Engineering","technical":95,"programming":95,"analytics":85,"business":60,"communication":60,"leadership":55},
    {"role":"Software Engineer","domain":"Engineering","technical":90,"programming":95,"analytics":70,"business":50,"communication":60,"leadership":50},

    # Analytics
    {"role":"Data Analyst","domain":"Analytics","technical":75,"programming":70,"analytics":85,"business":75,"communication":75,"leadership":50},
    {"role":"Business Analyst","domain":"Analytics","technical":60,"programming":50,"analytics":70,"business":90,"communication":85,"leadership":65},
    {"role":"Product Analyst","domain":"Analytics","technical":70,"programming":60,"analytics":80,"business":80,"communication":75,"leadership":60},

    # L&D / Management
    {"role":"Project Manager","domain":"L&D-Mgmt","technical":50,"programming":40,"analytics":60,"business":90,"communication":90,"leadership":90},
    {"role":"Operations Manager","domain":"L&D-Mgmt","technical":55,"programming":40,"analytics":65,"business":85,"communication":85,"leadership":85},
    {"role":"L&D Specialist","domain":"L&D-Mgmt","technical":40,"programming":30,"analytics":60,"business":80,"communication":95,"leadership":80},
    {"role":"HR Analyst","domain":"L&D-Mgmt","technical":50,"programming":40,"analytics":70,"business":80,"communication":85,"leadership":75},
])

SKILLS = ["technical","programming","analytics","business","communication","leadership"]

# -------------------------------
# DATA
# -------------------------------

def generate(n=500):
    np.random.seed(42)
    return pd.DataFrame({
        "candidate_id":[f"CAND-{i}" for i in range(n)],
        "experience":np.random.randint(0,8,n),
        "technical":np.random.randint(30,100,n),
        "programming":np.random.randint(30,100,n),
        "analytics":np.random.randint(30,100,n),
        "business":np.random.randint(30,100,n),
        "communication":np.random.randint(30,100,n),
        "leadership":np.random.randint(20,95,n),
    })

# -------------------------------
# JD VECTOR (NO CLASSIFICATION)
# -------------------------------

def jd_to_vector(jd_text):
    jd = jd_text.lower()

    return pd.Series({
        "technical":sum(k in jd for k in ["python","ml","sql","data","model","engineering"]),
        "programming":sum(k in jd for k in ["python","java","sql","coding"]),
        "analytics":sum(k in jd for k in ["analysis","analytics","statistics","insight"]),
        "business":sum(k in jd for k in ["business","strategy","stakeholder","finance"]),
        "communication":sum(k in jd for k in ["communication","presentation","report"]),
        "leadership":sum(k in jd for k in ["leadership","management","team"])
    }).clip(0,100)

# -------------------------------
# SCORING ENGINE
# -------------------------------

def compute_similarity(a, b):
    return cosine_similarity(a.values.reshape(1,-1), b.values.reshape(1,-1))[0][0]

def build(resumes, selected_role=None):

    rows = []

    for _, r in resumes.iterrows():
        for _, j in ROLES.iterrows():

            if selected_role and j["role"] != selected_role:
                continue

            sim = compute_similarity(r[SKILLS], j[SKILLS])

            gap = np.maximum(j[SKILLS].values - r[SKILLS].values, 0).mean()

            score = (sim*100*0.7) + ((100-gap)*0.3)

            rows.append({
                "candidate_id": r["candidate_id"],
                "role": j["role"],
                "domain": j["domain"],
                "similarity": sim*100,
                "skill_gap": gap,
                "score": score
            })

    df = pd.DataFrame(rows)

    df["rank"] = df.groupby("candidate_id")["score"].rank(ascending=False)

    return df

# -------------------------------
# EXPLAINABILITY (SIMPLE)
# -------------------------------

def explain(row):
    reasons = []

    if row["similarity"] > 70:
        reasons.append("Strong skill alignment")
    if row["skill_gap"] < 20:
        reasons.append("Low skill gap")
    if row["score"] > 80:
        reasons.append("High overall suitability")

    return " | ".join(reasons) if reasons else "Moderate match"
