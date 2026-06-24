
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Roles (same as final notebook)
# -----------------------------
ROLES = pd.DataFrame([
    {"role":"Data Scientist","technical":90,"programming":90,"analytics":90,"business":70,"communication":70,"leadership":60,"experience":2},
    {"role":"Data Analyst","technical":75,"programming":70,"analytics":85,"business":75,"communication":75,"leadership":50,"experience":1},
    {"role":"Business Analyst","technical":60,"programming":50,"analytics":70,"business":90,"communication":85,"leadership":65,"experience":1},
    {"role":"ML Engineer","technical":95,"programming":95,"analytics":85,"business":60,"communication":60,"leadership":55,"experience":3}
])

SKILLS = ["technical","programming","analytics","business","communication","leadership"]

# -----------------------------
# Synthetic Data
# -----------------------------
def generate_resumes(n=600, seed=42):
    np.random.seed(seed)
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

# -----------------------------
# Core Scoring (MATCHES NOTEBOOK)
# -----------------------------
def compute_scores(r, j):

    sim = cosine_similarity(
        r[SKILLS].values.reshape(1,-1),
        j[SKILLS].values.reshape(1,-1)
    )[0][0]

    exp = min(r["experience"]/max(j["experience"],1),1)*100

    gap = np.maximum(j[SKILLS].values - r[SKILLS].values,0).mean()

    match = (sim*100*0.6) + (exp*0.25) + ((100-gap)*0.15)

    return sim*100, exp, gap, match

# -----------------------------
# Build Matching Table
# -----------------------------
def build_matches(resumes):

    rows = []

    for _,r in resumes.iterrows():
        for _,j in ROLES.iterrows():

            sim,exp,gap,match = compute_scores(r,j)

            rows.append({
                "candidate_id":r["candidate_id"],
                "role":j["role"],
                "similarity":sim,
                "experience_score":exp,
                "skill_gap":gap,
                "match_score":match,

                # per-skill gaps (correct alignment)
                "gap_technical": max(j["technical"] - r["technical"],0),
                "gap_programming": max(j["programming"] - r["programming"],0),
                "gap_analytics": max(j["analytics"] - r["analytics"],0),
                "gap_business": max(j["business"] - r["business"],0),
                "gap_communication": max(j["communication"] - r["communication"],0),
                "gap_leadership": max(j["leadership"] - r["leadership"],0),
            })

    return pd.DataFrame(rows)

# -----------------------------
# Intelligence Layer
# -----------------------------
def add_intelligence(df):

    df = df.copy()

    df["suitability"] = (
        0.5*df["match_score"] +
        0.3*df["similarity"] +
        0.2*df["experience_score"]
    )

    df["rank"] = df.groupby("candidate_id")["suitability"].rank(ascending=False)

    def readiness(x):
        if x>80: return "High"
        elif x>60: return "Medium"
        else: return "Low"

    df["readiness"] = df["suitability"].apply(readiness)

    return df

# -----------------------------
# Role Insights
# -----------------------------
def role_summary(df):
    return df.groupby("role").agg(
        avg_match=("match_score","mean"),
        avg_suitability=("suitability","mean"),
        avg_gap=("skill_gap","mean"),
        candidates=("candidate_id","count")
    ).sort_values("avg_suitability",ascending=False)

# -----------------------------
# Shortlisting
# -----------------------------
def top_candidates(df):
    return df[df["rank"]==1]

def top_per_role(df):
    return df.sort_values("suitability",ascending=False).groupby("role").head(5)
