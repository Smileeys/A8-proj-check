
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROLES = pd.DataFrame([
    {"role":"Data Scientist","technical":90,"programming":90,"analytics":90,"business":70,"communication":70,"leadership":60,"experience":2},
    {"role":"Data Analyst","technical":75,"programming":70,"analytics":85,"business":75,"communication":75,"leadership":50,"experience":1},
    {"role":"Business Analyst","technical":60,"programming":50,"analytics":70,"business":90,"communication":85,"leadership":65,"experience":1},
    {"role":"ML Engineer","technical":95,"programming":95,"analytics":85,"business":60,"communication":60,"leadership":55,"experience":3}
])

SKILLS = ["technical","programming","analytics","business","communication","leadership"]

# ---------------- Synthetic resumes ----------------
def generate(n=500):
    return pd.DataFrame({
        "candidate_id":[f"CAND-{i}" for i in range(n)],
        "experience":np.random.randint(0,8,n),
        "technical":np.random.randint(30,100,n),
        "programming":np.random.randint(30,100,n),
        "analytics":np.random.randint(30,100,n),
        "business":np.random.randint(30,100,n),
        "communication":np.random.randint(30,100,n),
        "leadership":np.random.randint(20,95,n),
        "resume_text":np.random.choice([
            "python sql machine learning data analysis dashboards communication leadership",
            "excel business reporting stakeholder communication excel powerpoint",
            "deep learning neural networks python pytorch ml models",
            "statistics python data cleaning visualization storytelling"
        ], n)
    })

# ---------------- JD parser (simple NLP simulation) ----------------
def parse_job_description(text):
    text = text.lower()
    score = {
        "technical":sum(k in text for k in ["python","ml","machine","data","sql","cloud"]),
        "programming":sum(k in text for k in ["python","java","sql","coding"]),
        "analytics":sum(k in text for k in ["analysis","analytics","statistics","insight"]),
        "business":sum(k in text for k in ["business","stakeholder","finance","strategy"]),
        "communication":sum(k in text for k in ["communication","presentation","report"]),
        "leadership":sum(k in text for k in ["leadership","management","team"])
    }
    return pd.Series(score).clip(0,100)

# ---------------- Resume NLP simulation ----------------
def parse_resume_text(text):
    text = str(text).lower()
    return pd.Series({
        "technical":sum(k in text for k in ["python","ml","data","sql"]),
        "programming":sum(k in text for k in ["python","sql","java"]),
        "analytics":sum(k in text for k in ["analysis","statistics","visualization"]),
        "business":sum(k in text for k in ["business","strategy"]),
        "communication":sum(k in text for k in ["communication","report"]),
        "leadership":sum(k in text for k in ["leadership","team"])
    }).clip(0,100)

# ---------------- Core scoring ----------------
def compute(r,j):
    sim = cosine_similarity(r[SKILLS].values.reshape(1,-1),
                            j[SKILLS].values.reshape(1,-1))[0][0]
    exp = min(r["experience"]/max(j["experience"],1),1)*100
    gap = np.maximum(j[SKILLS].values - r[SKILLS].values,0).mean()
    match = (sim*100*0.6)+(exp*0.25)+((100-gap)*0.15)
    return sim*100,exp,gap,match

def build(resumes, jd=None):
    rows=[]
    for _,r in resumes.iterrows():
        for _,j in ROLES.iterrows():
            sim,exp,gap,match = compute(r,j)
            rows.append({
                "candidate_id":r["candidate_id"],
                "role":j["role"],
                "similarity":sim,
                "experience_score":exp,
                "skill_gap":gap,
                "match_score":match
            })
    df=pd.DataFrame(rows)

    df["suitability"]=0.5*df["match_score"]+0.3*df["similarity"]+0.2*df["experience_score"]
    df["rank"]=df.groupby("candidate_id")["suitability"].rank(ascending=False)

    df["readiness"]=df["suitability"].apply(lambda x:
        "High" if x>80 else "Medium" if x>60 else "Low"
    )
    return df

# ---------------- Explainability ----------------
def explain(row):
    reasons=[]
    if row["similarity"]>70:
        reasons.append("Strong skill similarity with role")
    if row["experience_score"]>60:
        reasons.append("Good experience match")
    if row["skill_gap"]<20:
        reasons.append("Low skill gaps")
    if row["suitability"]>80:
        reasons.append("High overall alignment score")
    return " | ".join(reasons) if reasons else "Moderate alignment profile"
