"""Generate interim-report EDA figures from the existing NHANES ML project.
Place in nhanes_ml_phase/scripts/ and run: python scripts/13_interim_eda_figures.py
Required: data/processed/nhanes_adults.parquet created by the existing merge workflow.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'nhanes_adults.parquet'
FIG = ROOT / 'reports' / 'figures'
OUT = ROOT / 'reports' / 'eda'
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
if not SOURCE.exists():
    raise FileNotFoundError(f'{SOURCE} not found. Run the NHANES download/merge pipeline first.')
df = pd.read_parquet(SOURCE)
required = ['phq9_total', 'elevated_symptoms', 'SLD012', 'sedentary_hours', 'RIDAGEYR',
            'INDFMPIR', 'BMXBMI', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'SMQ020']
missing = [c for c in required if c not in df.columns]
if missing: raise ValueError(f'Missing required columns: {missing}')
phq = pd.to_numeric(df.phq9_total, errors='coerce')
valid = phq.between(0,27)
a = df.loc[valid].copy()
a['phq9_total'] = phq[valid]
a['elevated_symptoms'] = (a.phq9_total >= 10).astype(int)
if len(a)==0:raise ValueError('No valid PHQ-9 scores')
print(f'Valid PHQ-9: {len(a):,}; positive: {a.elevated_symptoms.sum():,} ({a.elevated_symptoms.mean():.2%})')

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches='tight')
    plt.close()
    print('Wrote', FIG/name)

# Figure 1: PHQ-9 distribution
plt.figure(figsize=(8,4.8))
plt.hist(a.phq9_total, bins=np.arange(-.5,28.5,1), edgecolor='white')
plt.axvline(9.5, linestyle='--', color='black', label='Threshold: PHQ-9 ≥10')
plt.xlabel('PHQ-9 total score (0–27)');plt.ylabel('Participants (unweighted)')
plt.title('Figure 1. Distribution of PHQ-9 screening scores');plt.legend();save('eda_phq9_distribution.png')

# Figure 2: Predictor missingness
features = ['RIDAGEYR','INDFMPIR','SLD012','sedentary_hours','BMXBMI','RIAGENDR','RIDRETH3','DMDEDUC2','SMQ020']
valid_cat={'RIAGENDR':[1,2],'RIDRETH3':[1,2,3,4,6,7],'DMDEDUC2':[1,2,3,4,5],'SMQ020':[1,2]}
ranges={'RIDAGEYR':(20,120),'INDFMPIR':(0,5),'SLD012':(2,12),'sedentary_hours':(0,24),'BMXBMI':(0,100)}
clean=a[features].copy()
for c in features:
    clean[c]=pd.to_numeric(clean[c],errors='coerce')
    clean[c]=clean[c].mask(clean[c].abs()<1e-50,0)
    if c in valid_cat:clean.loc[~clean[c].isin(valid_cat[c]),c]=np.nan
    if c in ranges:clean.loc[~clean[c].between(*ranges[c]),c]=np.nan
miss=(clean.isna().mean()*100).sort_values(ascending=False)
pd.DataFrame({'variable':miss.index,'missing_n':[int(clean[c].isna().sum()) for c in miss.index],
              'missing_pct':miss.values}).to_csv(OUT/'eda_predictor_missingness.csv',index=False)
plt.figure(figsize=(9,5));plt.barh(miss.index[::-1],miss.values[::-1]);plt.xlabel('Missing or invalid after validation (%)')
plt.title('Figure 2. Missingness in candidate predictors');save('eda_missingness.png')

# Figure 3: Sleep duration
sleep=pd.to_numeric(a.SLD012,errors='coerce');sleep=sleep.where(sleep.between(2,12))
groups=[sleep[a.elevated_symptoms==v].dropna().values for v in [0,1]]
plt.figure(figsize=(7,4.8));plt.boxplot(groups,labels=['PHQ-9 <10','PHQ-9 ≥10'],showfliers=False)
plt.ylabel('Weekday sleep duration (hours)');plt.title('Figure 3. Sleep duration by screening group');save('eda_sleep_by_outcome.png')

# Figure 4: Sedentary hours
sed=pd.to_numeric(a.sedentary_hours,errors='coerce');sed=sed.where(sed.between(0,24))
groups=[sed[a.elevated_symptoms==v].dropna().values for v in [0,1]]
plt.figure(figsize=(7,4.8));plt.boxplot(groups,labels=['PHQ-9 <10','PHQ-9 ≥10'],showfliers=False)
plt.ylabel('Sedentary time (hours/day)');plt.title('Figure 4. Sedentary hours by screening group');save('eda_sedentary_by_outcome.png')

# Figure 5: unweighted fraction by sex (NHANES codes: 1 male, 2 female)
sex=pd.to_numeric(a.RIAGENDR,errors='coerce')
sex=sex.map({1:'Male (recorded)',2:'Female (recorded)'})
sub=pd.DataFrame({'sex':sex,'positive':a.elevated_symptoms}).dropna()
prev=sub.groupby('sex').positive.agg(['mean','size','sum']).reset_index()
prev['percent']=prev['mean']*100
prev.to_csv(OUT/'eda_unweighted_sex_prevalence.csv',index=False)
plt.figure(figsize=(7,4.8));plt.bar(prev.sex,prev.percent)
plt.ylabel('PHQ-9 ≥10 (%) — unweighted');plt.title('Figure 5. Screening fraction by recorded sex')
for i,v in enumerate(prev.percent):plt.text(i,v+.2,f'{v:.1f}%',ha='center')
save('eda_prevalence_by_group.png')

# Audit and summary tables (do not claim raw-source duplicate counts without source)
if 'SEQN' in a:
    dup=int(a.SEQN.duplicated().sum());print('Duplicate SEQN in valid-outcome table:',dup)
    pd.DataFrame({'measure':['eligible_valid_phq9','positive','negative','duplicate_seqn'],
                  'value':[len(a),int(a.elevated_symptoms.sum()),int((1-a.elevated_symptoms).sum()),dup]}).to_csv(OUT/'eda_cohort_summary.csv',index=False)
print('Completed EDA. The plots are descriptive/unweighted; use R survey package for national estimates.')
