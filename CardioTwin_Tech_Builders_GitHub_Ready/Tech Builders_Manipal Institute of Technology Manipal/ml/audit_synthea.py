"""Audit the supplied source without treating last encounter as proven ascertainment."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import pandas as pd


def audit(source,out):
    with zipfile.ZipFile(source) as z:
        def read(name):return pd.read_csv(z.open(next(n for n in z.namelist() if Path(n).name==name)))
        patients,obs,conditions,encounters=[read(n+'.csv') for n in ['patients','observations','conditions','encounters']]
    obs.DATE=pd.to_datetime(obs.DATE,utc=True);conditions.START=pd.to_datetime(conditions.START,utc=True)
    encounters.START=pd.to_datetime(encounters.START,utc=True)
    codes=['414545008','22298006','401314000','401303003','88805009']
    events=conditions[conditions.CODE.astype(str).isin(codes)].groupby('PATIENT').START.min()
    start=pd.Timestamp('2020-01-01',tz='UTC');end=start+pd.DateOffset(years=5)
    counts={'positive':0,'candidate_negative':0,'insufficient_followup':0,'ineligible':0}
    for _,p in patients.iterrows():
        age=(start-pd.Timestamp(p.BIRTHDATE,tz='UTC')).days/365.25
        event=events.get(p.Id,pd.NaT)
        if age<18 or obs[(obs.PATIENT==p.Id)&(obs.DATE<start)].empty or (pd.notna(event) and event<=start):
            counts['ineligible']+=1;continue
        last=encounters[encounters.PATIENT==p.Id].START.max()
        if pd.notna(event) and start<event<=end:counts['positive']+=1
        elif last>=end:counts['candidate_negative']+=1
        else:counts['insufficient_followup']+=1
    result={'source_filename':Path(source).name,'sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'patients':len(patients),'observations':len(obs),'candidate_index':start.isoformat(),
            'horizon_years':5,'candidate_codes':codes,'counts':counts,
            'decision':'Do not use this small cohort for neural benchmark claims.',
            'caveat':'Last encounter after horizon is only a follow-up proxy. Endpoint ascertainment and historical eligibility need further verification.',
            'relationship_to_benchmark':'No source patient was copied or joined into the custom simulator benchmark.'}
    Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',default='docs/synthea_audit.json')
    a=p.parse_args();audit(a.source,a.out)
