"""Transparent, unvalidated sandbox simulator. NOT a clinical risk equation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

STATIC = ['age', 'sex_male', 'bmi', 'systolic_bp', 'total_cholesterol', 'hdl', 'smoker', 'diabetes']
DYNAMIC = ['resting_hr', 'hrv_rmssd', 'steps', 'sleep_hours', 'home_systolic_bp', 'coverage']
LIMITS = [[30,80],[0,1],[18,42],[90,190],[120,320],[25,95],[0,1],[0,1]]
DYNAMIC_LIMITS = [[40,120],[8,110],[0,25000],[2,11],[85,200],[0,1]]


def generate(n=4000, seed=2026, out='data'):
    r = np.random.default_rng(seed)
    age = r.uniform(30,80,n); male = r.binomial(1,.5,n)
    bmi = np.clip(r.normal(27,4.5,n),18,42)
    bp = np.clip(112+.48*(age-30)+.9*(bmi-25)+r.normal(0,12,n),90,190)
    chol = np.clip(r.normal(205,35,n),120,320); hdl = np.clip(r.normal(53,13,n),25,95)
    smoke = r.binomial(1,.23,n); diabetes = r.binomial(1,1/(1+np.exp(-(-2.5+.06*(age-45)+.12*(bmi-27)))),n)
    static = np.column_stack([age,male,bmi,bp,chol,hdl,smoke,diabetes]).astype('float32')
    burden = r.normal(0,1,n); trend = r.normal(0,.65,n)
    sequences = []
    for day in range(28):
        state = burden + trend * day/27 + r.normal(0,.25,n)
        steps = np.clip(7000-1500*state-45*(age-50)+r.normal(0,1200,n),0,25000)
        sleep = np.clip(7.2-.65*state+r.normal(0,.55,n),2,11)
        hr = np.clip(67+5*state+2*smoke+r.normal(0,3,n),40,120)
        hrv = np.clip(52-9*state-.25*(age-50)+r.normal(0,5,n),8,110)
        sbp = np.clip(bp+5*state+r.normal(0,5,n),85,200)
        coverage = r.uniform(.7,1,n)
        values = np.column_stack([hr,hrv,steps,sleep,sbp,coverage])
        missing = r.random((n,5)) < .04
        values[:,:5][missing] = np.nan
        sequences.append(values)
    sequence = np.stack(sequences,axis=1).astype('float32')
    # Hypothetical first cardiovascular event in five years. Coefficients are
    # engineering assumptions, not fitted to evidence or a validated clinical score.
    logit = (-2.2 + .045*(age-50)+.025*(bp-125)+.014*(chol-200)-.025*(hdl-50)
             +.6*smoke+.7*diabetes+.10*(bmi-27)+.65*burden+.5*trend
             + r.normal(0,.45,n))
    probability = 1/(1+np.exp(-logit))
    outcomes = r.binomial(1,probability).astype('float32')
    ids = np.array([f'SIM-{i:05d}' for i in range(n)])
    path = Path(out); path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'benchmark.npz',static=static,sequence=sequence,y=outcomes,ids=ids)
    pd.DataFrame(static,columns=STATIC).assign(patient_id=ids).to_csv(path/'static.csv',index=False)
    dates = pd.date_range('2021-01-01',periods=28,tz='UTC')
    frame = pd.DataFrame(sequence.reshape(-1,6),columns=DYNAMIC)
    frame.insert(0,'date',np.tile(dates.strftime('%Y-%m-%d'),n))
    frame.insert(0,'patient_id',np.repeat(ids,28)); frame.to_csv(path/'daily_readings.csv',index=False)
    index = pd.Timestamp('2021-01-29',tz='UTC'); horizon = index+pd.DateOffset(years=5)
    eventdates = [(index+pd.Timedelta(days=int(r.integers(1,1826)))).isoformat() if y else '' for y in outcomes]
    pd.DataFrame({'patient_id':ids,'prediction_time':index.isoformat(),'event_date':eventdates,
                  'followup_end':horizon.isoformat(),'event_within_5y':outcomes.astype(int),
                  'provenance':'custom_sandbox_simulator_v1'}).to_csv(path/'outcomes.csv',index=False)
    meta = {'seed':seed,'patients':n,'days':28,'static_features':STATIC,'dynamic_features':DYNAMIC,
            'positive_count':int(outcomes.sum()),'prevalence':float(outcomes.mean()),
            'outcome':'Hypothetical first cardiovascular event within five years',
            'source':'Custom synthetic sandbox. Not Synthea-derived or clinically validated.',
            'index_time':index.isoformat(),'static_limits':LIMITS,'dynamic_limits':DYNAMIC_LIMITS,
            'limitations':['Simulator coefficients are unvalidated assumptions.',
                          'No real patient outcomes or Indian population calibration.',
                          'Held-out performance measures reproduction of this simulator only.']}
    (path/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'patients':n,'positive_count':int(outcomes.sum()),'prevalence':float(outcomes.mean())}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--patients',type=int,default=4000)
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--out',default='data')
    a=p.parse_args(); generate(a.patients,a.seed,a.out)
