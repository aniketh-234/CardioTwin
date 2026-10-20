import json
from pathlib import Path
import numpy as np
import torch
from ml.models import CardioModel
from ml.preprocessing import Preprocessor
from backend.rules import trend_rules

ROOT=Path(__file__).resolve().parents[1]


class Engine:
    def __init__(self):
        torch.set_num_threads(2)
        self.metrics=json.loads((ROOT/'artifacts/metrics.json').read_text())
        self.meta=json.loads((ROOT/'data/metadata.json').read_text())
        p=json.loads((ROOT/'artifacts/preprocessing.json').read_text())
        self.sp=Preprocessor.load(p['static']);self.dp=Preprocessor.load(p['dynamic'])
        self.models={}
        for kind in ['mlp','cnn','lstm','gru']:
            c=torch.load(ROOT/f'artifacts/{kind}.pt',map_location='cpu',weights_only=True)
            model=CardioModel(c['kind'],c['static_dim'],c['dynamic_dim'],c['days'],c['hidden'])
            model.load_state_dict(c['state_dict']);model.eval();self.models[kind]=model
        for kind in ['logistic_static','logistic_fusion']:
            self.models[kind]=json.loads((ROOT/f'artifacts/{kind}.json').read_text())

    def validate(self,static,sequence):
        s=np.asarray(static,dtype=float);d=np.asarray(sequence,dtype=float)
        if s.shape!=(8,) or d.shape!=(28,6):raise ValueError('Expected 8 static values and 28 rows of 6 daily values')
        for arr,limits in [(s,self.meta['static_limits']),(d,self.meta['dynamic_limits'])]:
            for k,(low,high) in enumerate(limits):
                v=arr[...,k]
                if np.isinf(v).any() or ((v<low)|(v>high)).any():raise ValueError(f'Feature {k} falls outside simulator limits')
        for k in [1,6,7]:
            if np.isfinite(s[k]) and s[k] not in [0,1]:raise ValueError('Binary inputs must be 0 or 1')
        return s,d

    def probability(self,static,sequence,kind=None):
        kind=kind or self.metrics['selected_model']
        if kind=='hybrid_cnn':kind='cnn'
        s=self.sp.transform(np.asarray(static,dtype=float)[None,:]);d=self.dp.transform(np.asarray(sequence,dtype=float)[None,:,:])
        if kind=='prevalence_baseline':return float(json.loads((ROOT/'data/metadata.json').read_text())['prevalence'])
        if kind not in self.models:raise ValueError('Unknown model')
        model=self.models[kind]
        if isinstance(model,dict):
            x=s if kind=='logistic_static' else np.concatenate([s,d.mean(1)],1)
            logit=float((x@np.array(model['coef']).T+model['intercept']).item())
            return float(1/(1+np.exp(-np.clip(logit,-50,50))))
        with torch.no_grad():return float(torch.sigmoid(model(torch.from_numpy(s),torch.from_numpy(d))).item())

    def evaluate(self,static,sequence,kind=None,explain=True):
        s,d=self.validate(static,sequence)
        kind=kind or self.metrics['selected_model']
        if kind not in list(self.models)+['prevalence_baseline','hybrid_cnn']:raise ValueError('Unknown model')
        coverage=float(np.nanmean(d[-7:,5])) if np.isfinite(d[-7:,5]).any() else 0
        missing=float(np.isnan(d[-7:,:5]).mean())
        quality={'coverage':coverage,'missing_fraction':missing,'window_days':28}
        reasons=[]
        if coverage<.6:reasons.append('Recent coverage is below 60%.')
        if missing>.4:reasons.append('More than 40% of recent sensor values are missing.')
        if np.isnan(s).mean()>.5:reasons.append('More than half of historical values are missing.')
        quality['reasons']=reasons
        rules=trend_rules(d,self.dp.scale)
        if coverage<.6 or missing>.4 or np.isnan(s).mean()>.5:
            return {'probability':None,'status':'insufficient_data','model':kind,'quality':quality,'drivers':[],
                    'rules':rules,'review_status':'insufficient_data','explanation':None,
                    'message':'Prediction withheld: insufficient recent coverage or too many missing inputs.'}
        p=self.probability(s,d,kind);drivers=[]
        temporal=[]
        if explain:
            for k,name in enumerate(self.meta['static_features']):
                changed=s.copy();changed[k]=self.sp.median[k]
                delta=p-self.probability(changed,d,kind)
                drivers.append({'feature':name,'delta_pp':round(delta*100,2),'reference':float(self.sp.median[k])})
            changed=d.copy();changed[:,:5]=self.dp.median[:5]
            drivers.append({'feature':'wearable_and_cuff_history','delta_pp':round((p-self.probability(s,changed,kind))*100,2),
                            'reference':'training medians'})
            for k,name in enumerate(self.meta['dynamic_features'][:5]):
                for week in range(4):
                    changed=d.copy();block=changed[week*7:(week+1)*7,k]
                    block[np.isfinite(block)]=self.dp.median[k]
                    temporal.append({'signal':name,'week':week+1,
                                     'delta_pp':round((p-self.probability(s,changed,kind))*100,4)})
        return {'probability':p,'status':'ready','model':kind,'quality':quality,
                'rules':rules,
                'review_status':('review_signal' if rules['review_flag'] else
                                 'limited_trend_data' if rules['available_signals']<5 else 'no_trend_flag'),
                'explanation':{'method':'grouped_reference_replacement',
                               'reference':'Training medians; temporal replacements preserve missing values.',
                               'temporal':temporal,
                               'note':'Sensitivity of this model, not SHAP, causation or treatment benefit. Values are not additive.'} if explain else None,
                'drivers':sorted(drivers,key=lambda a:abs(a['delta_pp']),reverse=True)[:5],
                'message':'Five-year event probability under the custom simulator. Not a clinical risk estimate.'}
