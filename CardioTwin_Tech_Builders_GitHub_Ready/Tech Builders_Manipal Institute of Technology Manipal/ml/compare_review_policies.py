"""Exploratory validation-matched alert-rate comparison; no model retraining."""
import hashlib
import json
import numpy as np
from backend.engine import Engine, ROOT
from backend.rules import RULE_VERSION


def select_threshold(validation_probabilities, target_alerts):
    """No labels or test inputs: closest alert count; ties prefer fewer alerts."""
    p=np.asarray(validation_probabilities,dtype=float)
    if not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected finite validation probabilities in [0, 1]')
    if not 0 <= target_alerts <= len(p):raise ValueError('Invalid target alert count')
    candidates=np.r_[np.nextafter(p.max(),np.inf),np.unique(p)]
    threshold=min(candidates,key=lambda t:(abs(int((p>=t).sum())-target_alerts),int((p>=t).sum()),-float(t)))
    return float(threshold)


def summarize(labels,alerts,ready):
    y=np.asarray(labels,dtype=bool);a=np.asarray(alerts,dtype=bool)&ready
    tp=int((a&y).sum());fp=int((a&~y).sum())
    fn=int((~a&y&ready).sum());tn=int((~a&~y&ready).sum());n=int(ready.sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'alerts':tp+fp,'evaluated':n,
            'abstained':int((~ready).sum()),'abstained_events':int((y&~ready).sum()),
            'alert_rate':(tp+fp)/n if n else None,'accuracy':(tp+tn)/n if n else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'precision':tp/(tp+fp) if tp+fp else 0,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


def run():
    engine=Engine();z=np.load(ROOT/'data/benchmark.npz')
    splits=json.loads((ROOT/'artifacts/splits.json').read_text())
    index={str(pid):i for i,pid in enumerate(z['ids'])}
    if set(splits['validation'])&set(splits['test']):raise ValueError('Overlapping patient partitions')
    def infer(partition):
        rows=[]
        for pid in splits[partition]:
            i=index[pid];r=engine.evaluate(z['static'][i],z['sequence'][i],'hybrid_cnn',False)
            rows.append({'patient_id':pid,'probability':r['probability'],
                         'rule_flag':r['rules']['review_flag'],'label':int(z['y'][i])})
        ready=np.array([r['probability'] is not None for r in rows])
        p=np.array([r['probability'] if r['probability'] is not None else np.nan for r in rows])
        flags=np.array([r['rule_flag'] for r in rows])&ready
        labels=np.array([r['label'] for r in rows],dtype=bool)
        return rows,p,flags,labels,ready
    val,pv,rv,yv,av=infer('validation')
    target=int((((pv>=.5)|rv)&av).sum())
    threshold=select_threshold(pv[av],target)
    # Threshold is fixed before test inference and is never adjusted on test results.
    test,pt,rt,yt,at=infer('test')
    def policies(p,r,y,a):
        return {'cnn_50':summarize(y,p>=.5,a),
                'hybrid':summarize(y,(p>=.5)|r,a),
                'cnn_matched':summarize(y,p>=threshold,a)}
    result={'status':'exploratory','selected_threshold':threshold,'rule_version':RULE_VERSION,
            'selection':'Match the validation hybrid alert count using CNN probabilities only. Closest achievable count; ties prefer fewer alerts, then higher threshold. No outcome labels or test data used for threshold selection.',
            'validation_target_alerts':target,'validation':policies(pv,rv,yv,av),
            'test':policies(pt,rt,yt,at),
            'limitations':'The test set was inspected before proposing this analysis. Validation was previously used for CNN early stopping. Results are exploratory on the existing synthetic cohort, not fresh independent validation. Matching validation alert rates does not guarantee equal test alert rates. No live threshold or model was changed.',
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                             ['data/benchmark.npz','artifacts/cnn.pt','artifacts/preprocessing.json','artifacts/splits.json','backend/rules.py']}}
    path=ROOT/'artifacts/hybrid_evaluation.json'
    report=json.loads(path.read_text());report['matched_alert_comparison']=result
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'artifacts/review_comparison_patients.json').write_text(json.dumps({'validation':val,'test':test},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
