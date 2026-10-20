"""Evaluate a frozen review policy without changing data, models or thresholds."""
import json
import numpy as np
from backend.engine import Engine, ROOT


def evaluate():
    engine=Engine();z=np.load(ROOT/'data/benchmark.npz')
    splits=json.loads((ROOT/'artifacts/splits.json').read_text())
    index={str(pid):i for i,pid in enumerate(z['ids'])}
    ix=[index[pid] for pid in splits['test']]
    y=z['y'][ix].astype(bool);cnn=[];rules=[];ready=[];probs=[]
    for i in ix:
        result=engine.evaluate(z['static'][i],z['sequence'][i],'hybrid_cnn',False)
        available=result['probability'] is not None
        ready.append(available);probs.append(result['probability'])
        cnn.append(available and result['probability']>=.5)
        rules.append(available and result['rules']['review_flag'])
    cnn=np.asarray(cnn);rules=np.asarray(rules);ready=np.asarray(ready)
    def counts(alert):
        tp=int((alert & y & ready).sum());fp=int((alert & ~y & ready).sum())
        fn=int((~alert & y & ready).sum());tn=int((~alert & ~y & ready).sum())
        return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else None,
                    precision=tp/(tp+fp) if tp+fp else 0,alerts=int(alert.sum()))
    result={'patients':len(ix),'abstained':int((~ready).sum()),
            'cnn_threshold':.5,'rule_version':'sandbox-trends-v1',
            'policy':'Comparison only: CNN >= 0.5 OR multi-signal rule flag. Dashboard keeps probability and trend flags separate.',
            'cnn_only':counts(cnn),'rules_only':counts(rules),'combined_review':counts(cnn|rules),
            'note':'Fixed engineering rules, not optimized on this test set. Future rules need fresh evaluation. Synthetic five-year labels do not validate short-term alerts. Hybrid probabilities equal CNN probabilities; no probability improvement is claimed.'}
    (ROOT/'artifacts/hybrid_evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':evaluate()
