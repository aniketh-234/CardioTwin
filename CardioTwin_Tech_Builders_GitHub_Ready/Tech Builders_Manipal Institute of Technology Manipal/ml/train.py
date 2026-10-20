import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,roc_auc_score,average_precision_score,
                             precision_score,recall_score,f1_score,brier_score_loss,confusion_matrix,log_loss)
from sklearn.model_selection import train_test_split
from ml.models import CardioModel
from ml.preprocessing import Preprocessor


def metrics(y,p):
    pred=(p>=.5).astype(int); n=len(y); acc=accuracy_score(y,pred); z=1.96
    center=(acc+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(acc),'accuracy_ci95':[float(center-half),float(center+half)],
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'roc_auc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)), 'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),
            'brier':float(brier_score_loss(y,p)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def run(data='data',out='artifacts',epochs=100,seed=42):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    z=np.load(Path(data)/'benchmark.npz'); y=z['y']; ids=z['ids']
    train,rest=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=seed)
    val,test=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=seed)
    sp=Preprocessor().fit(z['static'][train]);dp=Preprocessor().fit(z['sequence'][train])
    s=torch.from_numpy(sp.transform(z['static']));d=torch.from_numpy(dp.transform(z['sequence']));t=torch.from_numpy(y)
    (path/'preprocessing.json').write_text(json.dumps({'static':sp.dump(),'dynamic':dp.dump()}))
    (path/'splits.json').write_text(json.dumps({k:ids[v].tolist() for k,v in [('train',train),('validation',val),('test',test)]}))
    records=[]; predframe=pd.DataFrame({'patient_id':ids[test],'label':y[test]})
    # Baselines use exactly the same patient splits and train-fitted features.
    for name,x in [('logistic_static',s.numpy()),('logistic_fusion',np.concatenate([s.numpy(),d.numpy().mean(1)],1))]:
        model=LogisticRegression(max_iter=1000,random_state=seed).fit(x[train],y[train])
        pv=model.predict_proba(x[val])[:,1];pt=model.predict_proba(x[test])[:,1]
        records.append({'model':name,'validation_brier':float(brier_score_loss(y[val],pv)),**metrics(y[test],pt)})
        predframe[name]=pt
        (path/(name+'.json')).write_text(json.dumps({'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist()}))
    mean=float(y[train].mean())
    records.append({'model':'prevalence_baseline','validation_brier':float(brier_score_loss(y[val],np.full(len(val),mean))),
                    **metrics(y[test],np.full(len(test),mean))})
    history={}
    for kind in ['mlp','cnn','lstm','gru']:
        start=time.time();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=CardioModel(kind,s.shape[-1],d.shape[-1],d.shape[1],hidden=16)
        opt=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=.001)
        loss_fn=torch.nn.BCEWithLogitsLoss();best=float('inf');state=None;wait=0;curve=[]
        for epoch in range(epochs):
            model.train()
            order=np.random.permutation(train)
            for offset in range(0,len(order),128):
                ix=order[offset:offset+128];opt.zero_grad()
                loss=loss_fn(model(s[ix],d[ix]),t[ix]);loss.backward();opt.step()
            model.eval()
            with torch.no_grad():vp=torch.sigmoid(model(s[val],d[val])).numpy()
            score=float(brier_score_loss(y[val],vp));curve.append(score)
            if score<best-1e-5:best=score;state=copy.deepcopy(model.state_dict());wait=0;bestepoch=epoch+1
            else:wait+=1
            if wait>=12:break
        model.load_state_dict(state);model.eval()
        with torch.no_grad():pt=torch.sigmoid(model(s[test],d[test])).numpy()
        record={'model':kind,'validation_brier':best,'best_epoch':bestepoch,'seconds':round(time.time()-start,2),**metrics(y[test],pt)}
        records.append(record);predframe[kind]=pt;history[kind]=curve
        torch.save({'kind':kind,'static_dim':s.shape[-1],'dynamic_dim':d.shape[-1],'days':d.shape[1],
                    'hidden':16,'state_dict':state},path/(kind+'.pt'))
        print(json.dumps(record),flush=True)
    winner=min(records,key=lambda r:r['validation_brier'])['model']
    neural=min([r for r in records if r['model'] in ['mlp','cnn','lstm','gru']],key=lambda r:r['validation_brier'])['model']
    result={'dataset':'Entirely synthetic sandbox v1','clinical_validation':False,'patients':len(y),
            'train_count':len(train),'validation_count':len(val),'test_count':len(test),
            'test_positive_count':int(y[test].sum()),'seed':seed,'threshold':.5,'selected_model':winner,
            'selected_neural_model':neural,'selection_rule':'lowest validation Brier score, before interpreting test results',
            'models':records,'torch_version':torch.__version__}
    (path/'metrics.json').write_text(json.dumps(result,indent=2));pd.DataFrame(records).to_csv(path/'metrics.csv',index=False)
    (path/'training_history.json').write_text(json.dumps(history));predframe.to_csv(path/'test_predictions.csv',index=False)
    # Display subjects come from held-out patients; labels remain in evaluation files only.
    picks=test[:24]
    demo=[]
    for i in picks:
        seq=z['sequence'][i]
        demo.append({'id':str(ids[i]),'static':z['static'][i].tolist(),
                     'sequence':[[None if not np.isfinite(v) else float(v) for v in row] for row in seq]})
    (Path(data)/'demo_patients.json').write_text(json.dumps(demo,allow_nan=False))
    print('Selected model:',winner,'; selected neural model:',neural,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--out',default='artifacts')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();run(a.data,a.out,a.epochs,a.seed)
