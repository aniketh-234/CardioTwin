'use strict';
const $=id=>document.getElementById(id);
const names={mlp:'MLP',cnn:'1D CNN',lstm:'LSTM',gru:'GRU',logistic_static:'Logistic · static',logistic_fusion:'Logistic · fusion',prevalence_baseline:'Prevalence baseline'};
const featureNames={age:'Age',sex_male:'Recorded sex',bmi:'BMI',systolic_bp:'Systolic BP',total_cholesterol:'Total cholesterol',hdl:'HDL',smoker:'Smoking indicator',diabetes:'Diabetes indicator',wearable_and_cuff_history:'Wearable + cuff history'};
names.hybrid_cnn='Hybrid · CNN + rules';
Object.assign(featureNames,{resting_hr:'Resting HR',hrv_rmssd:'HRV',steps:'Steps',sleep_hours:'Sleep',home_systolic_bp:'Home BP'});
const hybridOption=new Option('Hybrid · CNN + rules','hybrid_cnn',true,true);
$('model-select').prepend(hybridOption);$('model-select').value='hybrid_cnn';
const hybridCss=document.createElement('link');hybridCss.rel='stylesheet';hybridCss.href='/hybrid.css';document.head.append(hybridCss);
const hybridPanel=document.createElement('article');hybridPanel.className='panel hybrid-panel';
hybridPanel.innerHTML='<h3>Hybrid review &amp; explainability</h3><p>CNN probability and independent trend rules. The hybrid demo uses the CNN; the original validation winner remains logistic fusion.</p><div id="hybrid-status" role="status"></div><p id="quality-reasons"></p><div id="rule-checks"></div><p class="fine">Recent 7 days compared with the preceding 21. Two directional changes of at least 0.5 training standard deviations trigger review. These are synthetic engineering thresholds, not clinical cutoffs.</p><h4>Which signals and weeks influence this model?</h4><p class="muted">Replace observed readings with training medians and measure the score difference. Positive values mean the original readings produced a higher score. Missing readings stay missing.</p><div class="table-scroll"><table id="temporal-table"><caption>Grouped reference replacement · percentage-point differences</caption><thead><tr><th>Signal</th><th>Week 1</th><th>Week 2</th><th>Week 3</th><th>Week 4 (recent)</th></tr></thead><tbody id="temporal-body"></tbody></table></div><p class="fine">These sensitivities depend on the reference, are not additive SHAP values, and do not establish causes or treatment benefits.</p><div id="update-explanation" aria-live="polite"></div><button id="trend-demo" class="ghost">Replay trend example (7 days)</button>';
document.querySelector('#twin .top-grid').after(hybridPanel);
function renderHybrid(r){
 const labels={insufficient_data:'Prediction withheld',review_signal:'Review signal: multiple trends',limited_trend_data:'Limited history for trend checks',no_trend_flag:'No multi-signal trend flag'};
 $('hybrid-status').textContent=labels[r.review_status];$('hybrid-status').className=r.review_status==='review_signal'?'review-flag':'review-normal';
 $('quality-reasons').textContent=r.quality.reasons.join(' ');$('rule-checks').replaceChildren();
 r.rules.checks.forEach(c=>{const row=document.createElement('p');row.className=c.triggered?'rule-triggered':'rule-neutral';row.textContent=featureNames[c.signal]+': '+(!c.available?'Insufficient history (need 14 earlier and 5 recent observations)':`${c.baseline.toFixed(1)} → ${c.recent.toFixed(1)} · ${c.triggered?'TRIGGERED':'not triggered'} · directional change ${c.standardized_change.toFixed(2)} SD`);$('rule-checks').append(row);});
 $('temporal-body').replaceChildren();
 if(r.explanation){['resting_hr','hrv_rmssd','steps','sleep_hours','home_systolic_bp'].forEach(signal=>{const tr=document.createElement('tr');const label=document.createElement('th');label.textContent=featureNames[signal];tr.append(label);r.explanation.temporal.filter(x=>x.signal===signal).forEach(x=>{const td=document.createElement('td');td.textContent=(x.delta_pp>0?'+':'')+x.delta_pp.toFixed(2)+' pp';td.className=x.delta_pp>.05?'effect-up':x.delta_pp<-.05?'effect-down':'effect-neutral';tr.append(td);});$('temporal-body').append(tr);});}
 else {const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=5;td.textContent='Explanations withheld until data quality recovers.';tr.append(td);$('temporal-body').append(tr);}
}
let patient=null,metrics=null,busy=false;
function showError(e){$('error').textContent=e.message||String(e);$('error').hidden=false;}
async function api(path,body){const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const value=await response.json();if(!response.ok)throw Error(value.error||'Request failed');return value;}
function selectedModel(){return $('model-select').value||undefined;}
function pct(v){return v===null?'Withheld':(100*v).toFixed(1)+'%';}
async function guarded(fn){if(busy)return;busy=true;$('error').hidden=true;document.querySelectorAll('button,select').forEach(b=>b.disabled=true);try{await fn();}catch(e){showError(e);}finally{busy=false;document.querySelectorAll('button,select').forEach(b=>b.disabled=false);}}
async function loadPatient(){const id=$('patient-select').value;patient=await api('/api/patients/'+id+($('model-select').value?'?model='+encodeURIComponent($('model-select').value):''));renderPatient();}
function renderPatient(){
 const p=patient,r=p.prediction,s=p.static;
 $('patient-name').textContent=p.id;$('patient-meta').textContent=`${Math.round(s[0])} years · ${s[1]?'Male':'Female'} · Synthetic held-out patient`;
 $('last-date').textContent=p.last_date;$('risk').textContent=pct(r.probability);$('risk').style.fontSize=r.probability===null?'42px':'';
 $('risk-note').textContent=r.status==='ready'?'Hypothetical first cardiovascular event':r.message;
 $('risk-fill').style.width=((r.probability||0)*100)+'%';$('model-name').textContent=names[r.model];
 $('coverage').textContent=pct(r.quality.coverage);$('updates').textContent=p.updates;$('quality-badge').textContent=r.status==='ready'?'READY FOR DEMO':'DATA INCOMPLETE';
 $('static-grid').replaceChildren();
 ['Age','Sex','BMI','Systolic BP','Cholesterol','HDL','Smoker','Diabetes'].forEach((label,i)=>{let v=i===1?(s[i]?'Male':'Female'):i>=6?(s[i]?'Yes':'No'):Math.round(s[i]);const el=document.createElement('div');const sm=document.createElement('small');sm.textContent=label;const st=document.createElement('strong');st.textContent=v;el.append(sm,st);$('static-grid').append(el);});
 $('drivers').replaceChildren();r.drivers.forEach(x=>{const row=document.createElement('div');row.className='driver';const a=document.createElement('span');a.textContent=featureNames[x.feature]||x.feature;const b=document.createElement('strong');b.textContent=(x.delta_pp>0?'+':'')+x.delta_pp.toFixed(1)+' pp';if(x.delta_pp<0)b.className='negative';row.append(a,b);$('drivers').append(row);});
 if(!r.drivers.length)$('drivers').textContent='Restore data coverage to view sensitivities.';
 $('scenario-result').textContent='';$('update-explanation').textContent='';renderHybrid(r);drawChart();
}
function drawChart(){if(!patient)return;const canvas=$('trajectory'),ctx=canvas.getContext('2d'),rect=canvas.getBoundingClientRect();if(rect.width===0)return;const ratio=window.devicePixelRatio||1;canvas.width=rect.width*ratio;canvas.height=220*ratio;ctx.scale(ratio,ratio);const width=rect.width,height=220,k=Number($('signal-select').value),vals=patient.sequence.map(r=>r[k]);const good=vals.filter(v=>v!==null&&Number.isFinite(v));ctx.clearRect(0,0,width,height);if(!good.length){ctx.fillText('No readings available',30,100);return;}
 let lo=Math.min(...good),hi=Math.max(...good),pad=Math.max((hi-lo)*.25,1);lo-=pad;hi+=pad;const left=38,right=14,top=15,bottom=25;
 const x=i=>left+i*(width-left-right)/27,y=v=>top+(hi-v)/(hi-lo)*(height-top-bottom);
 ctx.font='10px system-ui';ctx.lineWidth=1;
 for(let i=0;i<4;i++){let v=lo+i*(hi-lo)/3;ctx.strokeStyle='#edf2f5';ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(width-right,y(v));ctx.stroke();ctx.fillStyle='#91a1ac';ctx.fillText(Math.round(v),0,y(v)+3);}
 ctx.strokeStyle='#098c86';ctx.lineWidth=2.5;ctx.beginPath();let active=false;vals.forEach((v,i)=>{if(v===null){active=false;return;}if(active)ctx.lineTo(x(i),y(v));else ctx.moveTo(x(i),y(v));active=true;});ctx.stroke();ctx.fillStyle='#098c86';vals.forEach((v,i)=>{if(v!==null){ctx.beginPath();ctx.arc(x(i),y(v),2.5,0,Math.PI*2);ctx.fill();}});
 const end=new Date(patient.last_date+'T00:00:00Z'),start=new Date(end);start.setUTCDate(start.getUTCDate()-27);$('chart-start').textContent=start.toISOString().slice(0,10);$('chart-end').textContent=patient.last_date;
 canvas.setAttribute('aria-label',`${$('signal-select').selectedOptions[0].text}: ${good.length} observed days, range ${Math.min(...good).toFixed(1)} to ${Math.max(...good).toFixed(1)}`);
}
async function ingest(values){let next=new Date(patient.last_date+'T00:00:00Z');next.setUTCDate(next.getUTCDate()+1);const result=await api('/api/ingest',{patient_id:patient.id,model:selectedModel(),date:next.toISOString().slice(0,10),readings:values});await loadPatient();const u=result.prediction.update_comparison;$('update-explanation').textContent=`Latest update: ${pct(u.before)} → ${pct(u.after)}. ${u.delta_pp===null?'A probability comparison requires adequate data before and after the update.':(u.delta_pp>0?'+':'')+u.delta_pp.toFixed(2)+' percentage points.'} ${u.note}`;}
function appendMetricRow(label,r,hybrid=false){
 const tr=document.createElement('tr');
 if(hybrid){tr.id='hybrid-metrics-row';tr.className='hybrid-metrics-row';}
 else if(r.model===metrics.selected_model)tr.className='selected-row';
 const values=[label,pct(r.accuracy),r.accuracy_ci95.map(pct).join(' – '),
  hybrid?'—':r.roc_auc.toFixed(3),pct(r.recall),pct(r.precision),r.f1.toFixed(3),
  hybrid?'—':r.brier.toFixed(3),r.fp,r.fn];
 values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;
  if(hybrid&&(i===3||i===7))td.title='This row evaluates a binary review policy. The probability score is unchanged from the CNN; see its ROC-AUC and Brier above.';
  tr.append(td);});$('metrics-body').append(tr);
}
function renderMetrics(){
 const m=metrics;$('benchmark-summary').replaceChildren();
 [[m.patients,'Synthetic patients'],[m.train_count,'Training patients'],[m.test_count,'Held-out test patients'],[m.test_positive_count,'Test events']].forEach(([v,label])=>{const el=document.createElement('div'),s=document.createElement('strong'),sm=document.createElement('small');s.textContent=v.toLocaleString();sm.textContent=label;el.append(s,sm);$('benchmark-summary').append(el);});
 const head=$('metrics-body').closest('table').querySelector('thead tr');head.replaceChildren();
 ['Model / review policy','Accuracy','95% interval','ROC-AUC','Recall','Precision','F1','Brier ↓','False alerts','Missed events'].forEach(label=>{const th=document.createElement('th');th.textContent=label;head.append(th);});
 $('metrics-body').replaceChildren();m.models.forEach(r=>appendMetricRow(names[r.model]+(r.model===m.selected_model?' · selected':''),r));
}
function renderHybridMetricRow(report){
 renderMatchedComparison(report.matched_alert_comparison);
 const c=report.combined_review;if(!c)return;
 const n=c.tp+c.tn+c.fp+c.fn;if(!n)return;
 const accuracy=(c.tp+c.tn)/n,z=1.96,denom=1+z*z/n;
 const center=(accuracy+z*z/(2*n))/denom;
 const half=z*Math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/denom;
 $('hybrid-metrics-row')?.remove();
 appendMetricRow('Hybrid CNN + rules · review policy',{...c,accuracy,
  accuracy_ci95:[center-half,center+half],f1:2*c.tp/(2*c.tp+c.fp+c.fn)||0},true);
 let note=$('hybrid-metrics-note');
 if(!note){note=document.createElement('p');note.id='hybrid-metrics-note';note.className='fine';$('metrics-body').closest('.table-scroll').after(note);}
 note.textContent=`Hybrid row: alert when CNN probability ≥ 50% OR the multi-signal rule fires. Evaluated on ${n} patients; ${report.abstained} withheld. Accuracy, recall, precision and F1 measure that combined decision. ROC-AUC and Brier are shown only for the unchanged CNN probability. This row was not selected by validation Brier score.`;
}
function renderMatchedComparison(report){
 if(!report)return;
 $('matched-review-comparison')?.remove();
 const panel=document.createElement('article');panel.id='matched-review-comparison';panel.className='panel hybrid-panel';
 const heading=document.createElement('h3');heading.textContent='Exploratory comparison: similar validation alert rates';
 const summary=document.createElement('p');
 summary.textContent=`CNN threshold chosen on validation: ${(100*report.selected_threshold).toFixed(2)}%. Validation alerts: hybrid ${report.validation.hybrid.alerts}/${report.validation.hybrid.evaluated}; CNN at this threshold ${report.validation.cnn_matched.alerts}/${report.validation.cnn_matched.evaluated}. The threshold was frozen before test inference. Live predictions still use the original configuration.`;
 const wrap=document.createElement('div');wrap.className='table-scroll';
 const table=document.createElement('table'),head=document.createElement('thead'),header=document.createElement('tr'),body=document.createElement('tbody');
 ['Test policy','Alerts','Recall','Precision','F1','False alerts','Missed events'].forEach(name=>{const th=document.createElement('th');th.textContent=name;header.append(th);});head.append(header);
 [['cnn_50','CNN ≥ 50%'],['hybrid','CNN ≥ 50% OR rules'],['cnn_matched',`CNN ≥ ${(100*report.selected_threshold).toFixed(2)}%`]].forEach(([key,label])=>{
  const r=report.test[key],row=document.createElement('tr');
  [label,`${r.alerts}/${r.evaluated}`,pct(r.recall),pct(r.precision),r.f1.toFixed(3),r.fp,r.fn].forEach(value=>{const td=document.createElement('td');td.textContent=value;row.append(td);});body.append(row);
 });
 table.append(head,body);wrap.append(table);
 const counts=document.createElement('p');counts.textContent=`Test abstentions: ${report.test.cnn_matched.abstained}. Alert rates are matched on validation; test counts may differ.`;
 const caution=document.createElement('p');caution.className='fine';caution.textContent=report.limitations;
 panel.append(heading,summary,wrap,counts,caution);$('evidence').append(panel);
}
async function loadAudit(){const rows=await api('/api/audit');$('audit-list').replaceChildren();if(!rows.length){$('audit-list').textContent='No updates yet. Ingest a daily reading to start the journal.';return;}rows.forEach(r=>{const el=document.createElement('div');el.className='audit-row';const title=document.createElement('strong');title.textContent=r.patient+' · '+r.action;const sm=document.createElement('small');sm.textContent=r.time;const code=document.createElement('code');code.textContent=r.detail;el.append(title,sm,code);$('audit-list').append(el);});}
document.querySelectorAll('.nav').forEach(button=>button.onclick=()=>guarded(async()=>{document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==button.dataset.view);$('page-title').textContent={twin:'Patient workspace',evidence:'Model evidence',journal:'Event journal'}[button.dataset.view];if(button.dataset.view==='journal')await loadAudit();if(button.dataset.view==='twin')drawChart();}));
$('patient-select').onchange=()=>guarded(loadPatient);$('model-select').onchange=()=>guarded(loadPatient);$('signal-select').onchange=drawChart;window.addEventListener('resize',drawChart);
$('ingest').onclick=()=>guarded(()=>{const values=['hr','hrv','steps','sleep','bp','coverage'].map(k=>{const el=$('in-'+k);if(!el.value||!el.checkValidity())throw Error('Check the reading ranges before ingestion.');return Number(el.value);});return ingest(values);});
$('replay').onclick=()=>guarded(async()=>{for(let i=0;i<7;i++){await ingest([70+i,46-i,6500-200*i,7.4-.1*i,Math.min(190,Math.round(patient.static[3])+i),.95]);await new Promise(r=>setTimeout(r,250));}});
$('dropout').onclick=()=>guarded(async()=>{for(let i=0;i<4;i++)await ingest([null,null,null,null,null,.1]);});
$('reset').onclick=()=>guarded(async()=>{await api('/api/reset',{patient_id:patient.id});await loadPatient();});
$('trend-demo').onclick=()=>guarded(async()=>{for(let i=0;i<7;i++)await ingest([100,18,1500,4.5,175,.95]);});
async function loadHybridEvidence(){const m=await api('/api/hybrid-metrics');renderHybridMetricRow(m);const panel=document.createElement('article');panel.className='panel hybrid-panel';const title=document.createElement('h3');title.textContent='Hybrid review policy: measured tradeoff';const text=document.createElement('p');text.textContent=m.cnn_only?`On ${m.patients} held-out synthetic patients: CNN at 50% detects ${m.cnn_only.tp} events with ${m.cnn_only.fp} false alerts. CNN OR trend flag detects ${m.combined_review.tp} events with ${m.combined_review.fp} false alerts. Recall: ${pct(m.cnn_only.recall)} → ${pct(m.combined_review.recall)}; precision: ${pct(m.cnn_only.precision)} → ${pct(m.combined_review.precision)}. More detected events come with more false alerts. Probability estimates remain those of the CNN.`:m.note;const note=document.createElement('p');note.className='fine';note.textContent=m.note;panel.append(title,text,note);$('evidence').append(panel);}
[['scenario-bp','bp-out'],['scenario-steps','steps-out'],['scenario-sleep','sleep-out']].forEach(([input,out])=>$(input).oninput=()=>$(out).textContent=$(input).value);
$('scenario-run').onclick=()=>guarded(async()=>{const r=await api('/api/scenario',{patient_id:patient.id,model:selectedModel(),adjustments:{systolic_bp:Number($('scenario-bp').value),steps:Number($('scenario-steps').value),sleep_hours:Number($('scenario-sleep').value)}});$('scenario-result').textContent=`${pct(r.baseline.probability)} → ${pct(r.scenario.probability)} · hypothetical profile`;});
$('refresh-audit').onclick=()=>guarded(loadAudit);
$('export').onclick=()=>{if(!patient)return;const blob=new Blob([JSON.stringify({disclosure:'Synthetic sandbox only. Not clinical advice.',patient},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=patient.id+'-snapshot.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
guarded(async()=>{const patients=await api('/api/patients');patients.forEach(p=>{const option=document.createElement('option');option.value=p.id;option.textContent=`${p.id} · ${p.age}y ${p.sex}`;$('patient-select').append(option);});metrics=await api('/api/metrics');renderMetrics();await loadPatient();await loadHybridEvidence();});
