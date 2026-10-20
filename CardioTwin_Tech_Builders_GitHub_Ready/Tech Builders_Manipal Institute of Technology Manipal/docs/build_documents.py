"""Generate the required PDF presentation and architecture diagram."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
M=json.loads((ROOT/'artifacts/metrics.json').read_text())
W,H=960,540
FONT_DIR=Path('/usr/share/fonts/truetype/dejavu')
if FONT_DIR.exists():
    pdfmetrics.registerFont(TTFont('Body',str(FONT_DIR/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('Bold',str(FONT_DIR/'DejaVuSans-Bold.ttf')))
NAVY=HexColor('#123747');TEAL=HexColor('#00867d');INK=HexColor('#244253');MUTED=HexColor('#6c8492')


def text(c,value,x,y,size=20,color=INK,width=820,font='Helvetica'):
    if FONT_DIR.exists():font='Bold' if font=='Helvetica-Bold' else 'Body'
    c.setFillColor(color);c.setFont(font,size)
    lines=simpleSplit(value,font,size,width)
    for line in lines:c.drawString(x,y,line);y-=size*1.4
    return y


def base(c,title,number):
    c.setFillColor(HexColor('#f7fafb'));c.rect(0,0,W,H,fill=1,stroke=0)
    text(c,title,58,466,32,NAVY,font='Helvetica-Bold')
    c.setStrokeColor(TEAL);c.setLineWidth(3);c.line(58,437,155,437)
    text(c,'CardioTwin | Synthetic research prototype',58,25,10,MUTED)
    text(c,str(number),885,25,10,MUTED)


def bullets(c,items,y=385):
    for heading,body in items:
        y=text(c,heading,58,y,22,NAVY,font='Helvetica-Bold')-7
        y=text(c,body,58,y,17,MUTED,width=820)-27


def architecture(c):
    # Explicit architecture diagram, not decorative artwork.
    boxes=[(60,295,240,90,'Historical profile','8 static EHR-style inputs'),
           (660,295,240,90,'Daily readings','28 days, 6 dynamic inputs'),
           (330,170,300,90,'Validate and preprocess','Range checks, masks, train-fit scaling'),
           (60,50,360,80,'Trained model service','Fusion inference + insufficient-data gate'),
           (540,50,360,80,'Doctor-facing dashboard','Timeline, sensitivity, evidence, journal')]
    for x,y,w,h,title,sub in boxes:
        c.setStrokeColor(TEAL);c.setFillColor(white);c.roundRect(x,y,w,h,8,fill=1,stroke=1)
        text(c,title,x+15,y+h-28,17,NAVY,width=w-30,font='Helvetica-Bold')
        text(c,sub,x+15,y+h-53,12,MUTED,width=w-30)
    c.setStrokeColor(MUTED);c.setLineWidth(1.5)
    for points in [[(180,295),(180,280),(415,280),(415,260)],[(780,295),(780,280),(545,280),(545,260)],
                   [(480,170),(480,150),(240,150),(240,130)],[(420,90),(540,90)]]:
        p=c.beginPath();p.moveTo(*points[0])
        for pt in points[1:]:p.lineTo(*pt)
        c.drawPath(p)
        x,y=points[-1];c.circle(x,y,3,fill=1,stroke=0)


def main():
    out=ROOT/'docs';c=canvas.Canvas(str(out/'presentation.pdf'),pagesize=(W,H))
    c.setTitle('CardioTwin - Digital Twin Challenge 2026')
    c.setFillColor(NAVY);c.rect(0,0,W,H,fill=1,stroke=0)
    text(c,'CardioTwin',65,335,62,white,font='Helvetica-Bold')
    text(c,'A working cardiovascular digital-twin sandbox',68,268,25,HexColor('#a9ddd6'))
    text(c,'Historical profile + dynamic readings + trained inference',68,220,19,white)
    text(c,'All data and outcomes are simulated. Clinical validity remains unestablished.',68,95,15,HexColor('#c2d6dd'))
    text(c,'Team and college: complete before submission',68,57,12,HexColor('#8da9b9'));c.showPage()
    base(c,'The prototype and its boundary',2)
    bullets(c,[('Focused use case','Explore a five-year hypothetical cardiovascular event probability as a patient twin receives new observations.'),
               ('Doctor-facing workflow','Review history, inspect daily changes, compare an input scenario and trace each update.'),
               ('Evidence boundary','This is an engineering proof of concept. It does not estimate validated real-patient risk or prescribe treatment.')]);c.showPage()
    base(c,'The data decision',3)
    bullets(c,[('Original sample audit','108 Synthea patients. A candidate 2020 index produced only 5 later events, 56 candidate negatives and 7 with insufficient follow-up.'),
               ('Matched synthetic benchmark','4,000 new simulated adults, 28 days per patient and a stochastic five-year outcome. No unrelated patient datasets are joined.'),
               ('Transparent assumptions','The simulator equation is open in ml/generate.py. Its coefficients are engineering assumptions, not clinical estimates.')]);c.showPage()
    base(c,'Architecture',4);architecture(c);c.showPage()
    base(c,'Model comparison protocol',5)
    bullets(c,[('Four neural models and simple baselines','MLP, 1D CNN, LSTM and GRU fuse a static branch with dynamic inputs. Logistic models test static-only and fused features.'),
               ('Patient-level isolation','2,400 training, 800 validation and 800 test patients. Only training patients determine preprocessing statistics.'),
               ('Selection before final evaluation','Early stopping and model choice use validation Brier score. The probability threshold stays fixed at 0.5.')]);c.showPage()
    base(c,'Measured results on simulated patients',6)
    cols=[60,290,440,590,740];headers=['Model','Accuracy','ROC-AUC','Recall','Brier']
    for x,h in zip(cols,headers):text(c,h,x,388,15,MUTED,font='Helvetica-Bold')
    y=345
    for r in M['models']:
        vals=[r['model'].replace('_',' '),f"{r['accuracy']*100:.2f}%",f"{r['roc_auc']:.3f}",f"{r['recall']*100:.1f}%",f"{r['brier']:.3f}"]
        for x,v in zip(cols,vals):text(c,v,x,y,16,TEAL if r['model']==M['selected_model'] else INK)
        y-=34
    text(c,'800 held-out patients, 177 events. Logistic fusion wins validation Brier score.',60,83,15,MUTED)
    text(c,'Accuracy alone hides low event recall. These results measure the simulator only.',60,58,14,MUTED);c.showPage()
    base(c,'The live demonstration',7)
    bullets(c,[('Update the twin','Ingest a daily summary or replay seven days. The rolling context updates and the backend recomputes the score.'),
               ('Handle missing information','Simulate sensor dropout. Insufficient recent data withholds the score rather than presenting a reassuring number.'),
               ('Inspect the decision trail','Show input sensitivities, the separate what-if comparison, model evidence and the persisted event journal.')]);c.showPage()
    base(c,'What makes the prototype credible',8)
    bullets(c,[('Reproducible results','Generator seed, patient splits, preprocessing, four trained checkpoints and individual test predictions ship with the code.'),
               ('Visible limitations','Every screen identifies synthetic data. Scenario changes are model sensitivities, not treatment effects.'),
               ('Tested behavior','Automated checks verify split isolation, temporal labels, train-only preprocessing, serving parity and missing-data abstention.')]);c.showPage()
    base(c,'The route to useful clinical evidence',9)
    bullets(c,[('Replace the simulator with matched follow-up','Obtain an appropriately governed longitudinal cohort with adjudicated cardiovascular events and dated sensor data.'),
               ('Measure transportability','Evaluate calibration, event recall and subgroup performance on external patients, including the intended Indian setting.'),
               ('Validate the workflow','Work with clinicians on acceptable missingness, decision thresholds and prospective usefulness before patient-facing deployment.')]);c.showPage()
    base(c,'Submission and reproducibility',10)
    bullets(c,[('Included in the repository package','Source, synthetic data, checkpoints, metrics, architecture PDF, presentation PDF, model card and a timed demonstration script.'),
               ('Team-owned completion','Add team and college details, record a minimum 20-minute walkthrough, publish the repository and insert the accessible video link.'),
               ('Rules source','Unstop: Digital Twin Challenge 2026, Happiest Health. Requirements checked 13 October 2026.')]);c.showPage();c.save()
    c=canvas.Canvas(str(out/'architecture.pdf'),pagesize=(W,H));c.setTitle('CardioTwin architecture')
    base(c,'CardioTwin system architecture',1);architecture(c);c.save()
    print('Created presentation.pdf (10 pages) and architecture.pdf (1 page).')


if __name__=='__main__':main()
