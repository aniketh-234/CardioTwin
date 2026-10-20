"""Local-only demo service. JSON API + static frontend + SQLite event journal."""
import argparse,copy,json,sqlite3,threading
from contextlib import contextmanager
from datetime import date,datetime,timedelta,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from backend.engine import Engine,ROOT

ENGINE=None
LOCK=threading.RLock()
DB=ROOT/'runtime.sqlite3'
FIXTURES={p['id']:p for p in json.loads((ROOT/'data/demo_patients.json').read_text())}


@contextmanager
def connect():
    conn=sqlite3.connect(DB)
    try:
        with conn:
            conn.execute('CREATE TABLE IF NOT EXISTS patients(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, time TEXT, patient TEXT, action TEXT, detail TEXT)')
            yield conn
    finally:
        conn.close()


def get_patient(pid):
    if pid not in FIXTURES:raise ValueError('Unknown patient')
    with connect() as db:
        row=db.execute('SELECT payload FROM patients WHERE id=?',(pid,)).fetchone()
    p=json.loads(row[0]) if row else copy.deepcopy(FIXTURES[pid])
    p.setdefault('last_date','2021-01-28');p.setdefault('updates',0)
    return p


def write_patient(p,action,detail):
    with connect() as db:
        db.execute('INSERT OR REPLACE INTO patients VALUES (?,?)',(p['id'],json.dumps(p,allow_nan=False)))
        db.execute('INSERT INTO audit(time,patient,action,detail) VALUES (?,?,?,?)',
                   (datetime.now(timezone.utc).isoformat(),p['id'],action,json.dumps(detail,allow_nan=False)))


class Handler(BaseHTTPRequestHandler):
    def send_json(self,value,status=200):
        raw=json.dumps(value,allow_nan=False).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        try:
            url=urlparse(self.path);query=parse_qs(url.query)
            with LOCK:
                if url.path=='/api/health':return self.send_json({'status':'ok','mode':'synthetic_sandbox'})
                if url.path=='/api/metrics':return self.send_json(ENGINE.metrics)
                if url.path=='/api/hybrid-metrics':
                    report=ROOT/'artifacts/hybrid_evaluation.json'
                    return self.send_json(json.loads(report.read_text()) if report.exists() else {'note':'Run python -m ml.evaluate_hybrid to generate this report.'})
                if url.path=='/api/meta':return self.send_json(ENGINE.meta)
                if url.path=='/api/patients':
                    return self.send_json([{'id':p['id'],'age':round(p['static'][0]),'sex':'M' if p['static'][1] else 'F'} for p in FIXTURES.values()])
                if url.path.startswith('/api/patients/'):
                    p=get_patient(url.path.rsplit('/',1)[-1]);p['prediction']=ENGINE.evaluate(p['static'],p['sequence'],query.get('model',[None])[0])
                    return self.send_json(p)
                if url.path=='/api/audit':
                    with connect() as db:rows=db.execute('SELECT time,patient,action,detail FROM audit ORDER BY id DESC LIMIT 30').fetchall()
                    return self.send_json([dict(zip(['time','patient','action','detail'],r)) for r in rows])
            files={'/':'index.html','/app.js':'app.js','/style.css':'style.css','/hybrid.css':'hybrid.css'}
            if url.path not in files:return self.send_json({'error':'Not found'},404)
            file=ROOT/'frontend'/files[url.path];raw=file.read_bytes();self.send_response(200)
            self.send_header('Content-Type',{'.html':'text/html','.js':'application/javascript','.css':'text/css'}[file.suffix])
            self.send_header('Content-Length',str(len(raw)));self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(raw)
        except (ValueError,TypeError,KeyError) as e:self.send_json({'error':str(e)},400)
        except Exception as e:
            print('Server error:',repr(e));self.send_json({'error':'Internal server error'},500)

    def do_POST(self):
        try:
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://{self.headers.get("Host")}']:
                return self.send_json({'error':'Cross-origin writes are not accepted'},403)
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=65536:raise ValueError('Expected a JSON body under 64 KiB')
            body=json.loads(self.rfile.read(size),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON number')))
            with LOCK:
                p=get_patient(body['patient_id']);kind=body.get('model')
                if self.path=='/api/ingest':
                    day=date.fromisoformat(body['date'])
                    if day!=date.fromisoformat(p['last_date'])+timedelta(days=1):raise ValueError('Daily readings must advance exactly one day; duplicates are rejected')
                    row=body['readings']
                    if not isinstance(row,list) or len(row)!=6:raise ValueError('Expected six daily values')
                    updated=p['sequence'][1:]+[row]
                    before=ENGINE.evaluate(p['static'],p['sequence'],kind,False)
                    prediction=ENGINE.evaluate(p['static'],updated,kind)
                    prediction['update_comparison']={'before':before['probability'],'after':prediction['probability'],
                        'delta_pp':None if before['probability'] is None or prediction['probability'] is None else
                        (prediction['probability']-before['probability'])*100,
                        'note':'Net effect of dropping the oldest day and appending the new day.'}
                    p['sequence']=updated;p['last_date']=day.isoformat();p['updates']+=1
                    write_patient(p,'ingest',{'date':p['last_date'],'readings':row,'model':prediction['model'],'probability':prediction['probability'],
                                            'rule_version':prediction['rules']['version'],'review_status':prediction['review_status'],
                                            'triggered_signals':prediction['rules']['triggered_signals'],
                                            'update_comparison':prediction['update_comparison']})
                    return self.send_json({'patient':p,'prediction':prediction})
                if self.path=='/api/scenario':
                    s=p['static'].copy();d=copy.deepcopy(p['sequence'])
                    adjustments=body.get('adjustments',{})
                    allowed={'systolic_bp','steps','sleep_hours','smoker'}
                    if set(adjustments)-allowed:raise ValueError('Unknown scenario input')
                    for key,value in adjustments.items():
                        if not isinstance(value,(int,float)) or isinstance(value,bool):raise ValueError('Scenario values must be numeric')
                        if key=='systolic_bp':
                            s[3]=value
                            for row in d:row[4]=value
                        elif key=='smoker':s[6]=value
                        else:
                            k=2 if key=='steps' else 3
                            for row in d:row[k]=value
                    baseline=ENGINE.evaluate(p['static'],p['sequence'],kind,False)
                    scenario=ENGINE.evaluate(s,d,kind,False)
                    return self.send_json({'baseline':baseline,'scenario':scenario,
                                           'note':'Input sensitivity only. This does not estimate a causal treatment effect.'})
                if self.path=='/api/reset':
                    p=copy.deepcopy(FIXTURES[p['id']]);p['last_date']='2021-01-28';p['updates']=0
                    write_patient(p,'reset',{});return self.send_json({'status':'reset'})
                return self.send_json({'error':'Not found'},404)
        except (ValueError,TypeError,KeyError) as e:self.send_json({'error':str(e)},400)
        except Exception as e:
            print('Server error:',repr(e));self.send_json({'error':'Internal server error'},500)


def main():
    global ENGINE
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);a=p.parse_args()
    ENGINE=Engine()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(f'CardioTwin sandbox: http://127.0.0.1:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()


if __name__=='__main__':main()
