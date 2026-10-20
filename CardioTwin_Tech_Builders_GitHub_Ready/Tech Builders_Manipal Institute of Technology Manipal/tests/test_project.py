import copy,json,tempfile,unittest,threading,urllib.request,urllib.error
from pathlib import Path
import numpy as np
import pandas as pd
from backend.engine import Engine,ROOT
from ml.preprocessing import Preprocessor
from backend import server


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=Engine()
        cls.patient=json.loads((ROOT/'data/demo_patients.json').read_text())[0]

    def test_patient_splits_and_outcome_timing(self):
        splits=json.loads((ROOT/'artifacts/splits.json').read_text());sets=[set(v) for v in splits.values()]
        self.assertEqual(sum(map(len,sets)),4000)
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        outcomes=pd.read_csv(ROOT/'data/outcomes.csv');positive=outcomes[outcomes.event_within_5y==1]
        self.assertTrue((pd.to_datetime(positive.event_date)>pd.to_datetime(positive.prediction_time)).all())
        self.assertTrue((pd.to_datetime(positive.event_date)<=pd.to_datetime(positive.followup_end)).all())

    def test_serving_matches_saved_test_predictions(self):
        frame=pd.read_csv(ROOT/'artifacts/test_predictions.csv').set_index('patient_id')
        p=self.patient
        for name in ['mlp','cnn','lstm','gru','logistic_static','logistic_fusion']:
            prediction=self.engine.probability(p['static'],p['sequence'],name)
            self.assertAlmostEqual(prediction,frame.loc[p['id'],name],places=5)

    def test_missing_data_withholds_prediction(self):
        p=copy.deepcopy(self.patient)
        p['sequence'][-7:]=[[None,None,None,None,None,.1]]*7
        result=self.engine.evaluate(p['static'],p['sequence'])
        self.assertEqual(result['status'],'insufficient_data');self.assertIsNone(result['probability'])

    def test_range_and_shape_validation(self):
        p=copy.deepcopy(self.patient);p['static'][3]=999
        with self.assertRaises(ValueError):self.engine.evaluate(p['static'],p['sequence'])
        with self.assertRaises(ValueError):self.engine.evaluate(self.patient['static'],[[1]*6])

    def test_preprocessing_fit_matches_train_only(self):
        z=np.load(ROOT/'data/benchmark.npz');splits=json.loads((ROOT/'artifacts/splits.json').read_text())
        ix=np.isin(z['ids'],splits['train'])
        expected=Preprocessor().fit(z['static'][ix])
        np.testing.assert_allclose(expected.mean,self.engine.sp.mean)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.old_db=server.DB
        server.DB=Path(cls.temp.name)/'test.sqlite3';server.ENGINE=Engine()
        cls.http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.http.server_port}'
        cls.pid=next(iter(server.FIXTURES))

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close();cls.thread.join();server.DB=cls.old_db;cls.temp.cleanup()

    def request(self,path,body=None):
        req=urllib.request.Request(self.base+path,data=json.dumps(body).encode() if body else None,
                                   headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as response:return json.load(response)

    def test_ingest_duplicate_and_scenario_isolation(self):
        self.request('/api/reset',{'patient_id':self.pid})
        before=self.request('/api/patients/'+self.pid)
        scenario=self.request('/api/scenario',{'patient_id':self.pid,'adjustments':{'steps':8000}})
        self.assertIsNotNone(scenario['scenario']['probability'])
        after=self.request('/api/patients/'+self.pid)
        self.assertEqual(before,after)
        body={'patient_id':self.pid,'date':'2021-01-29','readings':[72,45,6500,7,130,.95]}
        body['model']='hybrid_cnn'
        result=self.request('/api/ingest',body)
        self.assertIn('update_comparison',result['prediction'])
        self.assertEqual(result['prediction']['model'],'hybrid_cnn')
        self.assertEqual(len(result['prediction']['explanation']['temporal']),20)
        with self.assertRaises(urllib.error.HTTPError) as cm:self.request('/api/ingest',body)
        self.assertEqual(cm.exception.code,400)
        self.assertEqual(self.request('/api/patients/'+self.pid)['updates'],1)

    def test_unknown_model_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:self.request('/api/patients/'+self.pid+'?model=invalid')
        self.assertEqual(cm.exception.code,400)


if __name__=='__main__':unittest.main()
