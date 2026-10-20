import copy
import json
import unittest
import numpy as np
from backend.engine import Engine, ROOT
from backend.rules import trend_rules


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=Engine()
        cls.patient=json.loads((ROOT/'data/demo_patients.json').read_text())[0]

    def test_hybrid_preserves_cnn_probability(self):
        p=self.patient
        a=self.engine.evaluate(p['static'],p['sequence'],'hybrid_cnn')
        self.assertAlmostEqual(a['probability'],self.engine.probability(p['static'],p['sequence'],'cnn'),places=7)
        self.assertEqual(len(a['explanation']['temporal']),20)

    def test_temporal_explanation_matches_actual_perturbation(self):
        p=self.patient;d=np.asarray(p['sequence'],dtype=float)
        original=d.copy();a=self.engine.evaluate(p['static'],d,'hybrid_cnn')
        block=d[:7,0];block[np.isfinite(block)]=self.engine.dp.median[0]
        expected=100*(a['probability']-self.engine.probability(p['static'],d,'cnn'))
        self.assertAlmostEqual(a['explanation']['temporal'][0]['delta_pp'],expected,places=3)
        np.testing.assert_equal(original,np.asarray(p['sequence'],dtype=float))

    def test_rules_require_multiple_directional_changes(self):
        d=np.tile([70,45,6500,7,125,.95],(28,1)).astype(float)
        scales=np.ones(6)
        self.assertFalse(trend_rules(d,scales)['review_flag'])
        d[-7:,0]+=5
        self.assertFalse(trend_rules(d,scales)['review_flag'])
        d[-7:,2]-=100
        self.assertTrue(trend_rules(d,scales)['review_flag'])
        d[:21,2]=np.nan
        self.assertFalse(trend_rules(d,scales)['review_flag'])

    def test_quality_gate_overrides_trend_flag(self):
        p=copy.deepcopy(self.patient)
        p['sequence'][-7:]=[[100,18,1500,4.5,175,.1]]*7
        a=self.engine.evaluate(p['static'],p['sequence'],'hybrid_cnn')
        self.assertIsNone(a['probability']);self.assertIsNone(a['explanation'])
        self.assertEqual(a['review_status'],'insufficient_data')
        self.assertTrue(a['quality']['reasons'])


if __name__=='__main__':unittest.main()
