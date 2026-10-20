import unittest
import numpy as np
from ml.compare_review_policies import select_threshold,summarize


class ReviewComparisonTests(unittest.TestCase):
    def test_exact_count_and_unordered_probabilities(self):
        p=np.array([.3,.9,.1,.6])
        t=select_threshold(p,2)
        self.assertEqual(int((p>=t).sum()),2)

    def test_tied_scores_prefer_fewer_alerts(self):
        p=np.array([.9,.5,.5,.1])
        t=select_threshold(p,2)
        self.assertEqual(int((p>=t).sum()),1)

    def test_zero_and_all_alerts(self):
        p=np.array([0.,.5,1.])
        self.assertEqual(int((p>=select_threshold(p,0)).sum()),0)
        self.assertEqual(int((p>=select_threshold(p,3)).sum()),3)

    def test_abstentions_are_reported_not_counted_as_negatives(self):
        r=summarize([1,1,0,0],[1,0,1,0],np.array([True,False,True,True]))
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['tn']),(1,1,0,1))
        self.assertEqual(r['abstained_events'],1)


if __name__=='__main__':unittest.main()
