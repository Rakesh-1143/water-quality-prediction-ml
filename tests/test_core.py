import unittest
import numpy as np
import pandas as pd
from config import FEATURES, DATA_PATH
from wqi.core import load_data, split_data, preprocessor, shap_importance, input_kernel, average, metrics

class CoreTests(unittest.TestCase):
    def test_dataset_and_disjoint_stratified_splits(self):
        X,y=load_data(DATA_PATH)
        self.assertEqual(X.shape,(3276,9))
        tr,va,te,a,b,c=split_data(X,y)
        sets=[set(z.index) for z in [tr,va,te]]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),3276)
        for labels in [a,b,c]: self.assertLess(abs(labels.mean()-y.mean()),.003)
    def test_train_only_median_and_scaling(self):
        tr=pd.DataFrame(np.tile([0.,2.,np.nan,4.],(9,1)).T,columns=FEATURES)
        test=pd.DataFrame([[100.]*9],columns=FEATURES)
        p=preprocessor(); p.fit(tr)
        np.testing.assert_allclose(p.named_steps['imputer'].statistics_,[2.]*9)
        self.assertTrue((p.transform(test)>1).all())
        np.testing.assert_allclose(p.named_steps['scaler'].data_max_,[4.]*9)
    def test_shap_weight_matrix(self):
        vals=np.array([np.arange(1,10),-np.arange(1,10)])
        score,w=shap_importance(vals)
        np.testing.assert_allclose(score,np.arange(1,10))
        self.assertAlmostEqual(w.sum(),1)
        k=input_kernel(w); self.assertEqual(k.shape,(9,16))
        np.testing.assert_allclose(k[:,0],w,rtol=1e-6)
        np.testing.assert_allclose(k[:,15],w,rtol=1e-6)
        with self.assertRaises(ValueError): shap_importance(np.zeros((2,9)))
    def test_fusion_and_probability_metrics(self):
        np.testing.assert_allclose(average([.2,.8],[.6,.4]),[.4,.6])
        with self.assertRaises(ValueError): average([.1],[.2],1.1)
        m=metrics([0,1],[.2,.8])
        self.assertEqual(m['accuracy'],1); self.assertAlmostEqual(m['rmse'],.2)
        self.assertAlmostEqual(m['mae'],.2); self.assertEqual(m['roc_auc'],1)
        self.assertEqual(metrics([0,1],[.49,.5])['accuracy'],1)
        with self.assertRaises(ValueError): metrics([0,1],[np.nan,.5])

if __name__=='__main__': unittest.main()
