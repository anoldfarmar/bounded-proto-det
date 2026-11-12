"""Numerical invariants of the spectral cut and its automatic foreground selection."""
from pathlib import Path
import json
import unittest
import numpy as np
from scipy.linalg import eigh
from ncut import cosine_graph,solve_graph,select_regions,ncut_regions

CFG=json.loads(Path(__file__).with_name('config.json').read_text())['ncut']


class NCutTests(unittest.TestCase):
    def test_normalized_solver_matches_generalized_problem(self):
        rng=np.random.RandomState(7);a=rng.rand(30,30);a=(a+a.T)/2+.01
        values,vector=solve_graph(a);degree=np.diag(a.sum(axis=1))
        expected,vectors=eigh(degree-a,degree,subset_by_index=[1,2])
        np.testing.assert_allclose(values,expected,atol=1e-12)
        correlation=abs(np.dot(vector,vectors[:,0])/(np.linalg.norm(vector)*np.linalg.norm(vectors[:,0])))
        self.assertGreater(correlation,1-1e-10)
        np.testing.assert_allclose((degree-a)@vector,values[0]*degree@vector,atol=1e-12)

    def test_selection_invariant_to_eigenvector_sign(self):
        vector=np.array([-2.,-.6,-.7,-.5,-.4,.8,.9,.7,1.1])
        first=select_regions(vector,(3,3),CFG)
        second=select_regions(-vector,(3,3),CFG)
        np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])

    def test_two_separable_blocks_recovered_without_labels(self):
        rng=np.random.RandomState(2)
        features=np.concatenate([np.tile([1.,0.,0.],(32,1)),np.tile([0.,1.,0.],(32,1))])+.01*rng.randn(64,3)
        main,_,info,_=ncut_regions(features,(8,8),CFG,.5)
        self.assertEqual(main.sum(),32)
        self.assertTrue(main[:4].all() or main[4:].all())
        self.assertIsNone(info['main']['fallback'])

    def test_constant_features_fall_back_instead_of_arbitrary_mask(self):
        main,center,info,_=ncut_regions(np.ones((64,6)),(8,8),CFG,.5)
        self.assertTrue(main.all() and center.all())
        self.assertEqual(info['main']['fallback'],'constant_graph')

    def test_cosine_scale_invariance(self):
        rng=np.random.RandomState(6);features=rng.randn(10,6)
        np.testing.assert_allclose(cosine_graph(features),cosine_graph(features*np.arange(1,11)[:,None]),atol=1e-14)


if __name__=='__main__':unittest.main()
