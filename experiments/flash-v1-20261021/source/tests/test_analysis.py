import unittest
from flash.analysis import diagnose
class AnalysisTest(unittest.TestCase):
 def test_two_stage(self):
  rows=[{'comm_hash':1,'rank':r,'avg_us':(10 if r==2 else 100)} for r in range(4)]
  rows +=[{'comm_hash':2,'rank':r,'avg_us':10} for r in range(4,8)]
  result=diagnose(rows,{'dp':[list(range(4)),list(range(4,8))]})
  self.assertEqual(result['identified_ranks'],[2])
 def test_uniform_is_not_fault(self):
  self.assertFalse(diagnose([{'comm_hash':1,'rank':r,'avg_us':10} for r in range(4)],{})['identified_ranks'])
 def test_missing_peers_not_diagnosed(self):
  self.assertFalse(diagnose([{'comm_hash':1,'rank':2,'avg_us':1}],{})['identified_ranks'])
if __name__=='__main__': unittest.main()

class CrossValidationTest(unittest.TestCase):
 def test_propagated_waiting_does_not_override_specific_group(self):
  rows=[{'comm_hash':1,'rank':r,'avg_us':(10 if r==2 else 1000)} for r in range(4)]
  # A downstream all-rank collective makes the entire first group late.
  # Its absolute minimum is rank 3, although ranks 0..3 are effectively tied.
  rows +=[{'comm_hash':2,'rank':r,'avg_us':(9 if r==3 else 10 if r<4 else 1000)} for r in range(8)]
  topo={'ep':[list(range(4))],'dp':[list(range(8))]}
  literal=diagnose(rows,topo,cross_validate=False)
  self.assertEqual(set(literal['identified_ranks']),{2,3})
  self.assertEqual(diagnose(rows,topo)['identified_ranks'],[2])

class AmbiguityTest(unittest.TestCase):
 def test_one_group_cannot_disambiguate_tied_late_cohort(self):
  rows=[{'comm_hash':1,'rank':r,'avg_us':(10+r*.001 if r<4 else 1000)} for r in range(8)]
  result=diagnose(rows,{'dp':[list(range(8))]})
  self.assertEqual(result['identified_ranks'],[])
  self.assertEqual(result['candidate_ranks'],[0,1,2,3])
  self.assertTrue(result['ambiguous'])

class PipelineTagsTest(unittest.TestCase):
 def test_different_pipeline_stages_are_not_compared_as_peer_cliques(self):
  rows=[{'comm_hash':1,'rank':0,'avg_us':10},{'comm_hash':1,'rank':1,'avg_us':100},
        {'comm_hash':2,'rank':2,'avg_us':100},{'comm_hash':2,'rank':3,'avg_us':100}]
  result=diagnose(rows,{'expert_ep':[[0,1],[2,3]],'pp':[[0,2],[1,3]]})
  self.assertEqual(result['identified_ranks'],[0])
  self.assertEqual(result['ranking'][0]['evidence'][0]['tag'],'ppstage0_expert_ep')

class PaperTextRulesTest(unittest.TestCase):
 def test_group_cv_does_not_imply_an_extra_clique_score_gate(self):
  rows=[{'comm_hash':1,'rank':0,'avg_us':1},{'comm_hash':1,'rank':1,'avg_us':100},
        {'comm_hash':2,'rank':2,'avg_us':80},{'comm_hash':2,'rank':3,'avg_us':80}]
  topology={'dp':[[0,1],[2,3]]}
  self.assertEqual(diagnose(rows,topology,paper_text=True)['identified_ranks'],[0])
  self.assertEqual(diagnose(rows,topology,cross_validate=False)['identified_ranks'],[])
 def test_rank_cv_then_minimum_kept_separate_from_undefined_20pct_formula(self):
  rows=[{'comm_hash':1,'rank':0,'avg_us':85},{'comm_hash':1,'rank':1,'avg_us':115}]
  self.assertEqual(diagnose(rows,{},paper_text=True)['identified_ranks'],[0])
  self.assertEqual(diagnose(rows,{},cross_validate=False)['identified_ranks'],[])
