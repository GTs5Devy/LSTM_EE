"""Regression tests for horizon boundaries, controls and warning counting."""
import unittest
import numpy as np
import pandas as pd
import torch
from dataset_lstm import FEATURES, build_windows
from diagnose_samples import self_test
from experiment_config import writable_output
from model_last_frame import LastFrameMLP
from event_evaluation import evaluate_events

class PipelineTests(unittest.TestCase):
    def data(self):
        d=pd.DataFrame({f:np.arange(80,dtype=float) for f in FEATURES})
        d['Lap']=1; d['CurrentTime']=np.arange(80)*16; d['Label']='Neutral'
        return d

    def test_existing_sanity_suite(self): self_test()

    def test_late_onset_and_future_input_separation(self):
        d=self.data(); d.loc[44:46,'Label']='Understeer'
        x,y,m=build_windows(d,[1]); self.assertEqual(y[0],1)
        self.assertEqual(m[0]['onset_ms']-m[0]['input_end_ms'],240)
        self.assertGreater(m[0]['confirmation_end_ms'],m[0]['future_end_ms'])
        d.loc[30:,FEATURES]=999
        np.testing.assert_array_equal(build_windows(d,[1])[0][0],x[0])

    def test_exact_250ms_is_included_and_later_onset_excluded(self):
        d=self.data(); d.loc[44,'CurrentTime']=d.loc[29,'CurrentTime']+250
        d.loc[44:46,'Label']='Oversteer'
        self.assertEqual(build_windows(d,[1])[1][0],2)
        d['Label']='Neutral'; d.loc[45:47,'Label']='Oversteer'
        self.assertEqual(build_windows(d,[1])[1][0],0)

    def test_confirmation_cannot_bridge_gap_or_lap_end(self):
        d=self.data(); d.loc[44:46,'Label']='Understeer'; d.loc[45:,'CurrentTime']+=100
        with self.assertRaises(ValueError): build_windows(d,[1])
        short=self.data().iloc[:46]
        with self.assertRaises(ValueError): build_windows(short.iloc[:31],[1])

    def test_last_frame_model_is_independent_of_history(self):
        torch.manual_seed(42); model=LastFrameMLP().eval(); x=torch.randn(3,30,12)
        altered=x.clone(); altered[:,:-1]=100*torch.randn(3,29,12)
        torch.testing.assert_close(model(x),model(altered),rtol=0,atol=0)

    def test_archive_output_is_rejected(self):
        with self.assertRaises(ValueError): writable_output('reference_results/2026-10-14')
        with self.assertRaises(ValueError): writable_output('reference_results/2026-10-14/new')

    def test_event_deduplication_and_false_warning_episodes(self):
        raw=self.data(); raw.loc[40:42,'Label']='Understeer'
        samples=pd.DataFrame({'lap':[1]*5,'input_end_ms':[464,480,496,512,528],
                              'onset_ms':[640,640,np.nan,np.nan,np.nan],'target':[1,1,0,0,0]})
        rows,events,episodes,_=evaluate_events(samples,[1,1,0,2,2],raw,[1])
        self.assertEqual(len(events),1); self.assertTrue(events.detected.iloc[0])
        self.assertEqual(len(episodes),2)
        self.assertEqual(rows[0]['detected_onsets'],1)
        self.assertEqual(rows[1]['false_warning_episodes'],1)

if __name__=='__main__': unittest.main()
