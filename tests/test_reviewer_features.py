import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from fraud_pipeline.features.causal import State,names,replay
from fraud_pipeline.features.windows import parse_window,window_profile
from fraud_pipeline.features.export import export_features
from fraud_pipeline.features.dataset import TemporalDataset
from fraud_pipeline.validation import validate_dataset
from fraud_pipeline.validation.contracts import validate_event
from fraud_pipeline.validation.core import Audit
from fraud_pipeline.split_export import export_splits
from fraud_pipeline.evaluation.feature_audit import feature_groups
from test_causal_features import event
from test_dataset_validation import fixture,mutate
OPTIONS=json.loads((ROOT/'configs/features-v3.json').read_text())['features']


def values(events,options=OPTIONS):
    return [dict(zip(names(options),v)) for _,v,_ in replay(events,State(options))]


class ReviewerFeatureTests(unittest.TestCase):
    def test_completed_mean_is_typed_prior_and_windowed(self):
        options=copy.deepcopy(OPTIONS);options['mean_window_seconds']=10;options['amount_min_history']=2
        income=event(3,'TransactionPosted',amount=9999);income['payload']['transactionType']='CASH_IN'
        rows=values([event(0,amount=10),event(0,'TransferCompleted',amount=10),
                     event(2,amount=30),event(2,'TransferCompleted',amount=30),income,
                     event(4,amount=1000),event(4,'TransferCompleted',amount=1000),event(15)],options)
        self.assertAlmostEqual(rows[2]['log_src_type_completed_mean_amount'],math.log1p(20))
        self.assertEqual(rows[2]['type_mean_reference_reliable'],1)
        self.assertAlmostEqual(rows[2]['log_amount_to_type_mean'],math.log1p(1000/20))
        self.assertEqual(rows[-1]['has_type_mean_history'],0)
        self.assertGreater(rows[-1]['log_src_type_completed_count'],0) # Median 30d still has history.

    def test_device_novelty_uses_prior_requests_and_missing_is_distinct(self):
        first=event(0);first['payload'].update(device_id='D1',authentication_method='OTP',authentication_verified=False,platform='ANDROID')
        simultaneous=event(0);simultaneous['payload']['device_id']='D1'
        repeat=event(1);repeat['payload']['device_id']='D1'
        new=event(2);new['payload']['device_id']='D2'
        absent=event(3)
        rows=values([first,simultaneous,repeat,new,absent])
        self.assertEqual([r['new_device'] for r in rows],[1,1,0,1,0])
        self.assertEqual(rows[0]['authentication_verified'],0)
        self.assertEqual(rows[0]['has_authentication_verified'],1)
        self.assertEqual(rows[-1]['has_authentication_verified'],0)
        self.assertEqual(rows[0]['authentication_method_otp'],1)
        self.assertEqual(rows[0]['platform_android'],1)
        self.assertEqual(rows[-1]['has_device'],0)
        self.assertEqual(rows[-1]['has_device_history'],1)

    def test_context_rejects_numeric_device_and_string_verified(self):
        for key,value in [('device_id',123),('authentication_verified','false')]:
            row=event(0);row['payload'][key]=value
            self.assertRaises(ValueError,State(OPTIONS).score,row)

    def test_window_aliases_profiles_and_legacy_compatibility(self):
        self.assertEqual(parse_window('1d'),86400);self.assertEqual(parse_window('1w'),parse_window('7d'))
        self.assertEqual(parse_window('1mo'),2592000)
        for invalid in ('0d','-1d','1month','1.5d','1'):
            self.assertRaises(ValueError,parse_window,invalid)
        config=json.loads((ROOT/'configs/features-v3.json').read_text())
        profile=window_profile(config,window='1w',graph_window='1d')
        self.assertEqual(config['features']['graph_window_seconds'],86400)
        self.assertEqual(profile['features']['mean_window_seconds'],604800)
        self.assertEqual(profile['features']['amount_window_seconds'],604800)
        self.assertEqual(profile['features']['graph_window_seconds'],86400)
        self.assertNotEqual(config['training']['output'],profile['training']['output'])
        old=json.loads((ROOT/'configs/dataset.json').read_text())['features']
        self.assertEqual(len(names(old)),35)
        groups=feature_groups(names(OPTIONS))
        self.assertEqual(sorted(i for cols in groups.values() for i in cols),list(range(len(names(OPTIONS)))))
        self.assertIn('authentication',groups);self.assertIn('device_context',groups)

    def test_snapshot_device_must_match_request(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder);fixture(source)
            def snapshot(rows):
                for row in rows:
                    if row['type']=='CASH_IN':row['device']={'device_id':'D1','platform':'android'}
            def request(rows):
                for row in rows:
                    if row['event_type']=='TransactionRequested' and row['payload']['transactionType']=='CASH_IN':
                        row['payload'].update(device_id='D1',platform='android')
            mutate(source,'transactions',snapshot);mutate(source,'events',request)
            result=validate_dataset(source);self.assertEqual(result['error_count'],0,result['issues'])
            def corrupt(rows):
                for row in rows:
                    if row['type']=='CASH_IN':row['device']['device_id']='D2'
            mutate(source,'transactions',corrupt)
            self.assertIn('device_snapshot_match',{i['code'] for i in validate_dataset(source)['issues']})

    def test_v3_fixture_export_preserves_request_context_and_no_future_posted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'raw';source.mkdir();fixture(source)
            def enrich(rows):
                for row in rows:
                    if row['event_type']=='TransactionRequested' and row['payload']['transactionType'] in ('CASH_IN','PAYMENT'):
                        row['payload'].update(device_id='D1',authentication_method='OTP',authentication_verified=False,platform='ANDROID')
            mutate(source,'events',enrich)
            result=validate_dataset(source)
            self.assertEqual(result['error_count'],0,result)
            report=root/'validation.json';report.write_text(json.dumps(result))
            export_splits(source,report,root/'splits',dict(train=.5,val=.3,test=.2,time_based=True))
            manifest=export_features(source,report,root/'splits',root/'features',OPTIONS)
            self.assertEqual(manifest['schema_version'],3)
            data=TemporalDataset(root/'features')
            column=manifest['feature_names'].index('new_device')
            self.assertEqual(data.arrays['transaction_features'][1,column],1)
            self.assertEqual(data.arrays['transaction_features'][3,column],0)
            request=json.loads((source/'events.jsonl').read_text().splitlines()[-2])
            request['payload']['posted_at']=request['occurred_at'];audit=Audit()
            self.assertFalse(validate_event(audit,request,'test'))

if __name__=='__main__':unittest.main()
