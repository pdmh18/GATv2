import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.models.batching import TemporalBatches


class BatchTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        class Fake:
            arrays={'train_mask':np.asarray([1,1,1,0,0],dtype=bool),
                    'validation_mask':np.asarray([0,0,0,1,0],dtype=bool),
                    'test_mask':np.asarray([0,0,0,0,1],dtype=bool),
                    'request_timestamps_us':np.asarray([0,10,20,30,40])*1000000,
                    'target_edge_index':np.asarray([[1,1,2,1,1],[2,2,1,2,2]]),
                    'transaction_features':np.ones((5,3),dtype='float32'),
                    'labels':np.asarray([0,1,0,1,1],dtype='uint8')}
            calls=[]
            def context(self,i):
                self.calls.append(i)
                return {'x':np.ones((3,2),dtype='float32'),'edge_index':np.empty((2,0),dtype='int64'),
                        'node_available_mask':np.asarray([True,True,False]),
                        'timestamp_us':int(self.arrays['request_timestamps_us'][i])}
        self.data=Fake()

    def test_first_cutoff_reused_for_microbatches_and_new_node_maps_unknown(self):
        batches=TemporalBatches(self.data,3600,2,self.temp.name)
        train=list(batches.iterate('train'))
        self.assertEqual([b['cutoff_us'] for b in train],[0,0])
        self.assertEqual(self.data.calls,[0])
        self.assertTrue(self.np.all(train[0]['target_edge_index'][1]==0))
        list(batches.iterate('train'))
        self.assertEqual(self.data.calls,[0])
        validation=list(batches.iterate('validation'))
        self.assertEqual(validation[0]['cutoff_us'],30000000)
        self.assertEqual(self.data.calls,[0,3])

    def test_test_access_requires_explicit_evaluation_gate(self):
        batches=TemporalBatches(self.data,3600,2,self.temp.name,allow_test=True)
        result=list(batches.iterate('test'))
        self.assertEqual(result[0]['indices'].tolist(),[4])

    def test_test_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'train/validation'):
            list(TemporalBatches(self.data,3600,2,self.temp.name).iterate('test'))

    def test_cache_rejects_policy_change_and_corruption(self):
        batches=TemporalBatches(self.data,3600,2,self.temp.name)
        list(batches.iterate('train'))
        with self.assertRaisesRegex(ValueError,'another feature artifact or policy'):
            TemporalBatches(self.data,600,2,self.temp.name)
        with (Path(self.temp.name)/'000000000.npz').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):
            list(batches.iterate('train'))


class TrainerTests(unittest.TestCase):
    def test_skip_training_config_persists_architecture(self):
        from fraud_pipeline.features.export import export_features
        from fraud_pipeline.ingestion.request_loader import file_hash
        from fraud_pipeline.models.trainer import train_run
        from fraud_pipeline.registry.save_load import load
        from fraud_pipeline.split_export import export_splits
        from fraud_pipeline.validation import validate_dataset
        from test_dataset_validation import fixture
        config=json.loads((ROOT/'configs/dataset.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir();fixture(raw)
            report=root/'validation.json';report.write_text(json.dumps(validate_dataset(raw)))
            export_splits(raw,report,root/'splits',config['split'])
            export_features(raw,report,root/'splits',root/'features',config['features'])
            training=dict(config['training'],model_kind='gatv2_skip',epochs=1,
                          model={'hidden_dim':8,'heads':2,'num_layers':2,'dropout':0.})
            summary=train_run(root/'features',root/'run',training)
            self.assertEqual(summary['model_config']['kind'],'gatv2_skip')
            model,checkpoint=load(root/'run/best.pt',
                expected_feature_sha256=file_hash(root/'features/manifest.json'))
            self.assertTrue(model.use_skip_connection)
            self.assertEqual(checkpoint['model_config']['kind'],'gatv2_skip')
            self.assertTrue(summary['checkpoint_reload_verified'])

    def test_real_training_checkpoint_reload_and_artifact_identity(self):
        import numpy as np
        from fraud_pipeline.features.export import export_features
        from fraud_pipeline.ingestion.request_loader import file_hash
        from fraud_pipeline.models.trainer import train_run
        from fraud_pipeline.registry.save_load import load
        from fraud_pipeline.split_export import export_splits
        from fraud_pipeline.validation import validate_dataset
        from test_dataset_validation import fixture
        config=json.loads((ROOT/'configs/dataset.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir();fixture(raw)
            report=root/'validation.json';report.write_text(json.dumps(validate_dataset(raw)))
            export_splits(raw,report,root/'splits',config['split'])
            export_features(raw,report,root/'splits',root/'features',config['features'])
            training=dict(config['training'],epochs=1,model={'hidden_dim':8,'heads':2,'num_layers':2,'dropout':0.})
            result=train_run(root/'features',root/'run',training)
            self.assertTrue(result['checkpoint_reload_verified'])
            self.assertFalse(result['test_evaluated'])
            model,checkpoint=load(root/'run/best.pt',expected_feature_sha256=file_hash(root/'features/manifest.json'))
            self.assertEqual(model.edge_dim,len(json.loads((root/'features/manifest.json').read_text())['feature_names']))
            self.assertTrue(checkpoint['optimizer_state_dict']['state'])
            again=train_run(root/'features',root/'run2',training)
            with np.load(root/'run/validation_predictions.npz') as first,np.load(root/'run2/validation_predictions.npz') as second:
                self.assertTrue(np.array_equal(first['scores'],second['scores']))
            with self.assertRaisesRegex(ValueError,'different feature'):
                load(root/'run/best.pt',expected_feature_sha256='wrong')
            with self.assertRaisesRegex(ValueError,'run exists'):
                train_run(root/'features',root/'run',training)
            from fraud_pipeline.pipeline import run_dataset_training
            config['dataset'].update(source=str(raw),validation_report=str(report),output=str(root/'splits'),features_output=str(root/'features'))
            config['training']=training
            path=root/'config.json';path.write_text(json.dumps(config))
            original=file_hash(root/'splits/manifest.json')
            complete=run_dataset_training(str(path),output=root/'all-training')
            self.assertTrue(complete['checkpoint_reload_verified'])
            self.assertEqual(original,file_hash(root/'splits/manifest.json'))
            config['features']['graph_window_seconds']+=1
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'different dataset or policy'):
                run_dataset_training(str(path),output=root/'stale-run')


if __name__=='__main__':
    unittest.main()
