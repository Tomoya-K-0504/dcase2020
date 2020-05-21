import argparse
import itertools
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from librosa.core import load
from ml.preprocess.preprocessor import Preprocessor
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import BaseExperimentor, typical_train, base_expt_args, get_metric_list
from ml.utils.notify_slack import notify_slack
from ml.utils.utils import dump_dict
from tqdm import tqdm

LABEL2INT = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic', 'tram',
             'bus', 'metro', 'park']
LABEL2INT = {key: i for i, key in enumerate(LABEL2INT)}


def mask_expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("DCASE2020 Experiment arguments")
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--only-test', action='store_true')
    expt_parser.add_argument('--mlflow', action='store_true')
    expt_parser.add_argument('--notify-slack', action='store_true')

    return parser


def label_func(row):
    return LABEL2INT[row[1]]


# def set_load_func(data_dir, sr, n_waves):
#     const_length = sr * 1
#
#     def one_wave_load_func(path):
#         wave = load(f'{data_dir}/{path[0]}', sr=sr)[0]
#
#         assert wave.shape[0] == const_length, f'{wave.shape[0]}, {const_length}'
#         return wave.reshape((1, -1))
#
#     if n_waves == 1:
#         return one_wave_load_func
#     elif n_waves > 1:
#         return multi_waves_load_func
#     else:
#         raise NotImplementedError


def load_func(path):
    wave = load(f'{path[0]}', sr=44100)[0]
    assert wave.shape[0] / 44100 == 10
    return wave[None, :]


class LoadDataSet(ManifestWaveDataSet):
    def __init__(self, manifest_path, data_conf, phase='train', load_func=None, transform=None, label_func=None):
        super(LoadDataSet, self).__init__(manifest_path, data_conf, phase, load_func, transform, label_func)

    def __getitem__(self, idx):
        try:
            path = Path(self.path_df.iloc[idx, 0])
            x = torch.load(str(path.parents[2] / 'processed' / path.name.replace('.wav', '.pt')))
        except FileNotFoundError as e:
            print(e)
            return super().__getitem__(idx)
        # print(x.size())
        label = self.labels[idx]

        return x, label


def parallel_logmel(expt_conf, load_func, label_func, phases):
    def parallel_preprocess(dataset, idx):
        processed, _ = dataset[idx]
        path = Path(dataset.path_df.iloc[idx, 0])
        torch.save(processed.to('cpu'), str(path.parents[2] / 'processed' / path.name.replace('.wav', '.pt')))

    for phase in tqdm(phases):
        process_func = Preprocessor(expt_conf, phase).preprocess
        dataset = ManifestWaveDataSet(expt_conf[f'{phase}_path'], expt_conf, phase, load_func, process_func,
                                      label_func)
        Parallel(n_jobs=8, verbose=0)(
            [delayed(parallel_preprocess)(dataset, idx) for idx in range(len(dataset))])
        print(f'{phase} done')


def main(expt_conf, hyperparameters, typical_train_func):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    expt_dir = (Path(__file__).resolve().parents[1] / 'output' / f"{expt_conf['expt_id']}")
    expt_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')
    expt_conf['log_dir'] = str(expt_dir / 'tensorboard')

    n_classes = 10
    expt_conf['class_names'] = list(range(n_classes))
    metrics_names = {'train': ['loss', 'uar'], 'val': ['loss', 'uar'], 'infer': []}

    expt_conf['sample_rate'] = 44100

    dataset_cls = LoadDataSet
    # dataset_cls = ManifestWaveDataSet
    # wav_path = Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav'
    # load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'])

    expt_conf['transform'] = hyperparameters['transform'][0]
    expt_conf['window_size'] = hyperparameters['window_size'][0]
    expt_conf['window_stride'] = hyperparameters['window_stride'][0]
    # parallel_logmel(expt_conf, load_func, label_func, ['train', 'val'])

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    groups = None

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"

        with mlflow.start_run():
            result_series, val_pred, _ = typical_train_func(expt_conf, load_func, label_func, process_func=None,
                                                            dataset_cls=dataset_cls, groups=groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    if expt_conf['only_test']:
        val_results = pd.read_csv(expt_dir / 'val_results.csv')
    else:
        # For debugging
        if expt_conf['n_parallel'] == 1:
            result_pred_list = [experiment(pattern, deepcopy(expt_conf)) for pattern in patterns]
        else:
            expt_conf['n_jobs'] = 0
            result_pred_list = Parallel(n_jobs=expt_conf['n_parallel'], verbose=0)(
                [delayed(experiment)(pattern, deepcopy(expt_conf)) for pattern in patterns])

        val_results.iloc[:, :len(hyperparameters)] = [[str(param) for param in p] for p in patterns]
        result_list = [result for result, pred in result_pred_list]
        val_results.iloc[:, len(hyperparameters):] = result_list
        pp.pprint(val_results)
        pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())

        val_results.to_csv(expt_dir / 'val_results.csv', index=False)
        print(f"Devel results saved into {expt_dir / 'val_results.csv'}")
        for (_, pred), pattern in zip(result_pred_list, patterns):
            pattern_name = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
            pd.DataFrame(pred).to_csv(expt_dir / f'{pattern_name}_val_pred.csv', index=False)
            dump_dict(expt_dir / f'{pattern_name}.txt', expt_conf)

    best_trial_idx = val_results['uar'].argmax()
    best_pattern = val_results.iloc[best_trial_idx, :].values
    print(val_results.iloc[best_trial_idx, :])
    for i, param in enumerate(val_results.columns):
        expt_conf[param] = best_pattern[i]
    dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

    # Train with train + devel dataset
    phases = ['train', 'infer']
    if expt_conf['only_test'] or expt_conf['infer']:
        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}_all.pth")

        # parallel_logmel(expt_conf, load_func, label_func, ['train', 'infer'])

        dataset_cls = LoadDataSet
        # dataset_cls = ManifestWaveDataSet

        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func=None, dataset_cls=dataset_cls)

        metrics = {p: get_metric_list(metrics_names[p]) for p in phases}
        # _, pred = experimentor.experiment_without_validation(metrics, seed_average=expt_conf['n_seed_average'])
        _, pred = experimentor._experiment(metrics, phases=['infer'])
        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}_all.pth")
        experimentor.train_manager.model_manager.save_model()

        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        if (expt_dir / sub_name).is_file():
            sub_df = pd.read_csv(expt_dir / sub_name)
        else:
            sub_df = manifest_df[manifest_df['file_name'].str.startswith('test')][['file_name']].reset_index(drop=True)

        if expt_conf['return_prob']:
            pd.DataFrame(pred['infer']).to_csv(expt_dir / f'prob_{sub_name}', index=False)
            pred['infer'] = np.argmax(pred['infer'], axis=1)

        sub_df['prediction'] = pd.Series(pred['infer']).apply(lambda x: list(LABEL2INT.keys())[x])
        sub_df.to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")


def create_manifest(expt_conf):
    data_path = Path(expt_conf[f'train_path']).parents[1]

    # TODO phase=testも追加
    for phase in ['train', 'val']:
        manifest = pd.read_csv(expt_conf[f'{phase}_path'], sep='\t')
        manifest['filename'] = manifest['filename'].apply(lambda x: f'{data_path}/{x}')
        manifest.to_csv(expt_conf[f'{phase}_path'].replace('.csv', '') + '_local.csv', index=False, header=None)
        expt_conf[f'{phase}_path'] = expt_conf[f'{phase}_path'].replace('.csv', '') + '_local.csv'

    return expt_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask experiment arguments')
    expt_conf = vars(mask_expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if not expt_conf['mlflow']:
        hyperparameters = {
            'lr': [1e-4],
            'batch_size': [2],
            'model_type': ['logmel_cnn'],
            'transform': ['logmel'],
            # 'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.06],
            'window_stride': [0.05],
            'epoch_rate': [0.05],
            'mixup_alpha': [0.0],
            'sample_balance': ['same'],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0],
        }
    else:
        hyperparameters = {
            'lr': [1e-4],
            'batch_size': [16],
            'model_type': ['panns'],
            'transform': ['logmel'],
            'kl_penalty': [0.0],
            'entropy_penalty': [0.0],
            'loss_func': ['ce'],
            'checkpoint_path': [None],
            'window_size': [0.1],
            'window_stride': [0.05],
            'epoch_rate': [1.0],
            'mixup_alpha': [0.0],
            'sample_balance': ['same'],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0]
        }

    expt_conf = create_manifest(expt_conf)

    main(expt_conf, hyperparameters, typical_train)

    if not expt_conf['mlflow']:
        import shutil

        shutil.rmtree('mlruns')

    if expt_conf['nofity_slack']:
        cfg = dict(
            body=f"Finished experiment {expt_conf['expt_id']}: \n" +
                 "Notion ticket: https://www.notion.so/DCASE-2020-010ca4ceda0f49828d2ee81b77b8e1a4",
            webhook_url='https://hooks.slack.com/services/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'
        )
        notify_slack(cfg)
