import argparse
import itertools
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from experiment import expt_args, label_func, load_func, LoadDataSet
from joblib import Parallel, delayed
from ml.tasks.gan_experiment import GANExperimentor
from ml.utils.notify_slack import notify_slack
from ml.utils.utils import dump_dict


def main(expt_conf, hyperparameters):
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

        # with mlflow.start_run():
        GANExperimentor(expt_conf, load_func, label_func, process_func=None, dataset_cls=dataset_cls)._experiment(metrics_names, phases=['train', 'val'])

            # mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            # mlflow.log_artifacts(expt_dir)

        return

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
    expt_conf = vars(expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if not expt_conf['mlflow']:
        hyperparameters = {
            'lr': [1e-4],
            'batch_size': [2],
            'model_type': ['panns_wavegram'],
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
            'batch_size': [32],
            'model_type': ['panns_wavegram'],
            'transform': ['logmel'],
            'checkpoint_path': ['cnn14.pth'],
            'kl_penalty': [0.0],
            'entropy_penalty': [0.0],
            'loss_func': ['ce'],
            'window_size': [0.1],
            'window_stride': [0.05],
            'epoch_rate': [1.0],
            'mixup_alpha': [0.0],
            'sample_balance': ['same'],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0]
        }

    expt_conf = create_manifest(expt_conf)

    main(expt_conf, hyperparameters)

    if not expt_conf['mlflow']:
        import shutil

        shutil.rmtree('mlruns')

    if expt_conf['notify_slack']:
        cfg = dict(
            body=f"Finished experiment {expt_conf['expt_id']}: \n" +
                 "Notion ticket: https://www.notion.so/DCASE-2020-010ca4ceda0f49828d2ee81b77b8e1a4",
            webhook_url='https://hooks.slack.com/services/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'
        )
        notify_slack(cfg)
