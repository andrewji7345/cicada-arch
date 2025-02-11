# arch.py creates an Optuna study to search for the best hyperparameters

import os
import logging
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
import optuna

from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, AveragePooling2D, Flatten, Input, Reshape, UpSampling2D, Conv2DTranspose
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm, quantized_bits
from tqdm import tqdm

from utils import IsValidFile, IsReadableDir, CreateFolder, predict_single_image
from drawing import Draw
from generator import RegionETGenerator
from cicada_training import loss, quantize, get_student_targets
from models import CNN_Trial, ViT_Trial

# Load data from h5 files
def get_data(config):
    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()

    X_train, X_val, X_test = gen.get_data_split(datasets)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)
    outlier_train = gen.get_data(config["exposure"]["training"])
    outlier_val = gen.get_data(config["exposure"]["validation"])

    X_train = np.concatenate([X_train, outlier_train])
    X_val = np.concatenate([X_val, outlier_val])
    X_test = X_test.reshape(-1, 18, 14, 1)
    return gen, X_train, X_val, X_test, X_signal

# Load data from npy files
def get_data_npy(config):
    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()

    X_train = gen.get_data_npy(config["training"]["inputs"])
    y_train = gen.get_targets_npy(config["training"]["targets"])
    X_val = gen.get_data_npy(config["validation"]["inputs"])
    y_val = gen.get_targets_npy(config["validation"]["targets"])
    _, _, X_test = gen.get_data_split(datasets)
    X_test = X_test.reshape(-1, 18, 14, 1)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)

    return gen, X_train, y_train, X_val, y_val, X_test, X_signal

# Get targets, given a generator and training, validation, and test data
def get_targets_from_teacher(gen, X_train, X_val):
    teacher = load_model("models/teacher")
    gen_train = get_student_targets(teacher, gen, X_train)
    gen_val = get_student_targets(teacher, gen, X_val)
    return gen_train, gen_val

# Get targets, given file
def get_targets_from_npy(gen, X_train, y_train, X_val, y_val):
    gen_train = gen.get_generator(X_train.reshape((-1, 252, 1)), y_train, 1024, True)
    gen_val = gen.get_generator(X_val.reshape((-1, 252, 1)), y_val, 1024, True)
    return gen_train, gen_val

def train_model(
    model: Model,
    gen_train: data.Dataset,
    gen_val: data.Dataset,
    epoch: int = 1,
    steps: int = 1,
    callbacks=None,
    verbose: bool = False,
) -> float:
    history = model.fit(
        gen_train,
        steps_per_epoch=len(gen_train),
        initial_epoch=epoch,
        epochs=epoch + steps,
        validation_data=gen_val,
        callbacks=callbacks,
        verbose=verbose,
    )

    final_val_loss = history.history["val_loss"][-1] if "val_loss" in history.history else None
    return final_val_loss

def main(args) -> None:

    def objective(trial):
        aucs_tr = []
        val_losses_tr = []

        for i in tqdm(range(args.executions)):
            # Compile, prune if too small
            if args.type == 'cnn':
                model = CNN_Trial((252,)).get_trial(trial)
            elif args.type == 'vit':
                model = ViT_Trial((252,)).get_trial(trial)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
            if model.count_params() < 1000:
                raise optuna.TrialPruned()

            # Train
            val_loss = train_model(
                        model,
                        gen_train,
                        gen_val,
                        epoch=args.epochs,
                        verbose=args.verbose,
            )

            # Evaluate the model on the test set
            roc_aucs, _ = get_aucs(model)
            mean_auc = np.mean(roc_aucs)
            aucs_tr.append(mean_auc)
            val_losses_tr.append(val_loss)

        med_auc = np.median(aucs_tr)
        std_auc = np.std(aucs_tr)
        size = model.count_params()
        med_val_loss = np.median(val_losses_tr)
        std_val_loss = np.std(val_losses_tr)

        print(f'Median AUC: {med_auc} +- {std_auc}')
        print(f'Size: {size}')
        print(f'Median Validation Loss: {med_val_loss} +- {std_val_loss}')

        #aucs.append(med_auc)
        #std_aucs.append(std_auc)
        #sizes.append(size)
        #val_losses.append(med_val_loss)
        #std_val_losses.append(std_val_loss)
        #names.append(f'{model.name}')

        for name, arr in [['sizes', [size]], ['aucs', [med_auc]], ['std_aucs', [std_auc]], ['val_losses', [med_val_loss]], ['std_val_losses', [std_val_loss]]]:
            if os.path.exists(f'arch/{args.name}/{name}'):
                arr_temp = np.load(f'arch/{args.name}/{name}')
                arr_temp = np.concatenate((arr_temp, arr))
                np.save(f'arch/{args.name}/{name}', arr_temp)
            else:
                np.save(f'arch/{args.name}/{name}', arr)

        return med_auc, size, med_val_loss

    def get_aucs(model):
        y_loss_background = model.predict(
            X_test.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
        )
        results = {'2023 Zero Bias (Test)' : y_loss_background}
        y_true, y_pred, inputs = [], [], []
        for name, data in X_signal.items():
            inputs.append(np.concatenate((data, X_test)))
            y_loss = model.predict(
                data.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
            )
            results[name] = y_loss
            y_true.append(
                np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
            )
            y_pred.append(
                np.concatenate((y_loss, y_loss_background))
            )

        #draw.plot_anomaly_score_distribution(list(results.values()), [*results], f"anomaly-score-{model.name}")
        #draw.plot_roc_curve(y_true, y_pred, [*X_signal], inputs, f"roc_{model.name}")
        roc_aucs, std_aucs = draw.get_aucs(y_true, y_pred, use_cut_rate=True)
        return roc_aucs, std_aucs

    # Create folders
    for foldername in ['arch/', f'arch/{args.name}/', f'arch/{args.name}/plots/', f'arch/{args.name}/models/']:
        if not os.path.isdir(foldername):
            os.mkdir(foldername)

    # Load data, get student targets
    config = yaml.safe_load(open(args.config))

    draw = Draw(output_dir=f'arch/{args.name}/plots/', interactive=args.interactive)

    #gen, X_train, X_val, X_test, X_signal = get_data(config)
    gen, X_train, y_train, X_val, y_val, X_test, X_signal = get_data_npy(config)

    #gen_train, gen_val = get_targets_from_teacher(gen, X_train, X_val)
    gen_train, gen_val = get_targets_from_npy(gen, X_train, y_train, X_val, y_val)

    # Evaluate current students
    for i in [1,2]:
        cicada = load_model(f"models/cicada-v{i}")
        cicada.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
        cicada_size = cicada.count_params()
        cicada_roc_aucs, cicada_std_aucs = get_aucs(cicada)
        cicada_mean_auc = np.mean(cicada_roc_aucs)
        cicada_mean_std = np.mean(cicada_std_aucs)
        print(f"cicada_v{i} size: {cicada_size}")
        print(f"cicada_v{i} mean AUC: {cicada_mean_auc} +_ {cicada_mean_std}")

    # Add SQLite
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_name = f"sqlite:///arch/{args.name}/{args.name}.db"

    # Optuna study
    aucs, std_aucs, sizes, val_losses, std_val_losses, names = [], [], [], [], [], []
    study = optuna.create_study(
        directions=['maximize', 'minimize', 'minimize'], 
        study_name=args.name, 
        storage=storage_name, 
        load_if_exists=True,
    )
    if args.type == 'cnn' and len(study.trials) == 0:
        study.enqueue_trial({"n_filters": 4, 
                            "n_conv_layers": 0, 
                            "n_dense_units": 16, 
                            "n_dense_layers": 1, 
        }) # Include cicada_v1
        study.enqueue_trial({"n_filters": 4, 
                            "n_conv_layers": 1, 
                            "n_dense_units": 16, 
                            "n_dense_layers": 1, 
        }) # Include cicada_v2
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Save results
    for name, arr in [['sizes', sizes], ['aucs', aucs], ['std_aucs', std_aucs], ['val_losses', val_losses], ['std_val_losses', std_val_losses]]:
        if os.path.exists(f'arch/{args.name}/{name}'):
            arr_temp = np.load(f'arch/{args.name}/{name}')
            arr_temp = np.concatenate((arr_temp, arr))
            np.save(f'arch/{args.name}/{name}', arr_temp)
        else:
            np.save(f'arch/{args.name}/{name}', arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""CICADA training scripts""")
    parser.add_argument(
        "--config", "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="example",
        help="Name of study",
    )
    parser.add_argument(
        "-y", "--type",
        type=str,
        default="cnn",
        help="Type of model. Either cnn or vit",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs per execution",
        default=10,
    )
    parser.add_argument(
        "-x", "--executions",
        type=int,
        help="Number of executions per trial",
        default=10, 
    )
    parser.add_argument(
        "-t", "--trials",
        type=int,
        help="Number of trials",
        default=10, 
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())
