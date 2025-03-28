# arch.py creates an Optuna study to search for the best hyperparameters

import os
import logging
import sys
import io
import re
import shlex
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
import optuna
import tensorflow as tf

from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, AveragePooling2D, Flatten, Input, Reshape, UpSampling2D, Conv2DTranspose
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm
from tqdm import tqdm
from larq.models import summary

from utils import IsValidFile, IsReadableDir, CreateFolder, save_to_npy, predict_single_image, save_args
from drawing import Draw
from generator import RegionETGenerator
from cicada_training import loss, quantize, get_student_targets
from models import CNN_Trial, Binary_Trial, Bit_Binary_Trial

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
    datasets = [i["path"] for i in config["evaluation"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()

    X_train = gen.get_data_npy(config["training"]["inputs"])
    y_train = gen.get_targets_npy(config["training"]["targets"])
    X_val = gen.get_data_npy(config["validation"]["inputs"])
    y_val = gen.get_targets_npy(config["validation"]["targets"])
    _, _, X_test = gen.get_data_split(datasets)
    X_test = X_test.reshape(-1, 18, 14, 1)
    X_test = X_test[:250000] # Will get killed otherwise; too much data
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)

    return gen, X_train, y_train, X_val, y_val, X_test, X_signal

# Get targets, given a generator and training, validation, and test data
def get_targets_from_teacher(gen, X_train, X_val):
    teacher = load_model("models/teacher")
    gen_train = get_student_targets(teacher, gen, X_train)
    gen_val = get_student_targets(teacher, gen, X_val)
    return gen_train, gen_val

# Get targets, given a generator and training, validation, and test data
def get_targets_from_npy(gen, X_train, y_train, X_val, y_val):
    gen_train = gen.get_generator(X_train.reshape((-1, 252, 1)), y_train, 1024, True)
    gen_val = gen.get_generator(X_val.reshape((-1, 252, 1)), y_val, 1024, True)
    return gen_train, gen_val

def train_model(
    model: Model,
    gen_train: data.Dataset,
    gen_val: data.Dataset,
    epochs: int = 1,
    callbacks=None,
    shuffle: bool = False, 
    verbose: bool = False,
) -> float:
    history = model.fit(
        gen_train,
        steps_per_epoch=len(gen_train),
        epochs=epochs,
        validation_data=gen_val,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle, 
    )
    return history

def main(args) -> None:

    start_time = time.time()
    max_time = 4 * 0.8 * args.epochs * args.executions

    def objective(trial, trial_id):
        to_save = { 
            'Mean Signal AUC (0.3-3 kHz)': np.array([]), 
            f'{labels[0]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[1]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[2]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[3]} AUC (0.3-3 kHz)': np.array([]),
            f'{labels[4]} AUC (0.3-3 kHz)': np.array([]),
            'Validation Loss': np.array([]), 
            'Model Size (number of parameters)': np.array([]), 
            'Model Size (b)': np.array([]), 
        }

        for i in tqdm(range(args.executions)):
            if (time.time()-start_time) > max_time: break
            execution_id = i
            
            # Compile
            if args.type == 'cnn':
                model_gen = CNN_Trial((252,), execution_id)
                model, size_b = model_gen.get_trial(trial)
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
                if size_b < 50000 or size_b > 100000: # Prune if too small or too large
                    raise optuna.TrialPruned()
            elif args.type == 'bitbnn':
                model = Bit_Binary_Trial((252,), args.type, execution_id).get_trial(trial)
                model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
            elif args.type[0] == 'b':
                model = Binary_Trial((252,), args.type, execution_id).get_trial(trial)
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
            es = EarlyStopping(monitor='val_loss', patience=3, baseline=10, start_from_epoch=10)
            log = CSVLogger(f"arch/{args.name}/models/{trial_id}-{execution_id}-training.log", append=True)

            # Train
            history = train_model(
                model,
                gen_train,
                gen_val,
                epochs=args.epochs,
                callbacks=[es, log], 
                verbose=args.verbose,
                shuffle=True, 
            )
            log = pd.read_csv(f"arch/{args.name}/models/{trial_id}-{execution_id}-training.log")
            draw_execution.plot_loss_history(log["loss"], log["val_loss"], f"training-history-{trial_id}-{execution_id}")
            
            # Evaluate the model on the test set
            auc, _ = get_aucs(model, trial_id, execution_id)
            mean_auc = np.mean(auc)
            n_params = model.count_params()

            for name, val in [
                ['Mean Signal AUC (0.3-3 kHz)', mean_auc], 
                [f'{labels[0]} AUC (0.3-3 kHz)', auc[0]], 
                [f'{labels[1]} AUC (0.3-3 kHz)', auc[1]], 
                [f'{labels[2]} AUC (0.3-3 kHz)', auc[2]], 
                [f'{labels[3]} AUC (0.3-3 kHz)', auc[3]], 
                [f'{labels[4]} AUC (0.3-3 kHz)', auc[4]], 
                ['Validation Loss', history.history["val_loss"][-1]], 
                ['Model Size (number of parameters)', n_params], 
                ['Model Size (b)', size_b], 
                ]:

                to_save[name] = np.append(to_save[name], val)
                pathname = f'arch/{args.name}/trial_metrics/{name}/{trial_id}.npy'
                save_to_npy(val, pathname)

        max_mean_auc = np.max(to_save["Mean Signal AUC (0.3-3 kHz)"])
        med_mean_auc = np.median(to_save["Mean Signal AUC (0.3-3 kHz)"])
        std_mean_auc = np.std(to_save["Mean Signal AUC (0.3-3 kHz)"])
        max_auc_0 = np.max(to_save[f"{labels[0]} AUC (0.3-3 kHz)"])
        med_auc_0 = np.median(to_save[f"{labels[0]} AUC (0.3-3 kHz)"])
        std_auc_0 = np.std(to_save[f"{labels[0]} AUC (0.3-3 kHz)"])
        max_auc_1 = np.max(to_save[f"{labels[1]} AUC (0.3-3 kHz)"])
        med_auc_1 = np.median(to_save[f"{labels[1]} AUC (0.3-3 kHz)"])
        std_auc_1 = np.std(to_save[f"{labels[1]} AUC (0.3-3 kHz)"])
        max_auc_2 = np.max(to_save[f"{labels[2]} AUC (0.3-3 kHz)"])
        med_auc_2 = np.median(to_save[f"{labels[2]} AUC (0.3-3 kHz)"])
        std_auc_2 = np.std(to_save[f"{labels[2]} AUC (0.3-3 kHz)"])
        max_auc_3 = np.max(to_save[f"{labels[3]} AUC (0.3-3 kHz)"])
        med_auc_3 = np.median(to_save[f"{labels[3]} AUC (0.3-3 kHz)"])
        std_auc_3 = np.std(to_save[f"{labels[3]} AUC (0.3-3 kHz)"])
        max_auc_4 = np.max(to_save[f"{labels[4]} AUC (0.3-3 kHz)"])
        med_auc_4 = np.median(to_save[f"{labels[4]} AUC (0.3-3 kHz)"])
        std_auc_4 = np.std(to_save[f"{labels[4]} AUC (0.3-3 kHz)"])
        min_val_loss = np.min(to_save["Validation Loss"])
        med_val_loss = np.median(to_save["Validation Loss"])
        std_val_loss = np.std(to_save["Validation Loss"])
        n_params = np.median(to_save["Model Size (number of parameters)"])
        size_b = np.median(to_save["Model Size (b)"])

        for name, arr in [
            ['Max of Mean Signal AUCs (0.3-3 kHz)', [max_mean_auc]], 
            ['Median of Mean Signal AUCs (0.3-3 kHz)', [med_mean_auc]], 
            ['Standard Deviation of Mean Signal AUCs (0.3-3 kHz)', [std_mean_auc]], 
            [f'Max of {labels[0]} AUCs (0.3-3 kHz)', [max_auc_0]], 
            [f'Median of {labels[0]} AUCs (0.3-3 kHz)', [med_auc_0]], 
            [f'Standard Deviation of {labels[0]} AUCs (0.3-3 kHz)', [std_auc_0]], 
            [f'Max of {labels[1]} AUCs (0.3-3 kHz)', [max_auc_1]], 
            [f'Median of {labels[1]} AUCs (0.3-3 kHz)', [med_auc_1]], 
            [f'Standard Deviation of {labels[1]} AUCs (0.3-3 kHz)', [std_auc_1]], 
            [f'Max of {labels[2]} AUCs (0.3-3 kHz)', [max_auc_2]], 
            [f'Median of {labels[2]} AUCs (0.3-3 kHz)', [med_auc_2]], 
            [f'Standard Deviation of {labels[2]} AUCs (0.3-3 kHz)', [std_auc_2]], 
            [f'Max of {labels[3]} AUCs (0.3-3 kHz)', [max_auc_3]], 
            [f'Median of {labels[3]} AUCs (0.3-3 kHz)', [med_auc_3]], 
            [f'Standard Deviation of {labels[3]} AUCs (0.3-3 kHz)', [std_auc_3]], 
            [f'Max of {labels[4]} AUCs (0.3-3 kHz)', [max_auc_4]], 
            [f'Median of {labels[4]} AUCs (0.3-3 kHz)', [med_auc_4]], 
            [f'Standard Deviation of {labels[4]} AUCs (0.3-3 kHz)', [std_auc_4]], 
            ['Min of Validation Losses', [min_val_loss]], 
            ['Median of Validation Losses', [med_val_loss]], 
            ['Standard Deviation of Validation Losses', [std_val_loss]], 
            ['Model Size (number of parameters)', [n_params]], 
            ['Model Size (b)', [size_b]]
            ]:

            pathname = f'arch/{args.name}/study_metrics/{name}.npy'
            save_to_npy(arr, pathname)

        print(f'Med Mean AUC: {med_mean_auc} +- {std_mean_auc}; Max Mean AUC: {max_mean_auc}')
        print(f'Med Val Loss: {med_val_loss} +- {std_val_loss}; Min Val Loss: {min_val_loss}')
        print(f'n_params: {n_params}')
        print(f'size_b: {size_b}')
        print(f'Total time: {time.time()-start_time}\nTime per execution: {(time.time()-start_time)/args.executions}\nTime per trial: {(time.time()-start_time)/args.executions/args.epochs}')

        return max_mean_auc, min_val_loss, n_params, size_b

    def get_aucs(model, trial_id, execution_id):
        #with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
        y_loss_background = model.predict(X_test.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose)
        y_loss_background = np.nan_to_num(y_loss_background, nan=0.0, posinf=255, neginf=0)
        results = {'2023 Zero Bias' : y_loss_background}
        y_true, y_pred, inputs = [], [], []
        for name, data in X_signal.items():
            inputs.append(np.concatenate((data, X_test)))
            #with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
            y_loss = model.predict(data.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose)
            y_loss = np.nan_to_num(y_loss, nan=0.0, posinf=255, neginf=0)
            results[name] = y_loss
            y_true.append(
                np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
            )
            y_pred.append(
                np.concatenate((y_loss, y_loss_background))
            )

        draw_execution.plot_roc_curve(y_true, y_pred, [*X_signal], inputs, f"roc-{trial_id}-{execution_id}")
        roc_aucs, std_aucs = draw_execution.get_aucs(y_true, y_pred, use_cut_rate=True)
        return roc_aucs, std_aucs

    # Get labels
    labels = [
        'SUEP', 
        'H to Long Lived', 
        'VBHF to 2C', 
        'TT', 
        'SUSY GGBBH', 
    ]

    # Create folders
    for foldername in [
        'arch/', 
        f'arch/{args.name}/', 
        f'arch/{args.name}/study_plots/', 
        f'arch/{args.name}/trial_plots/', 
        f'arch/{args.name}/execution_plots/', 
        f'arch/{args.name}/models/', 
        f'arch/{args.name}/study_metrics/', 
        f'arch/{args.name}/trial_metrics/', 
        f'arch/{args.name}/trial_metrics/Mean Signal AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[0]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[1]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[2]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[3]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/{labels[4]} AUC (0.3-3 kHz)/', 
        f'arch/{args.name}/trial_metrics/Validation Loss/', 
        f'arch/{args.name}/trial_metrics/Model Size (number of parameters)/', 
        f'arch/{args.name}/trial_metrics/Model Size (b)/', 
        ]:
        if not os.path.exists(foldername):
            os.mkdir(foldername)

    # Load data, get student targets
    config = yaml.safe_load(open(args.config))

    draw_execution = Draw(output_dir=f'arch/{args.name}/execution_plots/', interactive=args.interactive)

    gen, X_train, y_train, X_val, y_val, X_test, X_signal = get_data_npy(config)
    gen_train, gen_val = get_targets_from_npy(gen, X_train, y_train, X_val, y_val)

    # Add SQLite
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_name = f"sqlite:///arch/{args.name}/{args.name}.db"

    # Optuna study; if parallelized, reload study after each iteration
    for i in tqdm(range(args.trials)): # for parallelization
        study = optuna.create_study(
            directions=['maximize', 'minimize', 'minimize', 'minimize'], 
            study_name=args.name, 
            storage=storage_name, 
            load_if_exists=True, 
        )
        trial_id = i
        if (args.type == 'cnn') and len(study.trials) < 2:
            study.enqueue_trial({
                "n_conv_layers": 0, 
                "n_dense_layers": 1, 
                "n_layers": 1, 
                "n_dense_units_0": 4, 
                "q_kernel_conv_bits": 12, 
                "q_kernel_conv_ints": 3, 
                "q_kernel_dense_bits": 8, 
                "q_kernel_dense_ints": 1, 
                "q_bias_dense_bits": 8, 
                "q_bias_dense_ints": 3, 
                "q_activation_bits": 10, 
                "q_activation_ints": 6, 
                "shortcut": False, 
                "dropout": 0., 
            }) # Include cicada_v1
            study.enqueue_trial({
                "n_conv_layers": 1, 
                "n_dense_layers": 1, 
                "n_layers": 2, 
                "n_filters_0": 4, 
                "kernel_width_0": 2, 
                "kernel_height_0": 2, 
                "stride_width_0": 2, 
                "stride_height_0": 2, 
                "n_dense_units_0": 4, 
                "q_kernel_conv_bits": 12, 
                "q_kernel_conv_ints": 3, 
                "q_kernel_dense_bits": 8, 
                "q_kernel_dense_ints": 1, 
                "q_bias_dense_bits": 8, 
                "q_bias_dense_ints": 3, 
                "q_activation_bits": 10, 
                "q_activation_ints": 6, 
                "shortcut": False, 
                "dropout": 0., 
            }) # Include cicada_v2
        study.optimize(lambda trial: objective(trial, trial_id), n_trials=1, show_progress_bar=False)

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
        help="Type of model. One of cnn, vit, bnn, ban, bwn, bitbnn.",
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
    args = parser.parse_args()
    save_args(args)  # Save command-line arguments
    main(args)
