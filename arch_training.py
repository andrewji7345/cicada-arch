# arch-training.py retrieves the best performing trials on the Pareto front from an Optuna study and further trains it

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import yaml
import optuna

from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from utils import IsValidFile, CreateFolder
from generator import RegionETGenerator
from models import TeacherAutoencoder, CicadaV1, CicadaV2, CNN_Trial, ViT_Trial
from cicada_training import loss, quantize, get_student_targets, train_model
from arch import get_data, get_data_npy, get_targets_from_teacher, get_targets_from_npy

def main(args) -> None:

    # Load data, get student targets
    config = yaml.safe_load(open(args.config))

    #gen, X_train, X_val, X_test, X_signal = get_data(config)
    gen, X_train, y_train, X_val, y_val, X_test, X_signal = get_data_npy(config)

    #gen_train, gen_val = get_targets_from_teacher(gen, X_train, X_val)
    gen_train, gen_val = get_targets_from_npy(gen, X_train, y_train, X_val, y_val)

    # Load study and best trials parameters
    loaded_study = optuna.load_study(study_name=args.name, storage=f"sqlite:///arch/{args.name}/{args.name}.db")
    pareto_trials = loaded_study.best_trials
    pareto_params = [trial.params for trial in pareto_trials]
    
    # Train best trials
    for params in tqdm(pareto_params):
        if args.type == 'cnn':
            model = CNN_Trial((252,)).get_model(params)
        elif args.type == 'vit':
            model = ViT_Trial((252,)).get_model(params)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
        model._name = '_'.join(str(x) + '_' + str(y) for x, y in params.items())
        mc = ModelCheckpoint(f"arch/{args.name}/models/{model.name}", save_best_only=True)
        log = CSVLogger(f"arch/{args.name}/models/{model.name}/training.log", append=True)

        for epoch in tqdm(range(args.epochs)):
            train_model(
                model,
                gen_train,
                gen_val,
                epoch=epoch,
                steps=1,
                callbacks=[mc, log],
                verbose=args.verbose,
            )


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
        help="Name of study to be loaded",
    )
    parser.add_argument(
        "-y", "--type",
        type=str,
        default="cnn",
        help="Type of model. Either cnn or vit",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs",
        default=100,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())
