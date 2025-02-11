# arch-evaluating.py retrieves the best performing trials on the Pareto front from an Optuna study and evaluates them

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
import optuna
import shutil

from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model

from utils import IsValidFile, IsReadableDir, CreateFolder, predict_single_image
from drawing import Draw
from generator import RegionETGenerator
from cicada_evaluating import loss
from arch import get_data, get_data_npy

def main(args):

    def search_plots():
        aucs = np.load(f'arch/{args.name}/aucs.npy')
        if os.path.isfile(f'arch/{args.name}/std_aucs.npy'):
            std_aucs = np.load(f'arch/{args.name}/std_aucs.npy')
        else:
            std_aucs = np.zeros(aucs.shape[0])
        sizes = np.load(f'arch/{args.name}/sizes.npy')
        std_sizes = np.zeros(sizes.shape[0])
        val_losses = np.load(f'arch/{args.name}/val_losses.npy')
        if os.path.isfile(f'arch/{args.name}/std_val_losses.npy'):
            std_val_losses = np.load(f'arch/{args.name}/std_val_losses.npy')
        else:
            std_val_losses = np.zeros(val_losses.shape[0])

        to_enumerate = ['CV1 (search)', 'CV2 (search)']

        draw.plot_study_2d(sizes, aucs, yerr=std_aucs, xlabel='Model Size', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, name=f'{args.name}-scatter-size-auc')
        draw.plot_study_2d(sizes, val_losses, yerr=std_val_losses, xlabel='Model Size', ylabel='Validation Loss', to_enumerate=to_enumerate, name=f'{args.name}-scatter-size-loss')
        draw.plot_study_2d(val_losses, aucs, xerr=std_val_losses, yerr=std_aucs, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, name=f'{args.name}-scatter-loss-auc')
        draw.plot_study_3d(val_losses, aucs, sizes, xerr=std_val_losses, yerr=std_aucs, zerr=std_sizes, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', zlabel='Model Size', to_enumerate=to_enumerate, name=f'{args.name}-scatter-loss-auc-size')

    def evaluate_teacher(teacher):
        aucs, sizes, val_losses = [], [], []
        log = pd.read_csv(f"arch/{args.name}/models/{teacher.name}/training.log")
        draw.plot_loss_history(
            log["loss"], log["val_loss"], f"training-history-{teacher.name}"
        )

        y_pred_background_teacher = teacher.predict(X_test, batch_size=512, verbose=args.verbose)
        y_loss_background_teacher = loss(X_test, y_pred_background_teacher)

        teacher_results = dict()
        teacher_results["2024 Zero Bias"] = y_loss_background_teacher

        y_true = []
        y_pred_teacher = []
        inputs = []
        for name, data in X_signal.items():
            inputs.append(np.concatenate((data, X_test)))

            y_loss_signal_teacher = loss(
                data, teacher.predict(data, batch_size=512, verbose=args.verbose)
            )

            teacher_results[name] = y_loss_signal_teacher

            y_true.append(
                np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
            )
            y_pred_teacher.append(
                np.concatenate((y_loss_signal_teacher, y_loss_background_teacher))
            )

        draw.plot_anomaly_score_distribution(
            list(teacher_results.values()),
            [*teacher_results],
            f"anomaly-score-{teacher.name}",
        )
        draw.plot_roc_curve(y_true, y_pred_teacher, [*X_signal], inputs, f"roc-{teacher.name}")
        roc_aucs, _ = draw.get_aucs(y_true, y_pred_teacher, use_cut_rate=True)
        aucs.append(np.mean(roc_aucs))
        sizes.append(1000)
        val_losses.append(log["val_loss"].to_numpy()[-1])

        return aucs, sizes, val_losses

    def evaluate_students(student_models):
        aucs, sizes, val_losses = [], [], []
        for student in student_models:
            log = pd.read_csv(f"arch/{args.name}/models/{student.name}/training.log")
            draw.plot_loss_history(
                log["loss"], log["val_loss"], f"training-history-{student.name}"
            )
            
            y_loss_background_student = student.predict(
                X_test.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
            )

            student_results = dict()
            student_results["2024 Zero Bias"] = y_loss_background_student

            y_true = []
            y_pred_student = []
            inputs = []
            for name, data in X_signal.items():
                inputs.append(np.concatenate((data, X_test)))

                y_loss_signal_student = student.predict(
                    data.reshape(-1, 252, 1), batch_size=512, verbose=args.verbose
                )

                student_results[name] = y_loss_signal_student

                y_true.append(
                    np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
                )

                y_pred_student.append(
                    np.concatenate((y_loss_signal_student, y_loss_background_student))
                )
        
            draw.plot_anomaly_score_distribution(
                list(student_results.values()),
                [*student_results],
                f"anomaly-score-{student.name}",
            )

            draw.plot_roc_curve(y_true, y_pred_student, [*X_signal], inputs, f"roc-{student.name}")
            roc_aucs, _ = draw.get_aucs(y_true, y_pred_student, use_cut_rate=True)
            aucs.append(np.mean(roc_aucs))
            sizes.append(student.count_params())
            val_losses.append(log["val_loss"].to_numpy()[-1])

        return aucs, sizes, val_losses

    # Load data
    for fromname, toname in [
        ['teacher', f'arch/{args.name}/models/teacher'], 
        ['cicada-v1', f'arch/{args.name}/models/cicada-v1'], 
        ['cicada-v2', f'arch/{args.name}/models/cicada-v2'], 
    ]:
        if not os.path.isdir(toname):
            shutil.copytree(f'{args.input}/{fromname}', toname)
    config = yaml.safe_load(open(args.config))

    draw = Draw(output_dir=f'arch/{args.name}/plots/', interactive=args.interactive)

    datasets = [i["path"] for i in config["evaluation"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()
    X_train, X_val, X_test = gen.get_data_split(datasets)
    X_test = X_test[:1000000] # Will get killed otherwise; too much data
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)

    # Load study and best trials
    loaded_study = optuna.load_study(study_name=args.name, storage=f"sqlite:///arch/{args.name}/{args.name}.db")
    pareto_trials = loaded_study.best_trials
    pareto_params = [trial.params for trial in pareto_trials]
    trial_models = []
    for params in pareto_params:
        model_name = '_'.join(str(x) + '_' + str(y) for x, y in params.items())
        trial_models.append(load_model(f'arch/{args.name}/models/{model_name}'))

    # Load models
    #shutil.copytree(f'{args.inputs}', f'arch/models')
    teacher = load_model(f"arch/{args.name}/models/teacher")
    cicada_v1 = load_model(f"arch/{args.name}/models/cicada-v1")
    cicada_v2 = load_model(f"arch/{args.name}/models/cicada-v2")
    student_models = [cicada_v1, cicada_v2] + trial_models

    # Generate plots for search
    search_plots()

    # Evaluate teacher
    aucs_teacher, sizes_teacher, val_losses_teacher = evaluate_teacher(teacher)
    
    # Evaluate students
    aucs_students, sizes_students, val_losses_students = evaluate_students(student_models)

    aucs = aucs_teacher + aucs_students
    sizes = sizes_teacher + sizes_students
    val_losses = val_losses_teacher + val_losses_students
    to_enumerate = ['Teacher', 'CV1', 'CV2', 'CV1 (search)', 'CV2 (search)']

    draw.plot_study_2d(sizes, aucs, xlabel='Model Size', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, name=f'{args.name}-scatter-size-auc-all')
    draw.plot_study_2d(sizes, val_losses, xlabel='Model Size', ylabel='Validation Loss', to_enumerate=to_enumerate, name=f'{args.name}-scatter-size-loss-all')
    draw.plot_study_2d(val_losses, aucs, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, name=f'{args.name}-scatter-loss-auc-all')
    draw.plot_study_3d(val_losses, aucs, sizes, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', zlabel='Model Size', to_enumerate=to_enumerate, name=f'{args.name}-scatter-loss-auc-size-all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""CICADA evaluation scripts""")
    parser.add_argument(
        "--input", "-i",
        action=IsReadableDir,
        type=Path,
        default="models/",
        help="Path to directory w/ trained models",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="example",
        help="Name of study to be loaded",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "--config",
        "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())
