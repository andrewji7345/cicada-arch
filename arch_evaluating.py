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
import shlex

from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model
from qkeras import quantized_bits

from utils import IsValidFile, IsReadableDir, CreateFolder, predict_single_image, load_args
from drawing import Draw
from generator import RegionETGenerator
from cicada_evaluating import loss
from arch import get_data, get_data_npy, get_targets_from_teacher, get_targets_from_npy

def main(args):

    def search_plots():
        all_names = [name for name in os.listdir(f'arch/{args.name}/study_metrics') if os.path.isfile(os.path.join(f'arch/{args.name}/study_metrics', name))]
        min_names = [name for name in all_names if (('AUC' not in name) and ('Min' in name))]
        med_names = [name for name in all_names if (('AUC' not in name) and ('Median' in name))]
        std_names = [name for name in all_names if (('AUC' not in name) and ('Standard Deviation' in name))]
        size_names = [name for name in all_names if (('Size' in name))]
        max_auc_names = [name for name in all_names if (('AUC' in name) and ('Max' in name))]
        med_auc_names = [name for name in all_names if (('AUC' in name) and ('Median' in name))]
        std_auc_names = [name for name in all_names if (('AUC' in name) and ('Standard Deviation' in name))]
        name_triples = []
        std_triples = []

        for i in range(len(min_names)):
            for j in range(len(max_auc_names)):
                for k in range(len(size_names)):
                    name_triples.append((min_names[i], max_auc_names[j], size_names[k]))
                    std_triples.append((None, None, None))
                    name_triples.append((med_names[i], med_auc_names[j], size_names[k]))
                    std_triples.append((std_names[i], std_auc_names[j], None))

        if args.type == 'cnn':
            to_enumerate = ['Cicada V1 (search)', 'Cicada V2 (search)']

        for ((name_a, name_b, name_c), (std_name_a, std_name_b, std_name_c)) in zip(name_triples, std_triples):
            draw_study.plot_3d_pareto(name_a, name_b, name_c, std_name_a, std_name_b, std_name_c, args.name, label_seeds=False, name=f'{args.name}-all-and-pareto-{name_a.replace(".npy", "")}-{name_b.replace(".npy", "")}-{name_c.replace(".npy", "")}')

    def trial_plots():
        names = [name for name in os.listdir(f'arch/{args.name}/trial_metrics') if (os.path.isdir(os.path.join(f'arch/{args.name}/trial_metrics', name)) and ('AUC' not in name) and ('Model Size' not in name))]
        auc_names = [name for name in os.listdir(f'arch/{args.name}/trial_metrics') if (os.path.isdir(os.path.join(f'arch/{args.name}/trial_metrics', name)) and ('AUC' in name) and ('Model Size' not in name))]
        name_pairs = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name_pairs.append((names[i], names[j]))
            for j in range(len(auc_names)):
                name_pairs.append((names[i], auc_names[j]))
        trial_names = os.listdir(f'arch/{args.name}/trial_metrics/Validation Loss')

        for (name_a, name_b) in name_pairs:
            for trial_name in trial_names:
                draw_trial.plot_2d_pareto(name_a, name_b, trial_names=[trial_name], argname=args.name, label_seeds=True, show_non_pareto=True, name=f'{args.name}-{trial_name}-all-and-pareto-{name_a}-{name_b}')
            draw_trial.plot_2d_pareto(name_a, name_b, trial_names=trial_names, argname=args.name, label_seeds=False, show_non_pareto=False, name=f'{args.name}-all-trials-all-and-pareto-{name_a}-{name_b}')
                
    def evaluate_teacher(teacher):
        aucs, sizes, val_losses = [], [], []
        log = pd.read_csv(f"arch/{args.name}/models/{teacher.name}/training.log")
        draw_trial.plot_loss_history(
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

        draw_trial.plot_anomaly_score_distribution(
            list(teacher_results.values()),
            [*teacher_results],
            f"anomaly-score-{teacher.name}",
        )
        draw_trial.plot_roc_curve(y_true, y_pred_teacher, [*X_signal], inputs, f"roc-{teacher.name}")
        roc_aucs, _ = draw_study.get_aucs(y_true, y_pred_teacher, use_cut_rate=True)
        aucs.append(np.power([10.], np.mean(roc_aucs)))
        sizes.append(1000)
        val_losses.append(log["val_loss"].to_numpy()[-1])

        return aucs, sizes, val_losses

    def evaluate_students(student_models):
        aucs, sizes, val_losses = [], [], []
        for student in student_models:
            log = pd.read_csv(f"arch/{args.name}/models/{student.name}/training.log")
            draw_trial.plot_loss_history(
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
        
            draw_trial.plot_anomaly_score_distribution(
                list(student_results.values()),
                [*student_results],
                f"anomaly-score-{student.name}",
            )

            draw_trial.plot_roc_curve(y_true, y_pred_student, [*X_signal], inputs, f"roc-{student.name}")
            roc_aucs, _ = draw_study.get_aucs(y_true, y_pred_student, use_cut_rate=True)
            aucs.append(np.power([10.], np.mean(roc_aucs)))
            sizes.append(student.count_params())
            val_losses.append(log["val_loss"].to_numpy()[-1])

        return aucs, sizes, val_losses

    config = yaml.safe_load(open(args.config))

    draw_trial = Draw(output_dir=f'arch/{args.name}/trial_plots/', interactive=args.interactive)
    draw_study = Draw(output_dir=f'arch/{args.name}/study_plots/', interactive=args.interactive)

    # Evaluate search
    trial_plots()
    search_plots()

    if args.search_only == True:
        return

    # Load old models
    for fromname, toname in [
        ['teacher', f'arch/{args.name}/models/teacher'], 
        ['cicada-v1', f'arch/{args.name}/models/cicada-v1'], 
        ['cicada-v2', f'arch/{args.name}/models/cicada-v2'], 
    ]:
        if not os.path.isdir(toname):
            shutil.copytree(f'{args.input}/{fromname}', toname)

    gen, X_train, y_train, X_val, y_val, X_test, X_signal = get_data_npy(config)

    # Load study and best trials (use if had search)
    loaded_study = optuna.load_study(study_name=args.name, storage=f"sqlite:///arch/{args.name}/{args.name}.db")
    pareto_trials = loaded_study.best_trials
    pareto_params = [trial.params for trial in pareto_trials]

    # (use if want to train specific archs)
    #pareto_params = [
    # for cnn
    #    {'n_conv_layers': 1, 'n_filters_0': 5, 'n_dense_layers': 1, 'n_dense_units_0': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 6, 'n_dense_layers': 1, 'n_dense_units_0': 13}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 7, 'n_dense_layers': 1, 'n_dense_units_0': 11}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 6, 'n_dense_layers': 1, 'n_dense_units_0': 13}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 2, 'n_dense_units_0': 15, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 5, 'n_dense_layers': 2, 'n_dense_units_0': 12, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 6, 'n_dense_layers': 2, 'n_dense_units_0': 11, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 7, 'n_dense_layers': 2, 'n_dense_units_0': 10, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 8, 'n_dense_layers': 2, 'n_dense_units_0': 9, 'n_dense_units_1': 16}, # Final model 1
    #    {'n_conv_layers': 1, 'n_filters_0': 9, 'n_dense_layers': 2, 'n_dense_units_0': 8, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 10, 'n_dense_layers': 2, 'n_dense_units_0': 7, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 2, 'n_dense_units_0': 16, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 3, 'n_dense_units_0': 16, 'n_dense_units_1': 16, 'n_dense_units_2': 16}, # Final model 2
    #    {'n_conv_layers': 0, 'n_dense_layers': 4, 'n_dense_units_0': 16, 'n_dense_units_1': 16, 'n_dense_units_2': 16, 'n_dense_units_3': 16}, 
    #    in cnn_skip, cnn_skip_1, cnn_skip_2, cnn_skip_3
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'shortcut': True}, 
    #    in cnn_drop
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 20}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 10}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 6}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 5}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 4}, 
    #    {'n_conv_layers': 1, 'n_filters_0': 4, 'n_dense_layers': 1, 'n_dense_units_0': 16, 'dropout': 3}, 
    #    in cnn_quant
    #    {"q_kernel_conv": quantized_bits(12, 3, 1, alpha=1.0), "q_kernel_dense": quantized_bits(8, 1, 1, alpha=1.0), "q_bias_dense": quantized_bits(8, 3, 1, alpha=1.0), "q_activation": "quantized_relu(10, 6)"}, 
    #    {"q_kernel_conv": quantized_bits(8, 4, 1, alpha=1.0), "q_kernel_dense": quantized_bits(8, 4, 1, alpha=1.0), "q_bias_dense": quantized_bits(8, 4, 1, alpha=1.0), "q_activation": "quantized_relu(8, 4)"}, 
    #    {"q_kernel_conv": quantized_bits(16, 8, 1, alpha=1.0), "q_kernel_dense": quantized_bits(16, 8, 1, alpha=1.0), "q_bias_dense": quantized_bits(16, 8, 1, alpha=1.0), "q_activation": "quantized_relu(16, 8)"}, 
    #    {"q_kernel_conv": quantized_bits(32, 16, 1, alpha=1.0), "q_kernel_dense": quantized_bits(32, 16, 1, alpha=1.0), "q_bias_dense": quantized_bits(32, 16, 1, alpha=1.0), "q_activation": "quantized_relu(32, 16)"}, 
    # for bnn
    #    {'n_conv_layers': 0, 'n_dense_layers': 2, 'n_dense_units_0': 16, 'n_dense_units_1': 16}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 3, 'n_dense_units_0': 16, 'n_dense_units_1': 16, 'n_dense_units_2': 16}, 
    # for bitbnn
    #    in bitbnn_1
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 2, 'n_dense_units_0': 10, 'n_dense_units_1': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 3, 'n_dense_units_0': 10, 'n_dense_units_1': 10, 'n_dense_units_2': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 11}, 
    #    in bitbnn_2, bit_bnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 5}, # not in bitbnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 6}, # not in bitbnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 7}, # not in bitbnn_3
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 8}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 9}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 10}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 11}, 
    #    {'n_conv_layers': 0, 'n_dense_layers': 1, 'n_dense_units_0': 12}, # not in bitbnn_3
    #    in bitbnn_conv
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [4], 'shortcut': True}, 
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [6], 'shortcut': True}, 
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [8], 'shortcut': True}, 
    #    {'n_conv_layers': 1, 'n_filters': [4], 'n_dense_layers': 1, 'n_dense_units': [10], 'shortcut': True}, 
    #]

    trial_models = []
    for params in pareto_params:
        for i in range(args.executions):
            model_name = '_'.join(str(x) + '_' + str(y) for x, y in params.items())
            model_name = model_name + f"_x_{i}"
            model_name = model_name.replace('[', '').replace(']', '').replace(',', '_').replace(' ', '').replace('(', '_').replace(')', '_').replace('_alpha=1.0', '')
            trial_models.append(load_model(f'arch/{args.name}/models/{model_name}'))

    # Load models
    teacher = load_model(f"arch/{args.name}/models/teacher")
    cicada_v1 = load_model(f"arch/{args.name}/models/cicada-v1")
    cicada_v2 = load_model(f"arch/{args.name}/models/cicada-v2")
    student_models = [cicada_v1, cicada_v2] + trial_models

    # Evaluate teacher
    aucs_teacher, sizes_teacher, val_losses_teacher = evaluate_teacher(teacher)
    
    # Evaluate students
    aucs_students, sizes_students, val_losses_students = evaluate_students(student_models)

    aucs = aucs_teacher + aucs_students
    sizes = sizes_teacher + sizes_students
    val_losses = val_losses_teacher + val_losses_students
    
    if args.type == 'cnn':
        to_enumerate = ['Teacher', 'Cicada V1', 'Cicada V2']
    else: to_enumerate = []

    draw_study.plot_2d(sizes, aucs, xlabel='Model Size', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-size-auc-all')
    draw_study.plot_2d(sizes, val_losses, xlabel='Model Size', ylabel='Validation Loss', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-size-loss-all')
    draw_study.plot_2d(val_losses, aucs, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-loss-auc-all')
    draw_study.plot_3d(val_losses, aucs, sizes, xlabel='Validation Loss', ylabel='Mean AUC (<3 kHz)', zlabel='Model Size', to_enumerate=to_enumerate, label_seeds=False, name=f'{args.name}-scatter-loss-auc-size-all')

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
        "--search_only", "-s",
        type=bool,
        default=True,
        help="Only evaluate the search? Either True or False",
    )
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
    new_args = parser.parse_args()
    loaded_args = load_args(new_args.name)
    main(parser.parse_args(['--name'] + [f"{new_args.name}"] + ['--search_only'] + [f"{new_args.search_only}"] + loaded_args))
