import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def main(args) -> None:

    labels = [
        'Mean Signal', 
        'SUEP', 
        'H to Long Lived', 
        'VBHF to 2C', 
        'TT', 
        'SUSY GGBBH', 
    ]

    for i in range(args.batch):

        for subfoldername in [
            '../arch/', 
            f'../arch/{args.name}{i}/', 
            f'../arch/{args.name}{i}/study_plots/', 
            f'../arch/{args.name}{i}/trial_plots/', 
            f'../arch/{args.name}{i}/execution_plots/', 
            f'../arch/{args.name}{i}/models/', 
            f'../arch/{args.name}{i}/study_metrics/', 
            f'../arch/{args.name}{i}/trial_metrics/', 
            f'../arch/{args.name}{i}/trial_metrics/Mean Signal AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[0]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[1]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[2]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[3]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[4]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/{labels[5]} AUC (0.3-3 kHz)/', 
            f'../arch/{args.name}{i}/trial_metrics/Validation Loss/', 
            f'../arch/{args.name}{i}/trial_metrics/Model Size (number of parameters)/', 
            f'../arch/{args.name}{i}/trial_metrics/Model Size (b)/', 
            ]:
            if not os.path.exists(subfoldername):
                os.mkdir(subfoldername)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Create dirs""")
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="example",
        help="Name of study",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default="0",
        help="Number of batch jobs to parse. If not a batch job, 0.",
    )
    main(parser.parse_args())
