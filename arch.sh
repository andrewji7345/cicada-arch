#!/bin/bash
source /eos/user/a/aji/.conda/envs/cicada/bin/activate
python3 arch.py -n test_sub -y cnn -e 1 -x 1 -t 1 -c config_condor.yml
python3 arch_evaluating.py -n test_sub -c config_condor.yml
