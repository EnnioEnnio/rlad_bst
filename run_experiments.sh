#!/bin/bash
python3 rlad_bst/train.py  --config optimal_config.yaml
python3 rlad_bst/train.py  --config optimal_config.yaml --do-action-masking False
python3 rlad_bst/train.py  --config optimal_config.yaml --reward-function old

python3 rlad_bst/train.py  --config optimal_config.yaml --entropy-coefficient 0.0
python3 rlad_bst/train.py  --config optimal_config.yaml --entropy-coefficient 0.5
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 0.0
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 10.0
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 0.0 --entropy-coefficient 0.0

python rlad_bst/train.py  --config optimal_config.yaml --use_custom_value_net False
python rlad_bst/train.py  --config optimal_config.yaml --use_custom_action_net False
python rlad_bst/train.py  --config optimal_config.yaml --pretrained_encoder jina-not-pretrained
python rlad_bst/train.py  --config optimal_config.yaml --pretrained_encoder default
python rlad_bst/train.py  --config optimal_config.yaml --pretrained_encoder default --use_custom_action_net False --use_custom_value_net False

python3 rlad_bst/train.py  --config optimal_config.yaml --start_data_len: 2 --grow_data True
python3 rlad_bst/train.py  --config optimal_config.yaml --start_data_len: 2 --start_program_len_factor: 2 --grow_data True --grow_program_len True
python3 rlad_bst/train.py  --config optimal_config.yaml --start_program_len_factor: 2 --grow_program_len True
