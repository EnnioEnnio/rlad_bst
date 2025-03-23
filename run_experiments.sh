#!/bin/bash
python3 rlad_bst/train.py  --config optimal_config.yaml --run_name base
python3 rlad_bst/train.py  --config optimal_config.yaml --do_action_masking False --run_name no_action_masking
python3 rlad_bst/train.py  --config optimal_config.yaml --reward_function old --run_name old_reward
python3 rlad_bst/train.py  --config optimal_config.yaml --run_name incremental_reward --incremental_reward

python3 rlad_bst/train.py  --config optimal_config.yaml --run_name naive --naive True
python3 rlad_bst/train.py  --config optimal_config.yaml --run_name naive_with_incr_reward --naive True --incremental_reward

python3 rlad_bst/train.py  --config optimal_config.yaml --entropy_coefficient 0.0 --run_name no_entropy
python3 rlad_bst/train.py  --config optimal_config.yaml --entropy_coefficient 0.5 --run_name high_entropy
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 1.0 --run_name no_temperature
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 10.0 --run_name high_temperature
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 1.0 --entropy_coefficient 0.0 --run_name no_temp_entropy

python rlad_bst/train.py  --config optimal_config.yaml --use_custom_value_net False --run_name no_value_net
python rlad_bst/train.py  --config optimal_config.yaml --use_custom_action_net False --run_name no_action_net
python rlad_bst/train.py  --config optimal_config.yaml --pretrained_encoder jina-not-pretrained --run_name not_pretrained
python rlad_bst/train.py  --config optimal_config.yaml --pretrained_encoder default --run_name default_encoder
python rlad_bst/train.py  --config optimal_config.yaml --pretrained_encoder default --use_custom_action_net False --use_custom_value_net False --run_name nothing_custom

python3 rlad_bst/train.py  --config optimal_config.yaml --start_data_len 2 --grow_data True --run_name grow_data
python3 rlad_bst/train.py  --config optimal_config.yaml --start_data_len 2 --start_program_len_factor 2 --grow_data True --grow_program_len True --run_name grow_data_program
python3 rlad_bst/train.py  --config optimal_config.yaml --start_program_len_factor 2 --grow_program_len True --run_name grow_program_len
