#!/bin/bash
python3 rlad_bst/train.py  --config optimal_config.yaml --run-name base
python3 rlad_bst/train.py  --config optimal_config.yaml --do-action-masking False --run-name no_action_masking
python3 rlad_bst/train.py  --config optimal_config.yaml --reward-function old --run-name old_reward

# CHeck if not already done
python3 rlad_bst/train.py  --config optimal_config.yaml --entropy-coefficient 0.0 --run-name no_entropy
python3 rlad_bst/train.py  --config optimal_config.yaml --entropy-coefficient 0.5 --run-name high_entropy
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 0.0 --run-name no_temperature 
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 10.0 --run-name high_temperature
python3 rlad_bst/train.py  --config optimal_config.yaml --temperature 0.0 --entropy-coefficient 0.0 --run-name no_temp_entropy

python rlad_bst/train.py  --config optimal_config.yaml --use-custom-value-net False --run-name no_value_net
python rlad_bst/train.py  --config optimal_config.yaml --use-custom-action-net False --run-name no_action_net
python rlad_bst/train.py  --config optimal_config.yaml --pretrained-encoder jina-not-pretrained --run-name not_pretrained
python rlad_bst/train.py  --config optimal_config.yaml --pretrained-encoder default --run-name default_encoder
python rlad_bst/train.py  --config optimal_config.yaml --pretrained-encoder default --use-custom-action-net False --use-custom-value-net False --run-name base

python3 rlad_bst/train.py  --config optimal_config.yaml --start-data-len: 2 --grow-data True --run-name base
python3 rlad_bst/train.py  --config optimal_config.yaml --start-data-len: 2 --start-program-len-factor: 2 --grow-data True --grow-program-len True --run-name base
python3 rlad_bst/train.py  --config optimal_config.yaml --start-program-len-factor: 2 --grow-program-len True --run-name base
