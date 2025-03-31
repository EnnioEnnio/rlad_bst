# rlad_bst

## Poetry 
We use [Poetry](https://python-poetry.org/docs/) to manage our dependencies. To install the dependencies, run the following command:

```bash
poetry install
```

## Using the script
from within the `rlad_bst` directory, run the following command:

1. **Train a New Model**:
```bash
poetry run python3 train.py --config path/to/config.yaml
```
2. **Debug Mode**:
```bash
poetry run python3 train.py --config path/to/config.yaml --debug true
```
3. **Run with Pre-Trained Model**:
```bash
poetry run python3 train.py --config path/to/config.yaml --model-checkpoint path/to/model.zip
```
4. **To run all experiments**:
```bash
bash ./run_experiments.sh 
```

### VM Setup
connect to our VM using your HPI-Credentials:
```bash
ssh User.Name@vm-midea03.eaalab.hpi.uni-potsdam.de
```

after cloning the repository to your User Folder, you can install the dependencies using the following command:
```bash
poetry install
```

### Running the script on the VM
Create tmux session

```bash
tmux new -s mysession
````

run the script
 > see [Using the script](#using-the-script)

detach from the tmux session
```bash
Ctrl-b d
```

reattach to the tmux session
```bash
tmux attach -t mysession
```

If you forget the session name, list all sessions:
```bash
tmux ls
````

Kill the session
```bash
tmux kill-session -t mysession
```

## Starting a WandB-sweep
> Note: On the VM, you might have to run every command with `poetry run` in front of it. You can also start a poetry shell for that... 
Initialize a sweep:
```bash
wandb sweep sweep-config.yaml
````

Run the agent (also pops up in the terminal output):
```bash
wandb agent rlad_bst/rlad_bst/your_sweep_id
```

The agent now launches your program, passing parameters to tune, e.g.:
```bash
poetry run python3 rlad_bst/train.py --config=config.yaml --entropy_coefficient=<some_value>
```
