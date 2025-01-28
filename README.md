# rlad_bst

## Poetry 
We use [Poetry](https://python-poetry.org/docs/) to manage our dependencies. To install the dependencies, run the following command:

```bash
poetry install
```

To add a dependency, run the following command:

```bash
poetry add <package-name>
```

### Development

To install the development dependencies, run the following command:

```bash
poetry install --with dev
```

To add a development dependency, run the following command:

```bash
poetry add --group dev <package-name>
```

## Pre-commit hooks

We use pre-commit to ensure the code quality locally. The package pre-commit will be installed with poetry's dev dependencies. To install the pre-commit hooks, run the following command:

```bash
poetry run pre-commit install
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

### VM Setup
connect to our VM using your HPI-Credentials:
```bash
ssh User.Name@vm-midea03.eaalab.hpi.uni-potsdam.de
```

after cloning the repository to your User Folder, you can install the dependencies using the following command:
```bash
poetry install
```

### Running the script
see [Using the script](#using-the-script)
