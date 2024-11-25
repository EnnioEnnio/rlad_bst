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
