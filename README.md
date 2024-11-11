# ApisTox - dataset of pesticide toxicity to honey bees

This repository hosts ApisTox dataset, for applications of data analysis and ML in
ecotoxicology and agrochemistry.

Paper preprint is available [on ArXiv](https://arxiv.org/abs/2404.16196).

Dataset and code are released under [CC-BY-NC-4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

## Dataset files

Final dataset file is `outputs/dataset_final.csv`. For dataset splits, see
`outputs/splits` directory.

Raw input data is in `raw_data` directory. Other datasets from this area are
in `other_sources` directory (we **do not** recommend using them).

## Reproducing dataset

Setup virtual environment:
- Poetry (recommended), run `make install` or `poetry install --no-root`
- venv, run `pip install requirements.txt`

Scripts:
- recreate dataset: `python create_dataset.py`
- split dataset:`python create_dataset_splits.py`
- create analyses and plots: `python analyze_dataset.py`
