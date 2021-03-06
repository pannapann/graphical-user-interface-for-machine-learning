# graphical user interface for machine learning

<p align="center">
  <img src="assets/Jun-14-2565 21-09-35.gif" alt="Example usage" width="1920" />
</p>

<p align="center">
  <img src="assets/Jun-14-2565 21-07-19.gif" alt="Example usage" width="1920" />
</p>

## Installation

0. Create conda environment

```bash
conda create -n myenv python=3.8
```

1. Run these commands in terminal

```bash
pip install pycaret[full]
```

```bash
pip install streamlit
```

```bash
python -m spacy download en_core_web_sm
```

```bash
python -m textblob.download_corpora
```

or

2. Run these commands in terminal

```bash
pip install -r requirements.txt
```

```bash
python -m spacy download en_core_web_sm
```

```bash
python -m textblob.download_corpora
```

## Run command

```bash
streamlit run main.py
```
