# Kaggle Competition - Predicting the Beats-per-Minute of Songs

### Overview

This repository contain my work for the [**Predicting the Beats-per-Minute of Songs**](https://www.kaggle.com/competitions/playground-series-s5e9/overview) hosted on Kaggle.

The goal of this competition is to predict a song's beats-per-minute.

### Project Structure

```
├── data/               # Raw data
├── notebooks/          # Jupyter notebooks for exploration & experiments
├── src/                # Python scripts
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

### Setup Instructions

1. Clone the repository

```
git clone https://github.com/Ardacandra/kaggle-playground-series-s5e9.git
cd kaggle-playground-series-s5e9
```

2. Create environment

```
conda create -n kaggle-env python=3.10
conda activate kaggle-env
pip install -r requirements.txt
```

3. Download the dataset 

You need to place your `kaggle.json` in `~/.kaggle/` first.

```
pip install kaggle
kaggle competitions download -c playground-series-s5e9 -p data/
unzip data/*.zip -d data/
```

### How to Run

### Results

### Acknowledgements