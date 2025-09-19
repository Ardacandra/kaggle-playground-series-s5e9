# Kaggle Competition - Predicting the Beats-per-Minute of Songs

### Overview

This repository contain my work for the [**Predicting the Beats-per-Minute of Songs**](https://www.kaggle.com/competitions/playground-series-s5e9/overview) hosted on Kaggle.

The goal of this competition is to predict a song's beats-per-minute.

### Project Structure

```
├── configs/            # Configurations for training and prediction
├── data/               # Raw data
├── models/             # Store trained models
├── notebooks/          # Jupyter notebooks for exploration & experiments
├── output/             # Store model predictions
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

1. Train the model

```
python train.py --config configs/default.yaml
```

2. Test the model

```
python predict.py --config configs/default.yaml
```

### Results

### Acknowledgements

This project was developed as part of the Kaggle Playground Series – Season 5, Episode 9. [Adil Shamim's notebook](https://www.kaggle.com/code/adilshamim8/predicting-the-beats-per-minute-of-songs-101) is also used for reference and performance comparison.

- Walter Reade and Elizabeth Park. Predicting the Beats-per-Minute of Songs. https://kaggle.com/competitions/playground-series-s5e9, 2025. Kaggle.
- Adil Shamim. Predicting the Beats-per-Minute of Songs 101. https://www.kaggle.com/code/adilshamim8/predicting-the-beats-per-minute-of-songs-101, 2025.
