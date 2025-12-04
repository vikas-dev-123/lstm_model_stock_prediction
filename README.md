# Stock Price Prediction (LSTM)

Comprehensive project documentation for the notebook `stockPricePrediction.ipynb` — a working experiment that downloads historical stock prices, prepares sequential data, trains an LSTM to predict the closing price, and visualizes results.

**Table of contents**
- Project overview
- Quick start
- Notebook structure
- Data source and preprocessing
- Model architecture
- Training & evaluation
- Files in repo
- Research paper
- Reproducing results
- Suggestions and next steps
- Author & license

**Project overview**

This project demonstrates a sequence modeling approach (LSTM) to predict daily stock close prices. The notebook:
- downloads historical price data via `yfinance`;
- selects and scales the `Close` price;
- creates fixed-length look-back sequences for supervised learning;
- defines and trains a small LSTM network with dropout;
- evaluates performance using RMSE and visualizes predictions vs. actual prices.

This is an experimental research/demo project and is not financial advice.

**Quick start (Windows PowerShell)**

1. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

2. Start Jupyter and open the notebook:

```powershell
jupyter notebook stockPricePrediction.ipynb
# or
jupyter lab
```

3. Run notebook cells top-to-bottom. Change parameters such as `ticker_symbol`, `start_date`, `end_date`, `time_step`, and training hyperparameters as needed.

**Notebook structure (high-level)**

- Data download: uses `yfinance` to fetch OHLCV historical data for `ticker_symbol`.
- Data inspection & cleaning: prints shapes, checks nulls, and normalizes column names.
- Preprocessing: selects the `close` column, reshapes and scales values with `MinMaxScaler`.
- Sequence creation: builds input sequences and targets with a configurable `time_step` (look-back window).
- Train/test split: typically 80/20 split, reshaped for LSTM input `[samples, time_steps, features]`.
- Model definition: `tensorflow.keras.Sequential` with two LSTM layers and `Dropout`, compiled with `adam` and MSE loss.
- Training: `model.fit(...)` for configurable `epochs` and `batch_size`.
- Evaluation: predictions are inverse-transformed, RMSE is computed for train and test sets.
- Visualization: compares original series with train and test predictions using `matplotlib`.

**Data source & preprocessing**

- Source: `yfinance` (downloads data for the given ticker between `start_date` and `end_date`).
- Feature: the notebook focuses on the `Close` price as the target variable.
- Scaling: `MinMaxScaler(feature_range=(0,1))` to normalize values before training.
- Sequences: each sample contains `time_step` consecutive close prices as features, next-day close as the label.

**Model architecture**

- Input: sequences of shape `(time_step, 1)` where `time_step` (e.g., 60) is a look-back window.
- Layers:
	- LSTM(units=50, return_sequences=True) + Dropout(0.2)
	- LSTM(units=50, return_sequences=False) + Dropout(0.2)
	- Dense(units=1)
- Loss: mean squared error (MSE)
- Optimizer: Adam

These choices are a simple baseline; tuning units, layers, dropout, optimizer, or using attention/transformer blocks may improve results.

**Training & evaluation**

- Typical training config used in the notebook: `epochs=100`, `batch_size=64` (adjustable).
- After training, predictions are inverse-transformed back to the original scale and RMSE is computed for train and test sets.
- Visual inspection of prediction plots is recommended; consider additional metrics (MAE, MAPE) for more insight.

**Files in this repository**

- `stockPricePrediction.ipynb` — primary notebook with data download, model training, evaluation and visualizations.
- `requirements.txt` — minimal dependencies to run the notebook.
- `trans_rl_model.pdf` — research paper included in the folder (see below).
- `README.md` — this documentation.

**Research paper**

The PDF `trans_rl_model.pdf` is present in the project root. It contains your related research and background. To keep the repository tidy, move the file to a `papers/` folder:

```powershell
New-Item -ItemType Directory -Path .\papers -ErrorAction SilentlyContinue
Move-Item -Path .\trans_rl_model.pdf -Destination .\papers\
```

 

**Reproducibility / how to reproduce results**

1. Install packages from `requirements.txt` (see Quick start).
2. Optionally pin exact versions using `pip freeze > requirements.txt` after installing the environment you used for experiments.
3. Edit these notebook variables before running:
	- `ticker_symbol` (e.g., `'AAPL'`)
	- `start_date`, `end_date`
	- `time_step`, `epochs`, `batch_size`
4. Run the notebook cells in order; training progress and final RMSE are printed.

**Tips for faster iteration**
- Reduce `epochs` (e.g., 5–10) and `time_step` during development.
- Use a smaller dataset window (shorter date range) for quick checks.
- Consider saving model weights with `model.save('model.h5')` and loading them with `keras.models.load_model(...)`.

**Ideas for improvements**
- Add technical indicators (SMA, EMA, RSI) as additional features.
- Use multivariate input (Open, High, Low, Volume) instead of single feature.
- Try sequence models with attention or transformer encoder blocks.
- Use walk-forward validation instead of a single train/test split for more robust evaluation.
- Add hyperparameter tuning (grid/random search, Optuna).

**Authorship & Contact**

All work in this repository — including the notebook experiments, data processing, model design, training runs, analysis, and the included paper — was performed and authored solely by `vkas dev pandey`. This project documentation intentionally reflects the author's own work and was written to represent the author's methods and results.

 

**License**

  MIT, Apache-2.0. Tell me which license you prefer and I'll add a `LICENSE` file.

---

 
