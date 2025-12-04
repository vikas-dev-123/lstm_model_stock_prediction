# Stock Price Prediction (LSTM)

Project notebook that downloads historical stock price data and trains an LSTM model to predict the close price.

**Main file:** `stockPriceDeection.ipynb`

**Description:**
- Downloads historical stock data using `yfinance`.
- Preprocesses the `Close` prices (scaling and sequence creation).
- Builds and trains an LSTM model using `tensorflow.keras`.
- Evaluates model performance (RMSE) and visualizes predictions vs. actual prices.

**Dependencies**
The notebook uses these Python packages (minimum):

- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tensorflow` (or `tensorflow-cpu`)

You can install them with pip in a new virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install yfinance pandas numpy scikit-learn matplotlib tensorflow
```

If you have a GPU-enabled machine, install a GPU-enabled `tensorflow` package appropriate for your system.

**How to run**

1. Create and activate a virtual environment (see commands above).
2. Start Jupyter Notebook or Jupyter Lab:

```powershell
jupyter notebook stockPriceDeection.ipynb
# or
jupyter lab
```

3. Open `stockPriceDeection.ipynb` and run cells in order. Change the `ticker_symbol`, `start_date`, or model hyperparameters as needed.

**Notes & Tips**
- Training the LSTM can take time — reduce `epochs` or `time_step` for faster iterations.
- Save trained model weights if reusing the model frequently.
- Monitor memory/GPU usage when using large datasets or `tensorflow` with GPU.

**Suggested next steps**
- Create a `requirements.txt` using `pip freeze > requirements.txt` after installing the exact versions you want to pin.
- Add a short script (`run_training.py`) to run training outside of the notebook for reproducibility.
- Add a results/figures folder and save plots programmatically.

**Author**
 
Created by `vkas dev pandey` — notebook located at `stockPriceDeection.ipynb` in this folder.

**Research Paper**

- The research paper `trans_rl_model.pdf` is included in this project folder. It contains background and research details related to reinforcement learning / transformer models (or other topics you authored).
- To keep the project organized you can move the paper into a `papers/` folder. Example PowerShell commands:

```powershell
# create a papers folder and move the PDF into it
New-Item -ItemType Directory -Path .\papers -ErrorAction SilentlyContinue
Move-Item -Path .\trans_rl_model.pdf -Destination .\papers\
```

- If you would like, I can move the file into `papers/` and update the README accordingly — tell me to proceed and I'll do it.
