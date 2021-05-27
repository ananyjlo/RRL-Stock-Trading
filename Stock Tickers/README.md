## File Usage

1) Quant Connect Data Scraper.py

> For scraping data

You have to run this inside Quant Connect Terminal.

Create a new cell in their .ipynb file and run it. Your browser should appear a popup to download the .csv file

- Change the start and end period 
- Change the Resolution to "Resolution.Minute", "Resolution.Daily", "Resolution.Hour"

Refer to this [Github Notebook]([Lean/DataConsolidationAlgorithm.py at master · QuantConnect/Lean (github.com)](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/DataConsolidationAlgorithm.py)) for Quant Connect API Tutorial.

2) Data Cleaning for Train/Test.py or ipynb

> For data cleaning

You can run this in your local IDE or on Google Colab 

Remember to change your file path accordingly for the stock ticker data you will like to call

3) RRL_Simplified_VM.ipynb

> Solely for running models on Azure VM

- All other code changes that doesn't require a GPU will be done locally on laptop and then push to Github. 
- On Azure VM, remember to pull the latest files from Git before running this Jupyter Notebook to train your model.

4) Data Cleaning Final.ipynb

> Simplified version for data resampling

- Doesn't plot any graphs for data interpretation
- Outputs 15min, 30min and 60min data for model training

5) Ticker Data

- Obtain from the "Final Data" folder and select the respective timeframe you want

## Notes

LMND don't have enough data points as it only got listed in 2020

## Findings

> Deep Evolutionary Agent (DEA)

1) Increasing Max Buy and Sell size has a positive impact on the Net Returns % for the same ticker

2) Performance wise is better than RRL, 3 digit returns vs 2 digit returns. 

> Recurrent Reinforcement Learning (RRL)

1) Highly correlated data (from Heatmap) doesn't exhibit similar return performance when tested on. The initial assumption to use similar parameters for their training backfired and thus to obtain optimal performance, one has to customize each model parameters per stock (either grid/random search)

2) Increasing Epoch does not have a positive impact on performance

3) Increasing Lookback Window increases performance for 15min data however the maximum threshold has not been tested

4) Optimal learning rate is 0.001 and 0.003 based on different stocks. Increasing learning rate to 0.01 has a detrimental impact on performance as the learning is done faster but at the compromise of achieving sub-optimal set of weights. 

## Some Results

**DEA**

| Ticker | Performance (%) | Max Buy/Sell | Learning Rate |      |
| ------ | --------------- | ------------ | ------------- | ---- |
| QQQ    | 315.8           | 10           | 0.03          |      |
| QQQ    | 243.4           | 5            | 0.03          |      |
| BLK    | 404/334         | 5            | 0.03          |      |
| BLK    | 620             | 10           | 0.03          |      |
| BLK    | 2983            | 100          | 0.03          |      |
|        |                 |              |               |      |
|        |                 |              |               |      |
|        |                 |              |               |      |
|        |                 |              |               |      |

**RRL**

