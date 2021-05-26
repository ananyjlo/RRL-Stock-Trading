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
- Outputs 15min and 30min data for model training

5) Ticker Data

- Obtain from the "Final Data" folder and select the respective timeframe you want

## Notes

LMND don't have enough data points as it only got listed in 2020

