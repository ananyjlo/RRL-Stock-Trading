# RRL-Stock-Trading
Stock Trading with Recurrent Reinforcement Learning 

Using RRL as one of the team's algorithmic models in a volatile market environment. This will complement our LSTM and Locally Weight Regression approach. 



## Initial Thought Process

1) We did correlation to find out which tickers exhibit the similar distribution

We assume that tickers with strong correlation could potentially use the same model parameters. This help to increase our efficiency when training.

2) Next we resample our minute data into 15min, 30min and hourly.

The purpose was to test whether is there a significant improvement in their performance and whether is that justified by their training time. We are finding the best balance between computing time and performance.

We run each model on different timeframe for 3 times and then the average run time was calculated. This run time is then used to plot against their return level

3) Next we tried out on various level of parameters
Epoch and Lookback Window

We started off with a high level of epoch to observe whether does the epoch plot converge. Next we adjusted the lookback window in accordance with our selected timeframe.

For example, an hourly data could have a lookback window of 4hrs, 8hrs or 12hrs (need some justification)

^ We strive to find academic research backing on this aspect

4) Finally with the final round of parameters, we plot out the returns of the selected tickers.

## File Usage

[27 May Update]: The below files have been shifted to the folder "Old Code"

1) RRL Backup Version.ipynb

> This is the first version of RRL done on Colab before we shifted it to Jupyter on Azure's VM

2) RRL Simplified Draft.py

> This is the RRL Backup Version but in .py file. Comments were removed. 

3) RRL Simplified.ipynb

> This is the current version.
>
> Edit: Navigate to "Stock Tickers" folder and run "RRL_Simplified_VM" instead.

## Code Resources

**On Colab**

[Deep Trading]([Deep Trading.ipynb - Colaboratory (google.com)](https://colab.research.google.com/drive/1--LJeV_bRaYZZoZNJro4LJ79hKaHpXpZ#scrollTo=KIS0xonWMMMw))

[Deep Determinist Policy Gradient - RL]([Deep Determinist Policy Gradient.ipynb - Colaboratory (google.com)](https://colab.research.google.com/drive/1L3-D2ZmGZkPRsB9gb5BviGkSkMTLti7_#scrollTo=evR-UsF19IAu))

[Agent Trading Strategies]([Agent Trading.ipynb - Colaboratory (google.com)](https://colab.research.google.com/drive/1FzLCI0AO3c7A4bp9Fi01UwXeoc7BN8sW#scrollTo=MUZcLPJAnEAT))

[Deep Evolution Strategy (with and without Bayesian Optimisation)]([Free Agent Trading.ipynb - Colaboratory (google.com)](https://colab.research.google.com/drive/1oWGPasjf8lMkCMdf-LuUDVGAlhRfaUoc))

