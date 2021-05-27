### IMPORT LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


### SHARPE RATIO
def sharpe_ratio(rets): # mean returns over standard deviation of the returns
    return rets.mean() / rets.std()

### TRADER FUNCTION
# theta is our input parameters in this case the slope of our line
def positions(x, theta):
    M = len(theta) - 2
    T = len(x)
    Ft = np.zeros(T)
    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t - 1]]])
        Ft[t] = np.tanh(np.dot(theta, xt))
    return Ft

### RETURNS
def returns(Ft, x, delta):
    T = len(x)
    rets = Ft[0:T - 1] * x[1:T] - delta * np.abs(Ft[1:T] - Ft[0:T - 1])
    return np.concatenate([[0], rets])


### GRADIENT ASCENT
def gradient(x, theta, delta):
    Ft = positions(x, theta)
    rets = returns(Ft, x, delta)
    T = len(x)
    M = len(theta) - 2
    
    A = np.mean(rets)
    B = np.mean(np.square(rets))
    S = A / np.sqrt(B - A ** 2)

    grad = np.zeros(M + 2)  # initialize gradient
    dFpdtheta = np.zeros(M + 2)  # for storing previous dFdtheta
    
    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t-1]]])
        dRdF = -delta * np.sign(Ft[t] - Ft[t-1])
        dRdFp = x[t] + delta * np.sign(Ft[t] - Ft[t-1])
        dFdtheta = (1 - Ft[t] ** 2) * (xt + theta[-1] * dFpdtheta)
        dSdtheta = (dRdF * dFdtheta + dRdFp * dFpdtheta)
        grad = grad + dSdtheta
        dFpdtheta = dFdtheta

        
    return grad, S

### TRAIN MODEL
# ideal epoch size is around 210+ as it plateau off 
# using epoch = 300
def train(x, epochs=300, M=5, commission=0.0025, learning_rate = 0.1):
    theta = np.ones(M + 2)
    sharpes = np.zeros(epochs) # store sharpes over time
    for i in range(epochs):
        grad, sharpe = gradient(x, theta, commission)
        theta = theta + grad * learning_rate

        sharpes[i] = sharpe
    
    
    print("finished training")
    return theta, sharpes

### LOAD DATA FROM RESPECTIVE TICKER FOLDER
dir = "C:/Users/night/RRL-Stock-Trading/Stock Tickers/BX/Final/BX"
path = f'{dir}_Train.csv'

name = 'BX' # Keep this as BX

df = pd.read_csv(f"{dir}_Train.csv")

    
### USING PURELY TRAIN DATASET 

ticker = df

rets = ticker['high'].diff()[1:]

x = np.array(rets)
x = (x - np.mean(x)) / np.std(x) # normalize data

### VERSION 1
# N = int(len(ticker)*0.905) # 0.905 is from 2020-8-17 onwards) #Ctrl + 1 to comment/uncomment
# P = int(len(ticker)*(1-0.905)) 
# x_train = x[-(N+P):-P]
# x_test = x[-P:]

### VERSION 2
N = int(len(ticker)*1)
P = int(len(ticker)*(0)) 
x_train = x[-(N+P):-P]

x_test = pd.read_csv(f"{dir}_Test.csv")


"""Now we're ready to train! We'll give the model a look-back window of 12 ticks since each tick is an hour"""

theta, sharpes = train(x_train, epochs=1000, M=24, commission=0.0025, learning_rate=.01) # M is the look-back window



"""In order to see how well the training did, we can graph the resulting Sharpe ratio over each epoch, and hopefully see it converge to a maximum."""

plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio');

"""From the graph above, we can identify that the optimal epoch number for the model is around 210 as it converges towards a maximum Sharpe Ratio."""

train_returns = returns(positions(x_train, theta), x_train, 0.0025)
plt.plot((train_returns).cumsum(), label="Reinforcement Learning Model")
plt.plot(x_train.cumsum(), label="Buy and Hold")
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Training Data");

"""We can see that, over the training data, our reinforcement learning model greatly outperformed simply buying and holding the asset. Lets see how it does over the next 200 ticks, which have been held out from the model."""

test_returns = returns(positions(x_test, theta), x_test, 0.0025)

plt.figure(figsize=(20,7))
plt.plot((test_returns).cumsum(), label="Reinforcement Learning Model")
plt.plot(x_test.cumsum(), label="Buy and Hold")
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend(fontsize=20)
plt.title("RL Model vs. Buy and Hold - Test Data");
plt.show()


### RANDOM PRINTS

#  Print Time Period of Test Data (1000)

ticker.tail(1000)





