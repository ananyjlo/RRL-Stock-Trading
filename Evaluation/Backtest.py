### LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

### TEST CONDITIONS
"""
02 Jan 2020 to 31 Dec 2020
GOOG, AAPL, BABA, BX, C
10,000 USD
No fractional shares
"""

### LOAD DATA
dir = r"C:\Users\night\RRL-Stock-Trading\Stock Tickers\Final Data\Daily" # local path
name = 'Deep Evolution Strategy' 
name2 = 'AAPL' # ticker name
path = f'{dir}\{name2}_Daily*.csv' # 15min, 30min, 1hr, daily

import glob
files = glob.glob(path)
for f in files:
  print(f)
  
import pandas as pd
df = pd.DataFrame()
for f in files:
    df_full = pd.read_csv(f)

print(f"No. of data points in {name2} test dataset: {len(df_full)}")

### SPLICE DATA
"""
Select Start and End Period
"""

########################### BACKTEST PERIOD ###########################

df_full.set_index('time', inplace = True)
start = '2020-01' ## YYYY-MM
end = '2021-01' ## YYYY-MM

df_full = df_full.loc[start:end].copy()
print(f"No. of data points in {name2} test dataset: {len(df_full)}")

df_full.drop(['symbol'], axis=1, inplace=True)
df_full.reset_index(inplace=True)

df_full.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

start_date = df_full.at[0, 'Date'] 
end_date = df_full.at[len(df_full)-1, 'Date']
print("\n")
print (f"Start Data:{start_date}, End Date:{end_date}")


### EVOLUTION FUNCTION
solution = np.random.randn(100)
w = np.random.randn(100) # initial guess is random
def f(w):
    reward = -np.sum(np.square(solution - w))
    return reward

# hyperparameters 
npop = 50      # population size
sigma = 0.1    # noise standard deviation
alpha = 0.01  # learning rate

for i in range(5000): # try 5900 or 10000
    
    # print current fitness of the most likely parameter setting
    if (i + 1) % 200 == 0:
        print(
            'iter %d. w: %s, solution: %s, reward: %f'
            % (i + 1, str(w[-1]), str(solution[-1]), f(w))
        )
        
    # initialize memory for a population of w's, and their rewards
    N = np.random.randn(npop, 100) # samples from a normal distribution N(0,1)
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma * N[j] # jitter w using gaussian of sigma 0.1
        R[j] = f(w_try) # evaluate the jittered version
    
    # standardize the rewards to have a gaussian distributio
    A = (R - np.mean(R)) / np.std(R)
    
    # perform the parameter update. The matrix multiply below
    # is just an efficient way to sum up all the rows of the noise matrix N,
    # where each row N[j] is weighted by A[j]
    
    w = w + alpha / (npop * sigma) * np.dot(N.T, A)

# f(w)

### VISUALIZE EVOLUTIONARY RESULTS
'''
Compare real w computed from previous f(w) with first two individuals
'''
plt.figure(figsize=(10,5))

sigma = 0.1
N = np.random.randn(npop, 100)
individuals = []
bins = np.linspace(-10, 10, 100)

for j in range(2):
    individuals.append(w + sigma * N[j])
    
    
plt.hist(w, bins, alpha=0.5, label='w',color='r')
plt.hist(individuals[0], bins, alpha=0.5, label='individual 1')
plt.hist(individuals[1], bins, alpha=0.5, label='individual 2')
plt.legend()
plt.show()

### FEATURE ENGINEER
df= df_full.copy()
close = df.Close.values.tolist()

def get_state(data, t, n): # (close price, day, window size)
    d = t - n + 1
    block = data[d : t + 1] if d >= 0 else -d * [data[0]] + data[: t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    # print (block)
    return np.array([res])  # featuring technique for closing price 

# try out different window size: 10,20,30
# print("window size:10",get_state(close, 50, 10))
# print("window size:20",get_state(close, 50, 20))
# print("window size:30",get_state(close, 50, 30))

### DEEP EVOLUTION CLASS
class Deep_Evolution_Strategy:
    def __init__(
        self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    # def train(self, epoch = 100, print_every = 1):
    #     lasttime = time.time()
    #     for i in range(epoch):
    #         population = []
    #         rewards = np.zeros(self.population_size)
    #         for k in range(self.population_size):
    #             x = []
    #             for w in self.weights:
    #                 x.append(np.random.randn(*w.shape))
    #             population.append(x)
    #         for k in range(self.population_size):
    #             weights_population = self._get_weight_from_population(
    #                 self.weights, population[k]
    #             )
    #             rewards[k] = self.reward_function(weights_population)
    #         rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    #         for index, w in enumerate(self.weights):
    #             A = np.array([p[index] for p in population])
    #             self.weights[index] = (
    #                 w
    #                 + self.learning_rate
    #                 / (self.population_size * self.sigma)
    #                 * np.dot(A.T, rewards).T
    #             )
    #         if (i + 1) % print_every == 0:
    #             print(
    #                 'iter %d. reward: %f'
    #                 % (i + 1, self.reward_function(self.weights))
    #             )
    #     print('time taken to train:', (time.time() - lasttime)/60, 'minutes')
        
### MODEL CLASS
class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
#         print (decision)
        return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
        
### AGENT CLASS
import time


class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(
        self, model, money, max_buy, max_sell, close, window_size, skip
    ):
        self.window_size = window_size
        self.skip = skip
        self.close = close
        self.model = model
        self.initial_money = money
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.invest = 0
        self.save_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '-' + model_name + '-' + ticker
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
    
    def get_invest(self):
        return self.invest
    
    def get_name(self):
        return self.save_name
    
    def act(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
#         print (decision[0])
#         decision = decision[~np.isnan(decision)]
#         buy = buy[~np.isnan(buy)]
        return np.argmax(decision[0]), int(buy[0]) # returns either a buy, no action or sell, while the 2nd output tells u how many to buy

    def get_reward(self, weights):
        initial_money = self.initial_money
        starting_money = initial_money
        len_close = len(self.close) - 1

        self.model.weights = weights
        state = get_state(self.close, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, len_close, self.skip):
            action, buy = self.act(state)
            next_state = get_state(self.close, t + 1, self.window_size + 1)
            if action == 1 and initial_money >= (self.max_buy-10) * self.close[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * self.close[t] # Note
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                elif t == len_close:
                    sell_units = quantity
                else:
                    sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * self.close[t]
                initial_money += total_sell

            state = next_state
        return ((initial_money - starting_money) / starting_money) * 100

    # def fit(self, iterations, checkpoint):
    #     self.es.train(iterations, print_every = checkpoint)
        
    def save_weights(self, file_path):
        weights = self.model.get_weights()
        np.save(file_path + '-weight0', weights[0])
        np.save(file_path + '-weight1', weights[1])
        np.save(file_path + '-weight2', weights[2])
        np.save(file_path + '-weight3', weights[3])
        
    def load_weights(self, file_path):
        weights = []
        weights.append(np.load(file_path + '-weight0.npy'))
        weights.append(np.load(file_path + '-weight1.npy'))
        weights.append(np.load(file_path + '-weight2.npy'))
        weights.append(np.load(file_path + '-weight3.npy'))
        self.model.set_weights(weights)
        
    def buy(self):
        initial_money = self.initial_money
        len_close = len(self.close) - 1
        state = get_state(self.close, 0, self.window_size + 1)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0
        for t in range(0, len_close, self.skip):
            action, buy = self.act(state)
            next_state = get_state(self.close, t + 1, self.window_size + 1)
            if action == 1 and initial_money >= (self.max_buy-5) * self.close[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy            
                total_buy = buy_units * self.close[t] # Note
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                states_buy.append(t)
                print(
                    'Day %d: Buy %d Units at Price %f, Total Balance %f, Shares Hold %f'
                    % (t, buy_units, total_buy, initial_money, quantity)
                )
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                elif t == len_close:
                    sell_units = quantity
                else:
                    sell_units = quantity
                if sell_units < 1:
                    continue
                quantity -= sell_units
                total_sell = sell_units * self.close[t]
                initial_money += total_sell
                states_sell.append(t)
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'Day %d, Sell %d Units at Price %f, Investment %f %%, Total Balance %f, Shares Hold %f'
                    % (t, sell_units, total_sell, invest, initial_money, quantity)
                )
            state = next_state

#         invest = ((initial_money - starting_money) / starting_money) * 100
        self.invest = ((initial_money - starting_money) / starting_money) * 100
        print(
            '\nTotal Profits %f, Total Investment %f %%'
            % (initial_money - starting_money, self.invest)
        )
        plt.figure(figsize = (20, 10))
        plt.plot(close, label = 'Close Price', c = 'g')
        plt.plot(
            close, 'X', label = 'Buy Signal', markevery = states_buy, c = 'b'
        )
        plt.plot(
            close, 'o', label = 'Sell Signal', markevery = states_sell, c = 'r'
        )
#         plt.title('total gains %f, total investment %f%%'%(initial_money - starting_money, invest))
        plt.title(
            '''
            *** Model : %s *** Ticker : %s ***
            *** Start Period : %s *** End Period : %s ***
            *** Initial Capital : %i *** Total Profits : %i *** Total Investment : %i %% ***
            *** Max Buy : %i *** Max Sell : %i *** Window Size : %i *** Iteration : %i ***'''
            % (model_name, ticker,
            start_date, end_date,
            starting_money, round(initial_money - starting_money), self.invest,
            max_buy, max_sell, window_size, iteration)
            )
        plt.legend()
        plt.savefig(f"{model_name}_{ticker}_{max_buy}_Test")
        plt.show()
        
### PARAMETERS
model_name = "Deep Evolution Agent Learning"
ticker = name2

money = 10000
max_buy = 100
max_sell = 100   
layer_size = 500
window_size = 10 
iteration = 500
checkpoint = 20

save_weights = False # backtest version
load = True # backtest version
load_file = ""



model = Model(input_size = window_size, layer_size = layer_size, output_size = 3)
agent = Agent(
    model = model,
    money = money,
    max_buy = max_buy,
    max_sell = max_sell,
    close = close,
    window_size = window_size,
    skip = 1,
)


if not load:
    agent.fit(iterations = iteration, checkpoint = checkpoint)
else:
    agent.load_weights('weights/' + load_file)

agent.buy()

if save_weights:
    agent.save_weights('weights/' )
