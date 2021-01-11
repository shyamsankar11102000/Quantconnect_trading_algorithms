''' Copyright 2020, Shyam Sankar, All rights reserved'''

import math
import datetime
import random

class Tabular_Q(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018,1, 1)  #Set Start Date
        self.SetEndDate(2018,9,1)    #Set End Date
        self.SetCash(10000)           #Set Strategy Cash
        self.ibm=self.AddEquity("IBM", Resolution.Daily).Symbol
        self.alpha = 0.2  # learning coeff
        self.discount = 0.9  # discount factor
        # currently Tesla
        self.today_opening = 0
        self.yesterday_opening = 0
        self.change = 0
        self.agent = self.Q_Learner(self.discount, self.alpha)
        self.previous_state = (0, 0)
        self.previous_action = "buy" # first/default action is to buy
        self.current_state = None
        self.current_action = None
        self.time = 0
        self.epsilon_parameter = 0.01 # randomizing parameter

    # Testing use
        self.random_counter = 0 # counts number of random actions
        self.state_counter_dict = {} 
        
        self.Schedule.On(self.DateRules.EveryDay(self.ibm), self.TimeRules.AfterMarketOpen(self.ibm, minutes=1), Action(self.daily_data))
        self.Schedule.On(self.DateRules.EveryDay(self.ibm), self.TimeRules.AfterMarketOpen(self.ibm, minutes=1),Action(self.mainFunction))
    
    
    def OnData(self, data):
        pass
    
    def percent_change(self, previous, current):
        difference=current-previous
        change_quotient=difference/previous
        result=round(change_quotient*100)
        if result == -0.0:
            result = 0.0
        return result
        
    def daily_data(self):
        self.dataframe=self.History([self.ibm],2)
        self.dataframe["open"].unstack(level=0)
        self.today_opening = self.dataframe.iat[0,1]
        self.yesterday_opening = self.dataframe.iat[0,0]
        self.change = self.percent_change(self.yesterday_opening, self.today_opening)
    
    def buy(self):
        if not self.Transactions.GetOpenOrders():
            cash=self.Portfolio.Cash
            if cash < 0:
                return 1
            else:
                self.SetHoldings(self.ibm,1.0)
        return 0
        
    def sell(self):
        positions_value=self.Portfolio.TotalPortfolioValue
        self.SetHoldings(self.ibm, 0)
        return 0
        
    def mainFunction(self):
        self.change=self.percent_change(self.yesterday_opening, self.today_opening)
        
        while math.isnan(self.change):
            self.change=self.percent_change(self.yesterday_opening, self.today_opening)
        
        if not self.current_state:
            self.current_state=self.previous_state
            self.current_action=self.previous_action
        else:
            self.previous_state=self.current_state
            self.current_state=self.get_next_state(self.previous_action, self.previous_state)
            self.previous_action=self.current_action
            self.current_action=self.agent.getPolicy(self.current_state)
            
            if self.current_state in self.state_counter_dict:
                self.state_counter_dict[self.current_state]+=1
            else:
                self.state_counter_dict[self.current_state]=0
                
        epsilon=math.exp((-1)*self.epsilon_parameter*self.time)
        acceptance_probability=random.random()
        buy_arr=["sell","hold"]
        sell_arr=["buy","hold"]
        hold_arr=["buy","sell"]
        
        if acceptance_probability < epsilon:
            self.random_counter+=1
            if self.current_action=="buy":
                random.shuffle(buy_arr)
                self.current_action=buy_arr[0]
            elif self.current_action=="sell":
                random.shuffle(sell_arr)
                self.current_action=sell_arr[0]
            else:
                random.shuffle(hold_arr)
                self.current_action=hold_arr[0]
                
        if self.current_action=="buy":
             if self.Portfolio.Cash > self.today_opening:
                self.buy()
        elif self.current_action=="sell":
            if self.Portfolio.TotalPortfolioValue>0:
                self.sell()
        
        next_state_temp=self.get_next_state(self.current_action,self.current_state)
        
        self.agent.update(
            self.current_state,
            self.current_action,
            next_state_temp,
            self.compute_reward(
                self.current_action,
                self.change,
                self.current_state[0]))
                
        self.time+=1
        
    def compute_reward(self, action, percent_change, position):
        return percent_change*position
        
    def get_next_state(self, action, state):
        if not self.Portfolio.Invested:
            pos=0
        else:
            pos=1
        ret_state=(pos, self.change)
        return ret_state
        
    class Q_Learner:
        def __init__(self,discount,alpha):
            self.qValues=dict()
            self.discount=discount
            self.alpha=alpha
            self.actions=["buy","sell","hold"]
            
        def getQValue(self, state, actions):
            if (state, action) in self.qValues:
                return self.qValues[(state,action)]
            else:
                self.qValues[(state,action)]=0.0
                return 0.0
        
        def computeValueFromQValues(self,state):
            actions=self.actions
            max_val=max([self.getQValue(state,action) for action in actions])
            return max_val
            
        def computeActionFromQValues(self,state):
            max_val=None
            max_action=None
            for action in self.actions:
                curr_val=self.getQValue(state,action)
                if max_val==None or max_val<curr_val:
                    max_val=curr_val
                    max_action=action
            return max_action
        
        def update(self, state, action, nextState,reward):
            if(state, action) not in self.qValues:
                self.qValues[(state,action)]=0.0
            state_val=self.qValues[(state, action)]
            next_val=self.computeValueFromQValues(nextState)
            self.qValues[(state,action)]=state_val + self.alpha*(self.discount*next_val-state_val+reward)
            
        def getPolicy(self,state):
            return self.computeActionFromQValues