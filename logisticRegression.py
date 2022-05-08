from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings( "ignore" )


class logregression_regularized:
    def __init__(self,x_test,y_test, num_iters= 100, threshold= 0.5, tolerance= 1e-10, lr= 0.00001,batch_size=32):
        self.batch_size=batch_size
        self.num_iters=num_iters
        self.threshold=threshold
        self.tolerance=tolerance
        self.lr=lr
        self.theta=None
        self.cost_history=[]
        self.cost_history_test=[]
        self.x_test=x_test
        self.y_test=y_test
        self.losses=[]


    def add_ones(self, x):
        return np.hstack([np.ones((x.shape[0],1)),x])
    

    def sigmoid(self, x, theta):
        z = (x @ theta) #we don't put x.T because is not for one row but on every dataset y.T@(x @ theta)
        s = 1 / (1 + np.exp(-z))
        return s

    def cross_entropy(self, x, y_true):
        n =len(x) # Length of x
        z=-y_true*(x@self.theta)
        a = 1 / (1 + np.exp(-z))
        Cost = np.sum(np.log(a))
        return Cost/n 
    
    def fit(self, x,y):
        x= self.add_ones(x) # Add ones to x
        y= y.reshape(-1,1) # reshape y. This is optional, do it if needed
        self.theta= np.zeros((x.shape[1], 1)) # Initialize theta to zeros vector >>> (x.shape[1])
        current_iter= 1
        norm= 1

        self.x_test= self.add_ones(self.x_test) # Add ones to x
        self.y_test= self.y_test.reshape(-1,1) # reshape y. This is optional, do it if needed

        while (norm >= self.tolerance and current_iter < self.num_iters):
            dataset=np.c_[x,y]
            np.random.seed(3)
            np.random.shuffle(dataset)
            x=dataset[:,:-1]
            y=dataset[:,-1]
            theta_old = self.theta.copy() # Get old theta
            
            #mini-batch
            for i in range(0,len(x),self.batch_size):
                end=i+self.batch_size
                x_=x[i:end,:]
                y_=y[i:end].reshape(-1,1)

          # make predictions 
                z=-y_*(x_@self.theta)
                a = 1 / (1 + np.exp(-z))
                #y_pred= self.sigmoid(x_,self.theta) # using sigmoid function 
              # Gradient of cross-entropy
#                 print(f"y_pred: {y_pred}")
#                 print(f"shape of y_pred is {y_pred.shape} and x_ is {x_.T.shape} and y is {y_.shape}")
                
                grad= -x_.T@ (a*y_) #y_pred*(x_.T@ y_)
        
                grad= grad.reshape(-1,1) # Reshape, if it is needed

              # update rules
                self.theta= self.theta - grad * self.lr
              # Compute the training loss
                self.losses.append(self.cross_entropy(x_,y_))

            #training_loss=self.cross_entropy(x,y)
       
            self.cost_history.append(np.mean(self.losses))
            test_loss=self.cross_entropy(self.x_test,self.y_test)
            self.cost_history_test.append(test_loss)

              # Convergence criteria:
            if current_iter%100 == 0:
                print(f'cost for {current_iter} iteration : {self.cross_entropy(x_, y_)}')
            norm = np.linalg.norm(theta_old - self.theta)
            current_iter += 1

    def predict(self, x):
        proba= self.predict_proba(x) # Get probability of x
        result= [1 if i > self.threshold else -1 for i in proba] # Convert proba to 0 or 1. hint: list comprehension
        return np.array(result) 
  
    
    def predict_proba(self, x):
        x=self.add_ones(x) # Apply add ones to x
        y_pred_prob= self.sigmoid(x, self.theta) # Predict proba with sigmoid
        return y_pred_prob
    def accuracy(self,ypred,y_test):
        return (np.sum(y_test==ypred)/len(y_test))*100

    def plot(self):
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.plot(np.arange(len(self.cost_history)), self.cost_history, 'r', linewidth = "2", label= 'Train Loss')
        plt.plot(np.arange(len(self.cost_history_test)), self.cost_history_test, 'b', linewidth = "2", label= 'Validation Loss')
        plt.legend()
        plt.show()