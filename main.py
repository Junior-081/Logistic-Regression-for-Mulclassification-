from logisticRegression import logregression_regularized
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv('data/data_iris.csv')
# sns.FacetGrid(data,hue="category", height=6).map(plt.scatter,"PetalLength","SepalWidth").add_legend()

def split_data(df, train_percent= 0.8):
    np.random.seed(2)
    perm = np.random.permutation(df.index)

    n= len(df)
    train_index = int(train_percent * n)

    train = df.iloc[perm[:train_index]]
    test = df.iloc[perm[train_index:]]

    x_train, x_test, y_train, y_test= train.iloc[:, :-3], test.iloc[:, :-3], train.iloc[:,-1], test.iloc[:,-1]
    return x_train.values, x_test.values, y_train.values, y_test.values

x_train, x_test, y_train, y_test= split_data(data) 

# print(f'X train size: {x_train.shape}')
# print(f'X test size: {x_test.shape}')
# print(f'y train size: {y_train.shape}')
# print(f'y test size: {y_test.shape}')
model1 = logregression_regularized(x_test,y_test,num_iters=10000)
print("")
print("")
print("COST FOR THE LAST ITERATION")
print("")
model1.fit(x_train, y_train)
print(" ")
print("PREDICTION")
print("")
print(model1.predict(x_test))
print("")
print("PROBABILITIES PREDICTED")
print("")
print(model1.predict_proba(x_test))
print("")
print(f"Accuracy is : {model1.accuracy(model1.predict(x_test),y_test)} ")
print("")
print("")
