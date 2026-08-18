import numpy as np
import pandas as pd
import sys

def load_test_train(traindata_filepath,testdata_filepath):
    train_data = pd.read_csv(traindata_filepath)
    test_data = pd.read_csv(testdata_filepath)
    return (train_data,test_data)

# Normalized Mean Absolute Error
def NMAE(y_pred,y_real):
    y_mean = np.mean(y_real)
    n = y_real.shape[0]
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator+=abs(y_real[i,0]-y_pred[i,0])
        denominator+=abs(y_real[i,0]-y_mean)
    return numerator/denominator

# Normalized Mean Squared Error
def NMSE(y_pred,y_real):
    y_mean = np.mean(y_real)
    n = y_real.shape[0]
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator+=(y_real[i,0]-y_pred[i,0])**2
        denominator+=(y_real[i,0]-y_mean)**2
    return numerator/denominator

def get_X_y(data,target_col=None):
    X = data.to_numpy()
    y = None
    if(target_col):
        X = data.drop(columns=[target_col]).to_numpy()
        y = data[target_col].to_numpy().reshape((-1,1))
    n,m = X.shape
    X = np.column_stack((np.ones(n), X))
    return (X,y)

def train(X,y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w
def predict(X,w):
    return X @ w
def main(train_path, test_path, pred_path, weights_path):
    train_data,test_data = load_test_train(train_path,test_path)

    X_train,y_train = get_X_y(train_data,"hr")
    X_test,_ = get_X_y(test_data)

    w = train(X_train,y_train)
    y_pred_test = predict(X_test,w)

    np.savetxt(pred_path, y_pred_test.flatten(), fmt="%.16f")
    np.savetxt(weights_path, w.flatten(), fmt="%.16f")

if __name__ == "__main__":
    train_path, test_path, pred_path, weights_path = sys.argv[1:5]
    main(train_path, test_path, pred_path, weights_path)
