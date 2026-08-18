import numpy as np
import pandas as pd
import sys

def load_data(train_path, test_path, folds_path, reg_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    with open(folds_path, 'r') as file:
        text = file.read()
    array_string = text.split('[')[1].split(']')[0]
    fold_ends = [int(num) for num in array_string.split(',')]
    
    # Reconstruct the row-by-row folds array based on endpoints
    n_samples = train_data.shape[0]
    folds = np.zeros(n_samples, dtype=int)
    
    start_idx = 0
    for k, end_idx in enumerate(fold_ends):
        # Assign the fold index 'k' to the appropriate slice of rows
        folds[start_idx:end_idx] = k
        start_idx = end_idx

    # Load regularization values
    regularization_data = pd.read_csv(reg_path, header=None).values.flatten()
    
    return train_data, test_data, folds, regularization_data

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

def train(X, y, l):
   
    cols = X.shape[1]
    R = np.eye(cols)
    R[0, 0] = 0
    w = np.linalg.inv(X.T @ X + l * R) @ X.T @ y
    return w
def predict(X,w):
    return X @ w

def main():
    if len(sys.argv) != 9:
        print("Usage: python3 part_b.py train.csv test.csv folds.txt regularization.txt predictions.txt weights.txt bestlambda.txt crossvalidation_errors.txt")
        sys.exit(1)
        
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    folds_path = sys.argv[3]
    reg_path = sys.argv[4]
    pred_path = sys.argv[5]
    weights_path = sys.argv[6]
    best_lambda_path = sys.argv[7]
    cv_errors_path = sys.argv[8]
    train_df, test_df, folds, lambdas = load_data(train_path, test_path, folds_path, reg_path)

    X_train, y_train = get_X_y(train_df, target_col='hr')
    X_test, _ = get_X_y(test_df)

    cv_results = []
    best_lambda = None
    min_cv_nmse = float('inf')

    for l in lambdas:
        fold_errors = []
        for k in range(5):
            train_idx = (folds != k)
            val_idx = (folds == k)
            
            X_train_cv, y_train_cv = X_train[train_idx], y_train[train_idx]
            X_val_cv, y_val_cv = X_train[val_idx], y_train[val_idx]
            
            w_cv = train(X_train_cv, y_train_cv, l)
            y_pred_cv = predict(X_val_cv, w_cv)
            
            nmse_k = NMSE(y_pred_cv, y_val_cv)
            fold_errors.append(nmse_k)
            
        cv_nmse = np.mean(fold_errors)
        cv_results.append((l, cv_nmse))
        
        if cv_nmse < min_cv_nmse:
            min_cv_nmse = cv_nmse
            best_lambda = l

    w_final = train(X_train, y_train, best_lambda)
    
    y_test_pred = predict(X_test, w_final)

    np.savetxt(pred_path, y_test_pred, fmt='%.16f')

    # Write weights
    np.savetxt(weights_path, w_final, fmt='%.16f')
    
    # Write best lambda
    with open(best_lambda_path, 'w') as f:
        f.write(f"{best_lambda}\n")
        
    # Write cross-validation errors as comma-separated pairs
    with open(cv_errors_path, 'w') as f:
        for l, err in cv_results:
            f.write(f"{l},{err:f}\n")

if __name__ == "__main__":
    main()