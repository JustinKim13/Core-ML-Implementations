import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# q1: data visualization
def plot_data(filename):
    
    # read in data
    data = pd.read_csv(filename)
    x, y = data['year'].values, data['days'].values
    
    # plot data
    plt.plot(x, y)
    plt.xlabel("Year")
    plt.xticks(x.astype(int)) # want full years, no decimals
    plt.ylabel("Number of Frozen Days")
    
    # save and close plot
    plt.savefig("data_plot.jpg")
    plt.close()
    
# q2: data normalization
def normalize_data(filename):
    # read in data
    data = pd.read_csv(filename)
    X, y = np.array(data['year'].values), np.array(data['days'].values)

    # get vals for normalization
    m, M = min(X), max(X)
    
    # perform min-max normalization on x
    X_norm = (X - m) / (M - m)
    X_norm = np.column_stack((X_norm, np.ones(len(X)))) # add a bias col, filled with 1 for each xi, which we will then multiply with [w b]
    
    # y_hat (pred) = [w b] @ [Xi 1]^T  
    # we now have the [Xi 1]^T part!
            
    return X_norm, y, m, M # return for future use
  
# q3: closed-form solution to linear regression
def closed_form_solution(X, y):
    weights = np.linalg.inv(X.T @ X) @ (X.T @ y) # simply following the equation: (w b) = (X^T @ X)^-1 @ (X^T @ Y)
    return weights

# q4: linear regression with gradient descent
def grad_descent(X, y, lr, iterations):
    
    loss_per_iter = [] # we'll use this to plot our losses per iteration
    weights = np.zeros(X.shape[1]) # [w, b] initialized as vector [0, 0] 
        
    # y = y.reshape(-1, 1) # make sure y is a column vector of size (n, 1)
    
    for t in range(iterations):
        y_hat = X @ weights # multiply normalized feature matrix with weights matrix to get predictions -> y_hat is a col vector, containing pred for each row
        grad = (1 / len(y)) * (X.T @ (y_hat - y)) # calculate gradient (rate of change of loss) -> compare y_hat to actual y label to get gradient
        
        # print every 10 iterations
        if t % 10 == 0:
            print(weights)
        
        weights -= lr * grad # update weights after printing
        
        # save our MSE loss history
        loss_per_iter.append((1 / (2 * len(y)) * np.sum((y_hat - y) ** 2)))
        
    return weights, loss_per_iter

def loss_per_iter_plot(loss_per_iter):
    plt.plot(range(len(loss_per_iter)), loss_per_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")
    plt.close()
    
# q5: prediction
def predict(x, weights, m, M):
    # normalize x using min-max scaling
    x_norm = (x - m) / (M - m) # normalize our specific x value (year)
    
    #  predict with closed-form weights
    y_hat = weights[0] * x_norm + weights[1] # equation: y_hat = wx + b
    return y_hat # our prediction for that year

# q6: model interpretation
def symbol(w):
    if w > 0: 
        return ">"
    elif w < 0: 
        return "<"
    else:
        return "="

# q7: model limitations
def no_freeze_prediction(weights, m, M):
    w, b = weights # get our weight and bias vals
    
    # equation: x_star (pred) = m + (-b(M - m)) / w  <- after doing some simple manipulations
    x_star = m + (-b * (M - m)) / w 
    return x_star # the year by which Lake Mendota will no longer freeze

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py arg1 arg2 arg3")
        sys.exit(1)
        
    filename, learning_rate, iterations = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    
    # q1: data visualization
    plot_data(filename)
    
    # q2: data normalization
    X_norm, y, m, M = normalize_data(filename)
    print("Q2:")
    print(X_norm)
    
    # q3: closed-form solution to linear regression
    weights_closed_form = closed_form_solution(X_norm, y)
    print("Q3:")
    print(weights_closed_form)
    
    # q4: linear regression with gradient descent 
    print("Q4a:")
    grad_descent_weights, loss_per_iter = grad_descent(X_norm, y, learning_rate, iterations)
    print("Q4b: 0.3")
    loss_per_iter_plot(loss_per_iter)
    print("Q4c: 500")
    print("Q4d: I started with a small learning rate (0.01) and 100 epochs. This was very off, so I then upped the epochs to 250, and then 500, still being far from the closed-form solution. " +
          "I then started turning the learning rate up, while still keeping epochs at 500. I started with 0.1, which got me slightly closer, and I kept increasing it until I got a value that was " +
          "extremely close to the closed-form solution, using a learning rate of 0.3 and keeping epochs at 500. The weight and bias were still converging, so I knew my learning rate wasn't too high. ")
    
    # q5: prediction
    y_hat = predict(2023, weights_closed_form, m, M) # x is 2023 for the year we're predicting, we'll use the closed-form weights, and we can reuse m and M since we still normalize the same
    print("Q5: " + str(y_hat))
    
    # q6: model interpretation
    sym = symbol(weights_closed_form[0])
    print("Q6a: " + sym)
    print("Q6b: If the weight is greater than 0, it means that the total number of frozen days increases from year to year. " +
               "If the weight is less than 0, it means that the total number of frozen days decreases from year to year. " + 
               "If the weight is equal to 0, it means that the total number of frozen days remains constant from year to year.")

    # q7: model limitations
    x_star = no_freeze_prediction(weights_closed_form, m, M)
    print("Q7a: " + str(x_star))
    print("Q7b: x^*, being ~1813, is not a compelling prediction because it is in the past, which is unrealistic. This model also assumes linearity, but the effects of climate change are often " +
          "nonlinear, and may accelerate over time. It also doesn't take into account any other factors, causing unreliable long-term predictions.")
    
if __name__ == "__main__":
    main()