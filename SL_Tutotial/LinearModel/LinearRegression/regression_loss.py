import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mse(true,pred):
    """
    true: array of true values
    pred: array of predicted values

    returns: mean square error loss
    """
    return np.sum((true - pred)**2)
def mae(true,pred):
    return np.sum(np.abs(true-pred))

def sm_mae(true, pred, delta):
    # Smooth Mean Absolute Error/ Huber Loss
    loss = np.where(np.abs(true-pred) < delta, 0.5*((true-pred)**2),delta*np.abs(true-pred)-0.5*(delta**2))
    return np.sum(loss)

def logcosh(true,pred):
    loss = np.log(np.cosh(pred-true))
    return np.sum(loss)

if __name__ == "__main__":
    fig, ax1 = plt.subplots(1,1,figsize = (7,5))

    target = np.repeat(0,1000)
    pred = np.arange(-10,10,0.02)
    delta = [0.1, 1, 10]

    loss_mse = [mse(target[i], pred[i]) for i in range(len(pred))]
    loss_mae = [mae(target[i], pred[i]) for i in range(len(pred))]
    loss_sm_mae1 = [sm_mae(target[i],pred[i], 5) for i in range(len(pred))]
    loss_sm_mae2 = [sm_mae(target[i], pred[i], 10) for i in range(len(pred))]
    loss_logcosh = [logcosh(target[i], pred[i]) for i in range(len(pred))]

    losses = [loss_mse, loss_mae, loss_sm_mae1, loss_sm_mae2, loss_logcosh]
    names = ['MSE', 'MAE', 'Huber (5)', 'Huber (10)', 'Log-cosh']
    cmap = ['#d53e4f',
            '#fc8d59',
            '#fee08b',
            '#e6f598',
            '#99d594',
            '#3288bd']

    for lo in range(len(losses)):
        ax1.plot(pred,losses[lo], label=names[lo], color = cmap[lo])

    plt.xlabel('Predictions')
    plt.ylabel('Loss')
    plt.title('MSE Loss vs. Predictions')
    ax1.legend()
    ax1.set_ylim(bottom=0, top=40)
    plt.show()