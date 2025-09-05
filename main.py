import torch
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd
import time
import math
from Min_Var import min_var
from Mean_Var import mean_var
from l0_MSSIT_Net import l0_MSSIT_Net
from SSPO import SSPO_fun

# stocks data csv read
df = pd.read_csv('data.csv')
df = df.set_index('Date')

# s&p data csv read
df_sp = pd.read_csv('sp500.csv')
df_sp = df_sp.set_index('Date')
rbp = 1  # rebalancing period = one month


def data_process(df):
    df = df.pct_change()
    df = df.tail(-1)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    df = df.iloc[-1, :]
    df = df.to_numpy()
    df = torch.from_numpy(df).type(torch.Tensor)
    return df


def data_process_m(df, m):
    df = df.pct_change()
    df = df.tail(-1)
    df = df.tail(m)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    df = df.iloc[:, :]
    df = df.to_numpy()
    df = torch.from_numpy(df).type(torch.Tensor)
    return df


def date_slicer(df, start, duration, rebalancing_period=0):
    '''
    this function is used to slice out specific section of the data
    '''
    start = str(datetime.strptime(start, '%Y-%m-%d').date() + rd(months=rebalancing_period))
    end = str(datetime.strptime(start, '%Y-%m-%d').date() + rd(months=duration) - rd(days=1))
    output = df.loc[start:end]
    return output


def mday_cw(df, m):
    df = df.pct_change()
    df = df.tail(-m)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    df = df.iloc[-1, :]
    df = df.to_numpy()
    df = torch.from_numpy(df).type(torch.Tensor)
    return df


def weight_mday_cw(df, w, m):

    df = df.pct_change()
    df = df.tail(m)
    df = df.to_numpy()
    weighted_returns = torch.tensor(df, dtype=torch.float32) * w
    cumulative_returns = torch.cumprod(1 + weighted_returns, dim=0) - 1
    cumulative_returns = cumulative_returns[-1]
    cumulative_returns_tensor = cumulative_returns.clone().detach().float()
    return cumulative_returns_tensor


def daily_change(df):
    df = df.pct_change()
    df = df.tail(-1)
    return df


def daily_return(df):
    df = df.pct_change()
    df = df.tail(-1)
    df = df + 1
    return df


def RMSE(x, y, weights):
    temp = 0
    for i in range(len(x)):
        weighted_sum = np.dot(x.iloc[i].values, weights)
        temp += (weighted_sum - y.iloc[i]) ** 2
    return math.sqrt(temp / len(x))


def test_fun(x_test_m, q_t, i, model):
    start_time = time.time()
    x_change = daily_change(date_slicer(df, '2016-07-01', 6, i))
    y_change = daily_change(date_slicer(df_sp, '2016-07-01', 6, i))
    weights = np.array(model(x_test_m, q_t).detach())
    test_rmse = RMSE(x_change, y_change, weights)

    print(f'\nl0_MSSIT_Net test Results for model {(i / rbp) + 1}:')
    print(f'Test RMSE: {test_rmse}')
    print(f'l0_MSSIT_Net Testing time: {time.time() - start_time:.2f}')
    return test_rmse


def train_fun(model, optimizer, x_train, y_train, q_t, i):
    loss_fun = torch.nn.MSELoss(reduction='mean')
    start_time = time.time()
    num_epochs = 100
    print(f'\nl0_MSSIT_Net Training & Results for model {(i / rbp) + 1}:')
    epoch_losses = []
    for epoch in range(num_epochs):
        output = model(x_train, q_t)
        cumulative_change = sum(output * x_train)
        loss_l0_MSSIT_Net = loss_fun(cumulative_change, y_train)
        epoch_losses.append(loss_l0_MSSIT_Net.item())
        if epoch == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch + 1} of {num_epochs} | MSE: {loss_l0_MSSIT_Net.item()}')
        optimizer.zero_grad()
        loss_l0_MSSIT_Net.backward()
        optimizer.step()
    training_time = format(time.time() - start_time, '0.2f')
    print(f'Training time: {training_time}')
    return model


def main():
    l0_MSSIT_model_test_results = []
    for i in range(int(24 / rbp)):
        x_train = data_process(date_slicer(df, '2014-08-01', 36, i * rbp))
        x_train_m = data_process_m(date_slicer(df, '2014-08-01', 36, i * rbp), 120)
        y_train = data_process(date_slicer(df_sp, '2014-08-01', 36, i * rbp))
        x_test_m = data_process_m(date_slicer(df, '2016-07-01', 1, i * rbp), 19)

        sspo_weight = SSPO_fun(x_train)
        minvar_weight = min_var(x_train_m)
        meanvar_weight = mean_var(x_train_m)
        equal_weight = torch.ones_like(x_train) / 471

        # Construct characteristics matrix
        five_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 5)
        ten_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 10)
        fifteen_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 15)
        twenty_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 20)
        twentyfive_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 25)
        thirty_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 30)

        five_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 5)
        ten_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 10)
        fifteen_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 15)
        twenty_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 20)
        twentyfive_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 25)
        thirty_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 30)

        five_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 5)
        ten_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 10)
        fifteen_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 15)
        twenty_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 20)
        twentyfive_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 25)
        thirty_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 30)

        five_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 5)
        ten_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 10)
        fifteen_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 15)
        twenty_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 20)
        twentyfive_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 25)
        thirty_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 30)

        five_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weight, 5)
        ten_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weight, 10)
        fifteen_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weight, 15)
        twenty_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weight, 20)
        twentyfive_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weight, 25)
        thirty_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weight, 30)

        # Combine the asset characteristics
        q_t = torch.stack([
            five_day_cw, ten_day_cw, fifteen_day_cw, twenty_day_cw, twentyfive_day_cw, thirty_day_cw,
            five_day_sspo, ten_day_sspo, fifteen_day_sspo, twenty_day_sspo, twentyfive_day_sspo, thirty_day_sspo,
            five_day_minvar, ten_day_minvar, fifteen_day_minvar, twenty_day_minvar, twentyfive_day_minvar, thirty_day_minvar,
            five_day_equal, ten_day_equal, fifteen_day_equal, twenty_day_equal, twentyfive_day_equal, thirty_day_equal,
            five_day_meanvar, ten_day_meanvar, fifteen_day_meanvar, twenty_day_meanvar, twentyfive_meanvar, thirty_day_meanvar
        ])
        model = l0_MSSIT_Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = train_fun(model, optimizer, x_train_m, y_train, q_t, i * rbp)
        l0_MSSIT_model_test_results.append(test_fun(x_test_m, q_t, i * rbp, model))

    Average_test_rmse = sum(l0_MSSIT_model_test_results) / 24
    print(f'Selected Model Test Results are:')
    print(f'l0_MSSIT_Net best RMSE =', min(l0_MSSIT_model_test_results))
    print('l0_MSSIT_Net mean RMSE =:', Average_test_rmse)


if __name__ == "__main__":
    main()

