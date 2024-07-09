#!/usr/bin/env python
# coding: utf-8

# In[2]:


from matplotlib import dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# In[3]:


raw=yf.download('SPY AAPL', start='2010-01-01', end='2019-12-31')


# In[4]:


raw


# In[5]:


raw.columns


# In[6]:


get_ipython().run_line_magic('pinfo', 'raw.pipe')


# In[7]:


def fix_cols(df):
    columns = df.columns
    outer=[col[0] for col in columns]
    df.columns=outer
    return df


# In[51]:


(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
)


# In[33]:


fix_cols(raw)


# In[8]:


def tweak_data():
    raw=yf.download('SPY AAPL', start='2010-01-01', end='2019-12-31')
    
    return (raw
    .iloc[:, ::2]
    .pipe(fix_cols)
    )


# In[35]:


tweak_data()


# In[56]:


(raw
.iloc[:, :-2:2]
.pipe(fix_cols)
.plot()
)


# In[54]:


(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .Close
 .plot()
)


# In[58]:


(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .Volume
 .plot(figsize=(10,2))
)


# In[63]:


(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .resample('M') #average closing price of each month
 .Close
 .mean()
)


# In[64]:


(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .resample('2M') #average for every two months
 .Close
 .mean()
)


# In[65]:


(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .resample('Q') #average for every two months
 .Close
 .mean()
)


# In[70]:


(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .resample('Q') #average of each quarter
 .Close
 .mean()
 .plot()
)


# In[78]:


fig, ax = plt.subplots(figsize=(10, 5))
def plot_candle(df, ax):
    #wick
    ax.vlines(x=df.index, ymin=df.Low, ymax=df.High, colors='k', linewidth=1)
    #red - decrease
    red=df.query('Open > Close')
    ax.vlines(x=red.index, ymin=red.Close, ymax=red.Open, colors='r', linewidth=3)
    #green - increase
    green=df.query('Open <= Close')
    ax.vlines(x=green.index, ymin=green.Close, ymax=green.Open, colors='g', linewidth=3)
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
    ax.xaxis.set_minor_locator(dates.DayLocator())
    return df

(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .resample('d')
 .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
 .loc['jan 2018': 'jun 2018']
 .pipe(plot_candle, ax)
)


# In[83]:


fig, ax = plt.subplots(figsize=(10, 5))
(raw
 .iloc[:, ::2]
 .pipe(fix_cols)
 .resample('d')
 .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
 .loc['sep 2019': '2019']
 .pipe(plot_candle, ax)
)


# In[10]:


aapl = (raw
        .iloc[:, ::2]
        .pipe(fix_cols)
)


# In[85]:


aapl


# In[87]:


# Returns
aapl.pct_change()


# In[88]:


(aapl
 .pct_change()
 .Close
 .plot()
)


# In[90]:


(aapl
 .pct_change()
 .Close
 .hist(bins=30)
)


# In[92]:


(aapl
 .pct_change()
 .Close
 .iloc[-100:]
 .plot.bar()
)


# In[95]:


fig, ax = plt.subplots(figsize=(10, 5))

(aapl
 .pct_change()
 .Close
 .iloc[-100:]
 .plot.bar(ax=ax)
)

ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
ax.xaxis.set_minor_locator(dates.DayLocator())


# In[97]:


# Returns plot using matplotlib
def my_bar(ser, ax):
    ax.bar(ser.index, ser)
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
    ax.xaxis.set_minor_locator(dates.DayLocator())
    return ser

fig, ax = plt.subplots(figsize=(10, 4))

_ = (aapl
   .pct_change()
   .Close
   .iloc[-100:]
   .pipe(my_bar, ax)
)


# In[100]:


(aapl
 .Close
 .sub(aapl.Close[0])
 .div(aapl.Close[0])
 .plot()
)


# In[101]:


(aapl
 .Close
 .pct_change()
 .add(1)
 .cumprod()
 .sub(1)
 .plot()
)


# In[102]:


def calc_cum_returns(df, col):
    ser=df[col]
    return(ser
           .sub(ser[0])
           .div(ser[0])
            )


# In[103]:


(aapl
 .pipe(calc_cum_returns, 'Close')
 .plot()
)


# In[104]:


def get_returns(df):
    return calc_cum_returns(df, 'Close')


# In[105]:


get_returns(aapl)


# In[106]:


# Using anonymous function lambda

(lambda df: get_returns(df))(aapl)


# In[107]:


(aapl
 .assign(cum_returns=lambda df:calc_cum_returns(df, 'Close'))
)


# In[108]:


def my_bar(ser, ax):
    ax.bar(ser.index, ser)
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
    ax.xaxis.set_minor_locator(dates.DayLocator())
    return ser


# In[109]:


fig, ax=plt.subplots(figsize=(10, 4))
_ = (aapl
    .pipe(calc_cum_returns, 'Close')
    .iloc[-100:]
    .pipe(my_bar, ax)
)


# In[110]:


(aapl
 .Close
 .mean()
)


# In[111]:


(aapl
 .Close
 .std()
)


# In[112]:


(aapl
 .assign(pct_change_close=aapl.Close.pct_change())
 .pct_change_close
 .std()
)


# In[117]:


(aapl
 .assign(close_vol=aapl.rolling(30).Close.std(),
        per_vol=aapl.Close.pct_change().rolling(30).std())
 .iloc[:, -2:]
 .plot(subplots=True)
)


# In[118]:


(aapl
 .assign(pct_change_close=aapl.Close.pct_change())
 .resample('15D')
 .std()
)


# In[119]:


(aapl
 .assign(pct_change_close=aapl.Close.pct_change())
 .rolling(window=15, min_periods=15)
 .std()
)


# In[123]:


(aapl
 .assign(pct_change_close=aapl.Close.pct_change())
 .rolling(window=15, min_periods=15)
 .std()
 ['pct_change_close']
 .plot()
)


# In[131]:


(aapl
 .assign(pct_change_close=aapl.Close.pct_change())
 .rolling(window=30, min_periods=30)
 .pct_change_close
 .std()
 .loc['2015' : '2019']
 .plot()
)


# In[11]:


(aapl
 .assign(s1=aapl.Close.shift(1),
        s2=aapl.Close.shift(2),
        ma3=lambda df_:df_.loc[:,['Close', 's1', 's2']].mean(axis='columns'),
        ma3_builtin=aapl.Close.rolling(3).mean()
        )
)


# In[14]:


(aapl
 .assign(s1=aapl.Close.shift(1),
        s2=aapl.Close.shift(2),
        ma3=lambda df_:df_.loc[:,['Close', 's1', 's2']].mean(axis='columns'),
        ma3_builtin=aapl.Close.rolling(3).mean()
        )
 [['Close', 'ma3']]
 .iloc[-200:]
 .plot()
)


# In[18]:


(aapl
 .assign(ma50=aapl.Close.rolling(50).mean(),
         ma200=aapl.Close.rolling(200).mean()
 )
 [['Close', 'ma50', 'ma200']]
 .iloc[-400:]
 .plot()
)


# In[32]:


(aapl
 .assign(ema_1=aapl.Close.ewm(alpha=0.0392).mean(),
        ema_2=aapl.Close.ewm(alpha=0.00995).mean()
        )
 [['Close', 'ema_1', 'ema_2']]
 .loc['2015':'2015']
 .plot()
)


# In[33]:


aapl.Close


# In[34]:


aapl.Close.shift()


# In[35]:


def calc_obv(df):
    df=df.copy()
    df["OBV"]=0.0
    
    for i in range (1, len(df)):
        if df["Close"][i] > df["Close"][i-1]:
            df["OBV"][i] = df["OBV"][i-1] + df["Volume"][i]
        elif df["Close"][i] < df["Close"][i-1]:
            df["OBV"][i] = df["OBV"][i-1] - df["Volume"][i]
        else:
            df["OBV"][i] = df["OBV"][i-1]
    return df


# In[36]:


calc_obv(aapl)


# In[37]:


get_ipython().run_cell_magic('timeit', '', 'calc_obv(aapl)\n')


# In[38]:


pd.Series(np.select(condlist=[aapl.Close < 7.6, aapl.Close > 72],
                   choicelist=[7.55, 72], default=33))


# In[39]:


(aapl
 .assign(vol=np.select([aapl.Close > aapl.Close.shift(1),
                       aapl.Close == aapl.Close.shift(1),
                       aapl.Close < aapl.Close.shift(1)],
                      [aapl.Volume, 0, -aapl.Volume]),
        obv=lambda df_:df_.vol.cumsum()
        )
)


# In[42]:


def calc_obv(df, close_col='Close', vol_col='Volume'):
    close = df[close_col]
    vol = df[vol_col]
    close_shift = close.shift(1)
    return (df.
           assign(vol=np.select([close > close_shift,
                                close == close_shift,
                                close < close_shift],
                               [vol, 0, -vol]),
                 obv=lambda df_:df_.vol.fillna(0).cumsum()
                 )
           ['obv']
           )


# In[43]:


(aapl
 .assign(obv=calc_obv)
)


# In[44]:


def calc_ad(df, close_col='Close', low_col='Low', high_col='High', vol_col='Volume'):
    close = df[close_col]
    low = df[low_col]
    high = df[high_col]
    return (df
           .assign(mfm=((close-low) - (high-close))/(high - low),
                   mfv=lambda df_:df_.mfm * df_[vol_col],
                   cmfv=lambda df_:df_.mfv.cumsum())
           .cmfv
           )


# In[48]:


(aapl
 .assign(ad=calc_ad)
 .ad
 .plot()
)


# In[50]:


def avg(df, col, window_size=14):
    results = []
    window = []
    for i, val in enumerate(df[col]):
        window.append(val)
        if i < (window_size):
            results.append(np.nan)
        elif i == (window_size):
            window.pop(0)
            results.append(sum(window)/window_size)
        else:
            results.append((results[-1] * (window_size-1) + val)/window_size)
            
    return pd.Series(results, index=df.index)


# In[51]:


(aapl
 .assign(change=lambda df:df['Close'].diff(),
        gain=lambda df:df.change.clip(lower=0),
        loss=lambda df:df.change.clip(upper=0),
        avg_gain=lambda df:avg(df, col='gain'),
        avg_loss=lambda df:-avg(df, col='loss'),
        rs=lambda df:df.avg_gain/df.avg_loss,
        rsi=lambda df:np.select([df.avg_loss==0], [100], (100-(100/(1+df.rs))))
        )
)


# In[ ]:




