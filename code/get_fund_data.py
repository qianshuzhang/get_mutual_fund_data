import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

import pyarrow.feather as feather
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.regression.rolling import RollingOLS
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import datetime
import select_func

##########################
#     load raw data      #
##########################
datapath = '/home/qianshu/mutual_fund/data/qs1/'

with open(datapath+'style.feather','rb') as f:
    style=feather.read_feather(f)
with open(datapath+'names.feather','rb') as f:
    names=feather.read_feather(f)
with open(datapath+'fund_summary.feather','rb') as f:
    fund_summary=feather.read_feather(f)
last_names = names.drop_duplicates(subset=['crsp_fundno'], keep='last')
with open(datapath+'mflink.feather', 'rb') as f:
    mflink=feather.read_feather(f)
with open(datapath+'tna_ret_nav.feather','rb') as f:
    tna_ret_nav=feather.read_feather(f)
tna_ret_nav['mtna'] = tna_ret_nav['mtna'].replace(0,np.nan)
tna_ret_nav.dropna(subset=['mret','mtna'],inplace=True)

with open(datapath+'fund_fees.feather','rb') as f:
    fund_fees=feather.read_feather(f)
with open(datapath+'holdings.feather','rb') as f:
    holdings=feather.read_feather(f)
with open(datapath+'ff_monthly.feather', 'rb') as f:
    ff_monthly=feather.read_feather(f)
with open(datapath+'fund_char_result.feather','rb') as f:
    fund_char_result=feather.read_feather(f)
add_factor = pd.read_csv('/home/qianshu/US_factor/factor/30_factors.csv')
macro = pd.read_csv('/home/qianshu/mutual_fund/fund_tree/data/addition_factors_20220827.csv')
##########################
# several objective code #
##########################

crsp_policy_to_exclude = ['C & I', 'Bal', 'Bonds', 'Pfd', 'B & P', 'GS', 'MM', 'TFM']
lipper_class = ['EIEI', 'G', 'LCCE', 'LCGE', 'LCVE', 'MCCE', 'MCGE', 'MCVE',
                'MLCE', 'MLGE', 'MLVE', 'SCCE', 'SCGE', 'SCVE']
lipper_obj_cd = ['CA', 'EI', 'G', 'GI', 'MC', 'MR', 'SG']
si_obj_cd = ['AGG', 'GMC', 'GRI', 'GRO', 'ING', 'SCG']
wbrger_obj_cd=['G', 'G-I', 'AGG', 'GCI', 'GRI', 'GRO', 'LTG', 'MCG','SCG']

def get_equity_fund(style,last_names,fund_summary,mflink):
    # investing on average less than 80% of their assets, excluding cash, in common stocks
    style = select_func.get_per_com(fund_summary,style)

    # the first way to select funds
    funds = select_func.first_select(style,lipper_class,lipper_obj_cd,si_obj_cd,wbrger_obj_cd,crsp_policy_to_exclude)

    # the second way to select funds
    funds2 = select_func.second_select(style)

    # merge long way and short way
    funds3 = pd.merge(funds,funds2)

    # look for funds that have flip-flopped their style; can choose funds, funds2 or funds3
    funds4 = select_func.get_flipper(funds2,style)

    # choose max flipper as flipper 
    funds5 = funds4.groupby('crsp_fundno', group_keys=False).apply(lambda x: x.loc[x.flipper.idxmax()])
    funds5.reset_index(drop=True, inplace=True)

    # merge name 
    funds6 = pd.merge(funds5, last_names, on='crsp_fundno')

    # identify index and target date funds and drop them from the sample
    funds7 = select_func.del_index_fund(funds6)

    # final equity funds
    equity_funds = pd.merge(funds7, mflink, on='crsp_fundno')
    equity_funds = equity_funds[equity_funds['flipper'] == 0]

    return equity_funds

equity_funds = get_equity_fund(style,last_names,fund_summary,mflink)

returns_tmp = select_func.get_tna_ret(tna_ret_nav,names,fund_fees,equity_funds)

# aggregate multiple share class
returns = select_func.aggregate(returns_tmp)

# We also exclude fund observations before a fund passes the $5 million threshold for assets under management (AUM).
# All subsequent observations, including those that fall under the $5 million AUM threshold in the future, are included.
mtna_group = 5
returns = select_func.ex_tna(mtna_group, returns)

# exclude funds with TNA on average less than $5 million
returns = select_func.ex_avg_tna(mtna_group, returns)

# exclude funds with less than 36 months of observations
month = 36
returns = select_func.ex_obs(returns,month)

# at least 30 ret obs in last 36 months
time_interval = 36
least_month = 30
returns = select_func.ex_least_obs(returns, time_interval, least_month)

returns = returns[['wficn','date','mret','mtna','rret','turnover','exp_ratio','age','flow','vol','flow_vol','mgr_tenure']]

# calculate abnormal return : mret adjusted by four factor
# returns_abr = select_func.get_abr(ff_monthly, returns)

# get holding
returns_char = select_func.get_holding(returns, fund_char_result)

# merge factor and macro,standardize
returns_char = select_func.rank_macro_factor(returns_char, add_factor, macro)

# get fund momentum
returns_char = select_func.get_fund_mom(returns_char)

# get fund capm rvar alpha beta
returns_char = select_func.get_fund_capm(returns_char)

# get fund ff4 alpha
returns_char = select_func.get_fund_ff4(returns_char)

# standardize fund charateristics
returns_char = select_func.standard_fund_char(returns_char)

# shift fund return,date to get lag char
returns_char = select_func.shift_fund_date_ret(returns_char)

# get summary table
summary = select_func.get_summary(returns_char)
