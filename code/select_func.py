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

def get_per_com(fund_summary,style):
    fund_summary['per_com']=fund_summary['per_com'].replace(np.nan,0)
    per_com = fund_summary.groupby(['crsp_fundno']).apply(
        lambda x: np.nansum(x['per_com']) / np.count_nonzero(x['per_com']))
    per_com = per_com.reset_index()
    per_com.columns = ['crsp_fundno', 'per_com']
    style = pd.merge(style, per_com)
    style = style[style['per_com'] >= 80]
    return style

def first_select(style,lipper_class,lipper_obj_cd,si_obj_cd,wbrger_obj_cd,crsp_policy_to_exclude):
    funds = style[(style['lipper_class'].isin(lipper_class)) | (style['lipper_obj_cd'].isin(lipper_obj_cd)) | (
    style['si_obj_cd'].isin(si_obj_cd)) | (style['wbrger_obj_cd'].isin(wbrger_obj_cd))]
    funds = funds[~funds['policy'].isin(crsp_policy_to_exclude)]
    funds['long_way'] = 1
    funds = funds[['crsp_fundno', 'long_way']]
    funds.drop_duplicates(inplace=True)
    return funds

'''
E:Equity
D:Domestic
C Y : Cap-based Style
'''
def second_select(style):
    funds2 = style[(style['crsp_obj_cd'].str[:1] == 'E') & (style['crsp_obj_cd'].str[1:2] == 'D') & (
        style['crsp_obj_cd'].str[2:3].isin(['C', 'Y'])) & (~style['crsp_obj_cd'].str[2:4].isin(['YH', 'YS'])) & (
                        style['si_obj_cd'] != 'OPI')]
    funds2['short_way'] = 1
    funds2 = funds2[['crsp_fundno', 'short_way']]
    funds2.drop_duplicates(inplace=True)
    return funds2

def get_flipper(funds2, style):
    funds4 = pd.merge(funds2, style[['crsp_fundno', 'crsp_obj_cd', 'begdt']], on=['crsp_fundno'],
                  sort=['crsp_fundno', 'begdt'])
    funds4['flipper'] = 0
    funds4.loc[~funds4['crsp_obj_cd'].str[0:3].isin(['EDC', 'EDY']), 'flipper'] = 1
    return funds4


def del_index_fund(funds6):
    counties = funds6.groupby('crsp_fundno')['index_fund_flag'].sum() == 0
    funds7 = funds6.loc[counties.values == True]
    funds7['namex'] = funds7['fund_name'].str.lower()

    funds7 = funds7[(~funds7['namex'].str.contains('index')) & (~funds7['namex'].str.contains('s&p')) & (
        ~funds7['namex'].str.contains('idx')) & (
                        ~funds7['namex'].str.contains('etf')) &
                    (~funds7['namex'].str.contains('exchange traded')) & (
                        ~funds7['namex'].str.contains('exchange-traded')) &
                    (~funds7['namex'].str.contains('target')) & (~funds7['namex'].str.contains('2005')) &
                    (~funds7['namex'].str.contains('2010')) & (~funds7['namex'].str.contains('2015')) &
                    (~funds7['namex'].str.contains('2020')) & (~funds7['namex'].str.contains('2025')) &
                    (~funds7['namex'].str.contains('2030')) & (~funds7['namex'].str.contains('2035')) &
                    (~funds7['namex'].str.contains('2040')) & (~funds7['namex'].str.contains('2045')) &
                    (~funds7['namex'].str.contains('2050')) & (~funds7['namex'].str.contains('2055'))]
    return funds7

def get_tna_ret(tna_ret_nav,fund_fees,equity_funds):
    returns1 = pd.merge(equity_funds, tna_ret_nav, on='crsp_fundno')
    returns1.rename(columns={'caldt': 'date'}, inplace=True)
    returns2 = pd.merge(returns1, fund_fees, on='crsp_fundno')
    returns2 = returns2[(returns2['date'] >= returns2['begdt_y']) & (returns2['date'] <= returns2['enddt'])]
    returns2['rret'] = returns2['mret'] + returns2['exp_ratio'] / 12
    returns2.sort_values(by=['crsp_fundno', 'date','wficn'], inplace=True)

    returns2['flag'] = (returns2.crsp_fundno != returns2.crsp_fundno.shift()).astype(int)
    returns2['weight'] = returns2['mtna'].shift()
    returns2.loc[returns2['flag'] == 1, 'weight'] = returns2.loc[returns2['flag'] == 1, 'mtna']

    # keep only observations dated after the fund's first offer date.
    returns2 = returns2[returns2['date'] > returns2['first_offer_dt']]
    returns2=returns2[returns2['date']>returns2['begdt_x']]
    returns2.sort_values(by=['wficn', 'date'], inplace=True)   

    # get share class weight, for aggregating
    returns2['flag'] = (returns2.crsp_fundno != returns2.crsp_fundno.shift()).astype(int)
    returns2['weight'] = returns2['mtna'].shift()
    returns2.loc[returns2['flag'] == 1, 'weight'] = returns2.loc[returns2['flag'] == 1, 'mtna']
    returns2['age'] = (returns2['end_dt'] - returns2['first_offer_dt'])
    
    # keep only observations dated after the fund's first offer date.
    returns2 = returns2[returns2['date'] > returns2['first_offer_dt']]
    returns2=returns2[returns2['date']>returns2['begdt_x']]

    return returns2

def aggregate(returns_tmp):
    mret = returns_tmp.groupby(['wficn', 'date']).apply(lambda x: np.nansum(x['mret']*x['weight'])/np.nansum(x['weight']))
    returns = mret.reset_index()
    returns.columns = ['wficn', 'date', 'mret']

    # monthly total net assets
    mtna = returns_tmp.groupby(['wficn', 'date']).apply(lambda x: np.nansum(x['mtna']))
    mtna = mtna.reset_index()
    returns['mtna'] = mtna[0]

    # gross return
    rret = returns_tmp.groupby(['wficn', 'date']).apply(lambda x: np.nansum(x['rret']*x['weight'])/np.nansum(x['weight']))
    rret = rret.reset_index()
    returns['rret'] = rret[0]

    # turnover ratio
    turnover = returns_tmp.groupby(['wficn', 'date']).apply(lambda x: np.nansum(x['turn_ratio']*x['weight'])/np.nansum(x['weight']))
    turnover = turnover.reset_index()
    returns['turnover'] = turnover[0]

    # expense ratio
    expense = returns_tmp.groupby(['wficn', 'date']).apply(lambda x: np.nansum(x['exp_ratio']*x['weight'])/np.nansum(x['weight']))
    expense = expense.reset_index()
    returns['exp_ratio'] = expense[0]
    returns['exp_ratio']=returns['exp_ratio']/12

    # fund age
    age = returns_tmp.groupby(['wficn', 'date']).apply(lambda x: np.max(x['age']))
    age = age.reset_index()
    returns['age'] = age[0]

    # fund flow
    returns['cum_ret'] = (1 + returns['mret']).rolling(12).apply(np.prod, raw=True) - 1
    returns['flow'] = (returns['mtna'] - returns.groupby(['wficn'])['mtna'].shift(12) * returns['cum_ret']) / \
                    returns.groupby(['wficn'])['mtna'].shift(12)
    returns.loc[returns['flow'] > 1000, 'flow'] = np.nan
    returns['flow']=returns['flow'].replace(np.inf,np.nan)
    returns['flow']=returns['flow']*100

    # fund vol
    returns['vol'] = returns.groupby(['wficn'])['mret'].rolling(12).std().reset_index()['mret']

    return returns

# We also exclude fund observations before a fund passes the $5 million threshold for assets under management (AUM).
# All subsequent observations, including those that fall under the $5 million AUM threshold in the future, are included.

def ex_tna(returns, mtna_group = 5):
    returns['tna_ind'] = 0
    for w in returns['wficn'].unique():
        for index, row in returns[returns['wficn'] == w].iterrows():
            if row['mtna'] < mtna_group:
                returns.at[index, 'tna_ind'] += 1
            else:
                break

    returns.drop(index=returns[returns['tna_ind'] == 1].index, inplace=True)
    del returns['tna_ind']

    return returns

def ex_avg_tna( returns, mtna_group = 5):
    avg_tna=returns.groupby(['wficn'])['mtna'].mean()
    avg_tna=avg_tna.reset_index()
    avg_tna.columns=['wficn','avg_tna']
    returns=pd.merge(returns,avg_tna)
    returns = returns[returns['avg_tna'] >= mtna_group]
    del returns['avg_tna']

    return returns

def ex_obs(returns, month = 36):
    obs = returns.groupby(['wficn'])['mtna'].count()
    obs = obs.reset_index()
    obs.columns = ['wficn', 'obs']
    returns = pd.merge(returns, obs)
    returns = returns[returns['obs'] >= month]

    return returns

def ex_least_obs(returns, time_interval = 36, least_month = 30):
    returns['ret_bool']=returns['mret'].isna()
    aum = returns.sort_values(by=['wficn', 'date'], ascending=False).groupby(['wficn']).head(time_interval)
    ret_nan=aum.groupby(['wficn'])['ret_bool'].sum()
    ret_nan=ret_nan.reset_index()
    ret_nan.columns=['wficn', 'ret_ratio']
    returns = pd.merge(returns, ret_nan)
    returns=returns[returns['ret_ratio']<=time_interval-least_month]

    return returns

    



