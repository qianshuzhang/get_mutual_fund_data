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
                    (~funds7['namex'].str.contains('2050')) & (~funds7['namex'].str.contains('2055')) &
                    (~funds7['namex'].str.contains('2060')) & (~funds7['namex'].str.contains('2065')) ]
    return funds7

def get_tna_ret(tna_ret_nav,names,fund_fees,equity_funds):
    returns1 = pd.merge(equity_funds, tna_ret_nav, on='crsp_fundno')
    returns1.rename(columns={'caldt': 'date'}, inplace=True)

    ### get manager tenure
    mgr_dt = names[['crsp_fundno','chgdt','chgenddt','mgr_dt']]
    mgr_dt.rename(columns={'mgr_dt':'mgr_start_time'},inplace=True)
    mgr_dt['mgr_start_time'] = mgr_dt.groupby(['crsp_fundno'])['mgr_start_time'].ffill()

    returns1 = pd.merge(returns1,mgr_dt,on = ['crsp_fundno'],how = 'left')

    returns1 = returns1[(returns1['date'] >= returns1['chgdt_y']) & (returns1['date'] <= returns1['chgenddt_y'])]
    returns1['mgr_start_time'] = returns1['mgr_start_time'].replace(np.nan,'9000-00-00')
    returns1['mgr_tenure'] = (returns1['date'].astype(str).str.slice(0,4).astype(int)-returns1['mgr_start_time'].astype(str).str.slice(0,4).astype(int))*12+(returns1['date'].astype(str).str.slice(5,7).astype(int)-returns1['mgr_start_time'].astype(str).str.slice(5,7).astype(int))
    returns1['mgr_tenure'][returns1['mgr_tenure']<0] = np.nan
    ### end get

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
    returns2['age'] = (returns2['date'] - returns2['first_offer_dt']) / np.timedelta64(1, 'M')
    
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
    returns['flow'] = (returns.groupby(['wficn'])['mtna'].shift(0) - 
                    returns.groupby(['wficn'])['mtna'].shift(1) * (1+returns.groupby(['wficn'])['mret'].shift(0))) / \
                        returns.groupby(['wficn'])['mtna'].shift(1)
    returns.loc[returns['flow'] > 1000, 'flow'] = np.nan
    returns['flow']=returns['flow'].replace(np.inf,np.nan)
    returns['flow']=returns['flow']*100

    # fund vol
    returns['vol'] = returns.groupby(['wficn'])['mret'].rolling(12).std().reset_index()['mret']

    # fund flow vol
    returns['flow_vol'] = returns.groupby(['wficn'])['flow'].rolling(12).std().reset_index()['flow']

    # manager tenure
    returns['mgr_tenure'] = mgr_tenure[0]

    returns.drop_duplicates(subset=['wficn','date'],inplace=True)

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

def get_abr(ff_monthly,returns):
    ff_monthly['date'] = ff_monthly['dateff'] 
    ret_ff = pd.merge(returns, ff_monthly, on='date', how='left')
    ret_ff['exc_ret'] = ret_ff['rret'] - ret_ff['rf']
    ret_ff['Market_adj_ret'] = ret_ff['rret'] - ret_ff['mktrf']

    df_group = ret_ff.groupby('wficn')
    reg_result = []
    for k, g in df_group:
        endog=g['exc_ret']
        exog=sm.add_constant(g[['mktrf','smb','hml','umd']])
        mod = RollingOLS(endog, exog, window=36, min_nobs=30,expanding=True,missing='drop')
        rolling_res = mod.fit()
        reg_result.append(rolling_res.params.shift())

    reg_result_all=pd.concat(reg_result)
    reg_result_all.columns=['alpha','beta_mkt','beta_smb','beta_hml','beta_umd']
    returns_abr=pd.concat([ret_ff,reg_result_all],axis=1)
    returns_abr.dropna(subset=['alpha'],inplace=True)
    returns_abr['abr']=returns_abr['exc_ret']-returns_abr['beta_mkt']*returns_abr['mktrf']-returns_abr['beta_smb']*returns_abr['smb']-returns_abr['beta_hml']*returns_abr['hml']-returns_abr['beta_umd']*returns_abr['umd']
    return returns_abr


def get_holding(returns_abr,fund_char_result):
    returns_wficn=returns_abr['wficn'].unique()
    fund_char_result=fund_char_result[fund_char_result['wficn'].isin(returns_wficn)]
    returns_abr['month']=returns_abr['date'].astype(str).apply(lambda x:x[0:7])
    returns_char=pd.merge(returns_abr,fund_char_result,on=['wficn','month'])

    return returns_char

def standardize(df):
    df_temp = df.groupby(['date'], as_index=False)['wficn'].count()
    df_temp = df_temp.rename(columns={'wficn': 'count'})
    df = pd.merge(df, df_temp, how='left', on='date')
    col_names = df.columns.values.tolist()
    list_to_remove = ['wficn', 'date', 'count','mret', 'mtna', 'rret', 'turnover', 'exp_ratio', 'age', 'flow', 'vol','mktrf', 'smb', 'hml', 'rf', 'year', 'month', 'umd',
       'dateff', 'exc_ret', 'Market_adj_ret', 'alpha', 'beta_mkt', 'beta_smb',
       'beta_hml', 'beta_umd', 'abr', 'log_me', 'lag_me',]
    col_names = list(set(col_names).difference(set(list_to_remove)))
    df = df.fillna(0)
    for col_name in col_names:
        df['%s_rank' % col_name] = df.groupby(['date'])['%s' % col_name].rank()
        df[col_name] = (df['%s_rank' % col_name]-1)/(df['count']-1)*2 - 1
        df = df.drop(['%s_rank' % col_name], axis=1)
    return df

def rank_macro_factor(returns_char,add_factor,macro):
    returns_char = standardize(returns_char)
    returns_char['date']=pd.to_datetime(returns_char['date'])

    add_factor['date']=add_factor['date'].apply(lambda x : datetime.datetime.strptime(x,'%d/%m/%Y'))
    add_factor['month']=add_factor['date'].astype(str).apply(lambda x : x[0:7])
    
    macro['date']=macro['date'].apply(lambda x : datetime.datetime.strptime(x,'%Y/%m/%d'))
    macro['month']=macro['date'].astype(str).apply(lambda x : x[0:7])
    del add_factor['date']
    del macro['date']
    returns_char=returns_char.merge(add_factor,on=['month'],how='left')
    returns_char=returns_char.merge(macro,on=['month'],how='left')
    returns_char.rename(columns={'rank_rvar_mean':'rank_svar'},inplace=True)
    returns_char['weight']=returns_char.groupby(['wficn'])['mtna'].shift()

    returns_char.rename(columns={'rvar_mean':'svar'},inplace = True)    
    return returns_char

def get_summary(returns):
    list = ['turnover', 'age', 'flow', 'exp_ratio', 'mtna', 'vol','abr']

    Num = []
    for i in list:
        Num.append((~returns[i].isna()).sum())

    Mean = []
    for i in list:
        Mean.append(returns[i].mean())
    Mean[1] = returns.groupby(['wficn'])['age'].mean().mean()
    Mean[3] = Mean[3] * 100
    Mean[6] = Mean[6] * 100

    Std = []
    for i in list:
        Std.append(returns[i].std())
    Std[1] = returns.groupby(['wficn'])['age'].mean().std()
    Std[6] = Std[6] * 100

    Med = []
    for i in list:
        Med.append(returns[i].median())
    Med[1] = returns.groupby(['wficn'])['age'].mean().median()
    Med[6] = Med[6] * 100

    p10 = []
    for i in list:
        p10.append(np.nanpercentile(returns[i], 10))
    p10[1] = np.nanpercentile(returns.groupby(['wficn'])['age'].mean(), 10)

    p90 = []
    for i in list:
        p90.append(np.nanpercentile(returns[i], 90))
    p90[1] = np.nanpercentile(returns.groupby(['wficn'])['age'].mean(), 90)

    summary = pd.DataFrame([])
    for i in ['Num', 'Mean', 'Std', 'Med', 'p10', 'p90']:
        summary[i] = eval(i)
    summary.index = list
    return summary

def merge_factor(returns_char,add_factor,macro):

    returns_char = standardize(returns_char)
    returns_char['date']=pd.to_datetime(returns_char['date'])

    
    add_factor['date']=add_factor['date'].apply(lambda x : datetime.datetime.strptime(x,'%d/%m/%Y'))
    add_factor['month']=add_factor['date'].astype(str).apply(lambda x : x[0:7])
    
    macro['date']=macro['date'].apply(lambda x : datetime.datetime.strptime(x,'%Y/%m/%d'))
    macro['month']=macro['date'].astype(str).apply(lambda x : x[0:7])
    del add_factor['date']
    del macro['date']
    returns_char=returns_char.merge(add_factor,on=['month'],how='left')
    returns_char=returns_char.merge(macro,on=['month'],how='left')
    returns_char.rename(columns={'rank_rvar_mean':'rank_svar'},inplace=True)
    returns_char['weight']=returns_char.groupby(['wficn'])['mtna'].shift()

    returns_char.rename(columns={'rvar_mean':'svar'},inplace = True)
    return returns_char
    
def mom(start, end, df):
    """
    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['wficn'])['mret'].shift(i)
        result = result * (1+lag['mom%s' % i])
    result = result - 1
    return result

def calc_rvar_capm(df):
    df_group = df.groupby('wficn')
    reg_result = []
    params = []
    for k, g in df_group:
        endog=g['exc_ret']
        exog=sm.add_constant(g[['MKTRF']])
        mod = RollingOLS(endog, exog, window=36, min_nobs=18,expanding=True,missing='drop')
        model = mod.fit()
        reg_result.append(np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid)))
        params.append(model.params.shift())
    reg_result_all=pd.concat(reg_result)
    reg_result_all = pd.DataFrame(reg_result_all)
    reg_result_all.columns = ['capm_rvar']
    alpha_beta = pd.concat(params)
    alpha_beta.columns=['capm_alpha','capm_beta']
    return reg_result_all,alpha_beta

def mom(start, end, df):
    """
    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['wficn'])['mret'].shift(i)
        result = result * (1+lag['mom%s' % i])
    result = result - 1
    return result

def standard_fund_char(df):
    df_temp = df.groupby(['date'], as_index=False)['wficn'].count()
    df_temp = df_temp.rename(columns={'wficn': 'count'})
    df = pd.merge(df, df_temp, how='left', on='date')
    #col_names = df.columns.values.tolist()
    col_names = list(['exp_ratio','turnover','mtna','age','fundmom12m','fundmom1m','fundmom36m','vol','flow_vol','capm_rvar','capm_alpha','capm_beta','mgr_tenure'])
    #df = df.fillna(0)
    for col_name in col_names:
        df['%s_rank' % col_name] = df.groupby(['date'])['%s' % col_name].rank()
        df['rank_%s' % col_name] = (df['%s_rank' % col_name]-1)/(df['count']-1)*2 - 1
        df = df.drop(['%s_rank' % col_name], axis=1)
    df = df.drop(['count'],axis = 1)
    return df

def get_fund_mom(returns_char):
    returns_char['fundmom1m'] = mom(0,1,returns_char)
    returns_char['fundmom12m'] = mom(1,12,returns_char)
    returns_char['fundmom36m'] = mom(12,36,returns_char)
    returns_char['exc_ret'] = returns_char['mret']-returns_char['RF']
    return returns_char

def calc_rvar_capm(df):
    df_group = df.groupby('wficn')
    reg_result = []
    params = []
    for k, g in df_group:
        endog=g['exc_ret']
        exog=sm.add_constant(g[['MKTRF']])
        mod = RollingOLS(endog, exog, window=36, min_nobs=18,expanding=True,missing='drop')
        model = mod.fit()
        reg_result.append(np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid)))
        params.append(model.params.shift(0))
    reg_result_all=pd.concat(reg_result)
    reg_result_all = pd.DataFrame(reg_result_all)
    reg_result_all.columns = ['capm_rvar']
    alpha_beta = pd.concat(params)
    alpha_beta.columns=['a','capm_beta']
    alpha_beta = alpha_beta['capm_beta']
    return reg_result_all,alpha_beta

def get_fund_capm(returns_char):
    res,alpha_beta= calc_rvar_capm(returns_char)
    returns_char=pd.concat([returns_char,res,alpha_beta],axis=1)
    returns_char['capm_alpha'] = returns_char['exc_ret'] - returns_char['capm_beta']*returns_char['MKTRF']
    return returns_char

def calc_ff4alpha(df):
    df_group = df.groupby('wficn')
    reg_result = []
    for k, g in df_group:
        endog=g['exc_ret']
        exog=sm.add_constant(g[['MKTRF','SMB','HML','UMD']])
        mod = RollingOLS(endog, exog, window=36, min_nobs=18,expanding=True,missing='drop')
        rolling_res = mod.fit()
        reg_result.append(rolling_res.params.shift(0))
    reg_result_all=pd.concat(reg_result)
    reg_result_all.columns=['ff4_alpha','beta_mkt','beta_smb','beta_hml','beta_umd']
    return reg_result_all

def get_fund_ff4(returns_char):
    ff4 = calc_ff4alpha(returns_char)
    returns_char['ff4_alpha'] = returns_char['exc_ret'] - ff4['beta_mkt']*returns_char['MKTRF']-ff4['beta_smb']*returns_char['SMB']-ff4['beta_hml']*returns_char['HML']-ff4['beta_umd']*returns_char['UMD']
    return returns_char

def shift_fund_date_ret(returns_char):
    shift_ls = ['RF', 'MKTRF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD',
       'HMLM', 'QMJ', 'BAB', 'LIQ', 'STR', 'LTR', 'REG', 'IVOL', 'SUE', 'IA',
       'ROE', 'IMD', 'NI', 'BETA', 'PEAD', 'FIN', 'MGMT', 'PERF', 'UMO',
       'PLS_SENT', 'SENT', 'REALU', 'FINU', 'MACROU', 'x_ep', 'x_dy', 'x_ni',
       'x_lev', 'x_svar', 'x_infl_annual', 'x_infl_month', 'x_ill',
       'x_deltatbl', 'x_tms', 'x_dfy','date','mret','rret','exc_ret']
    for char in shift_ls:
        returns_char[char] =returns_char.groupby(['wficn'])[char].shift(-1)
    returns_char = returns_char.dropna(subset=['date']).reset_index(drop=True)
    return returns_char