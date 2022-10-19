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




