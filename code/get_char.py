import pandas as pd
import numpy as np
import datetime as dt
import pyarrow.feather as feather

with open('/home/qianshu/US_factor/chars60_rank_imputed.feather','rb') as f:
    char=feather.read_feather(f)
with open('holdings.feather','rb') as f:
   holdings=feather.read_feather(f)

char['date']=char['date'].astype('str')
char['month']=char['date'].apply(lambda x:x[0:4] + '-' + x[5:7])
holdings['month']=holdings['month'].astype('str')
a=pd.merge(char,holdings,on=['month','permno'])

char_list=[
       'lag_me', 'rank_ill', 'rank_me_ia', 'rank_chtx', 'rank_mom36m',
       'rank_re', 'rank_depr', 'rank_rd_sale', 'rank_roa', 'rank_bm_ia',
       'rank_cfp', 'rank_mom1m', 'rank_baspread', 'rank_rdm', 'rank_bm',
       'rank_sgr', 'rank_mom12m', 'rank_std_dolvol', 'rank_rvar_ff3',
       'rank_herf', 'rank_sp', 'rank_hire', 'rank_pctacc', 'rank_grltnoa',
       'rank_turn', 'rank_abr', 'rank_seas1a', 'rank_adm', 'rank_me',
       'rank_cash', 'rank_chpm', 'rank_cinvest', 'rank_acc', 'rank_gma',
       'rank_beta', 'rank_sue', 'rank_cashdebt', 'rank_ep', 'rank_lev',
       'rank_op', 'rank_alm', 'rank_lgr', 'rank_noa', 'rank_roe',
       'rank_dolvol', 'rank_rsup', 'rank_std_turn', 'rank_maxret',
       'rank_mom6m', 'rank_ni', 'rank_nincr', 'rank_ato', 'rank_rna',
       'rank_agr', 'rank_zerotrade', 'rank_chcsho', 'rank_dy',
       'rank_rvar_capm', 'rank_rvar_mean', 'rank_mom60m', 'rank_pscore',
       'rank_pm', 'log_me']
g = a.groupby(['wficn', 'month']).apply(lambda x: np.average(x['log_me'], weights=x['D']))
fund_char=g.reset_index()
fund_char.columns=['wficn','month','log_me']
for i in char_list:
    g = a.groupby(['wficn', 'month']).apply(lambda x: np.average(x[i], weights=x['D']))
    fund_char[i]=g.reset_index()[0]
    print(i+' finished')

month_list=[]
for i in range(1979,2022):
    tmp1=str(i)+'-'
    for j in range(1,13):
        tmp2=tmp1+str(j).rjust(2,'0')
        month_list.append(tmp2)
month_list=pd.DataFrame(month_list)
month_list.columns=['month']

wficn=pd.DataFrame(fund_char['wficn'].unique())
wficn.columns=['wficn']

month_list_all=[]
for i in fund_char['wficn'].unique():
    tmp=month_list.copy()
    tmp['wficn']=i
    month_list_all.append(tmp)
    
wficn_month=pd.concat(month_list_all)

char_all=pd.merge(fund_char,wficn_month,on=['wficn','month'],how='outer').sort_values(by=['wficn','month'])
char_all=char_all[char_all['month']<='2021-03']   

fund_char_result=char_all.fillna(method='ffill').dropna()
    
    
    
with open('fund_char_result.feather','wb') as f:
    feather.write_feather(fund_char_result,f)

