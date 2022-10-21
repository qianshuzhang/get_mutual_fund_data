import pandas as pd
import numpy as np
import datetime as dt
import sqlite3
import wrds
import pyarrow.feather as feather
from statsmodels.formula.api import ols
from scipy import stats

conn = wrds.Connection()

########################################
# part 1
########################################
'''
thomas investment objective code
            1            International   
            2            Aggressive Growth 
            3            Growth 
            4            Growth & Income 
            5            Municipal Bonds 
            6            Bond & Preferred 
            7            Balanced 
            8            Metals 
            9            Unclassified 
            
   Thomson Financial s12type1 file reports two dates: RDATE and FDATE. As of this writing,

   out of 1133304 observations in that file, only 242739 (21%) had the two dates equal.

   RDATE is the date as of which the positions are held by the fund. FDATE is a 'vintage date'

   and is used as a key for joining some databases.

'''
holdings1 = conn.raw_sql('''select fundno,rdate,fdate,ioc,assets from tfn.s12type1''')
holdings1 = holdings1.sort_values(by=['fundno', 'rdate', 'fdate'])
holdings1.drop_duplicates(['fundno', 'rdate'], inplace=True)

#    MFLINKS file was recently updated to allow more direct linking from Thomson Financial
#    Note that fundno+fdate+rdate is a unique identifier (key) in both mfl.mflink2 and hodlings1
#    only till 2018-12
mflink = conn.raw_sql('''select fundno,fdate,rdate,wficn from mfl.mflink2 ''')

holdings2 = pd.merge(holdings1, mflink, on=['fundno', 'rdate', 'fdate'])
holdings2.sort_values(by=['wficn', 'rdate', 'assets'], inplace=True)
holdings2.drop_duplicates(['wficn', 'rdate'], inplace=True)
holdings2.dropna(subset=['wficn'], inplace=True)
holdings2['fdate'] = holdings2['fdate'].astype('str')

s12type3 = conn.raw_sql('''select * from tfn.s12type3''')
s12type3['fdate'] = s12type3['fdate'].astype('str')

holdings3 = pd.merge(holdings2, s12type3, on=['fundno', 'fdate'])

msename = conn.raw_sql('''
                        select distinct ncusip, permno from crsp.msenames where ncusip is not null
                        ''')
msename.columns = ['cusip', 'permno']

holdings4 = pd.merge(holdings3[['wficn', 'cusip', 'rdate', 'fdate', 'assets', 'shares']], msename, on='cusip')

holdings4.sort_values(by=['wficn', 'rdate', 'fdate'], inplace=True)
holdings4['month'] = holdings4['fdate'].apply(lambda x: x[0:7])

crspmsf = conn.raw_sql(''' select permno,date,cfacshr,prc from crsp.msf ''')
crspmsf['date'] = crspmsf['date'].astype('str')
crspmsf['month'] = crspmsf['date'].apply(lambda x: x[0:7])

holdings5 = pd.merge(holdings4, crspmsf, on=['permno', 'month'])
holdings5.rename(columns={'cfacshr': 'cfacshr_fdate'}, inplace=True)

holdings5['month']=holdings5['rdate'].astype(str).apply(lambda x: x[0:7])
holdings6 = pd.merge(holdings5, crspmsf[['permno', 'date', 'cfacshr', 'month']], on=['permno', 'month'])

qry = '''
select a.*,round(shares*cfacshr_fdate/b.cfacshr,1) as shares_adj
      from holdings5 as a, crspmsf as b
      where a.permno = b.permno and CAST(SUBSTR(fdate, 1, 4) AS integer) = CAST(SUBSTR(date, 1, 4) AS integer) and CAST(SUBSTR(fdate, 6, 7) AS integer) = CAST(SUBSTR(date, 6, 7) AS integer);
'''
holdings6['shares_adj'] = round(holdings6['shares'] * holdings6['cfacshr_fdate'] / holdings6['cfacshr_fdate'], 1)
holdings6['D'] = holdings6['shares_adj'] * abs(holdings6['prc'])

#holdings = holdings6[['wficn', 'month', 'permno', 'D']]
holdings.sort_values(by=['wficn', 'month', 'permno'], inplace=True)

with open('holdings.feather','wb') as f:
    feather.write_feather(holdings,f)