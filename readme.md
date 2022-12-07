# Mutual fund documention
reference : http://www-2.rotman.utoronto.ca/simutin/aw_code.asp
## 1. Fund database (Two main resource): 
### (1) Center for Research in Security Prices (CRSP) Survivor Bias-Free Mutual Fund Database. 
	tables:
		i) fund_style: get Lipper, Strategic Insight, and Wiesenberger classifications and CRSP objective code
		ii) fund_names: get funds' names(to exclude index fund,etc)
		iii) fund_summary: get funds' percentage holding in stock
		iv) monthly_tna_ret_nav: get funds' total net assets, monthly net return, net assets values.
		v) fund_fees: get funds' expense ratio, turnover ratio
		

### (2) Thomson Reuters Mutual Fund Holdings database
	tables:
		i) s12type1: get funds (identifier: FUNDNO) and its holding date
		ii) mflink: link FUNDNO and WFICN, use WFICN as the unique identifier
		iii) s12type3: get funds' holdings 


## 2. CRSP Fund Selection Criteria:
### (1) select domestic equity fund, there are two ways:
	
	i) Lipper, Strategic Insight, and Wiesenberger classifications. 
	lipper_class = ['EIEI', 'G', 'LCCE', 'LCGE', 'LCVE', 'MCCE', 'MCGE', 'MCVE',
                'MLCE', 'MLGE', 'MLVE', 'SCCE', 'SCGE', 'SCVE']
	lipper_obj_cd = ['CA', 'EI', 'G', 'GI', 'MC', 'MR', 'SG']
	si_obj_cd = ['AGG', 'GMC', 'GRI', 'GRO', 'ING', 'SCG']
	wbrger_obj_cd=['G', 'G-I', 'AGG', 'GCI', 'GRI', 'GRO', 'LTG', 'MCG','SCG']

	ii) CRSP objective code:
	crsp_obj_cd[0:3] == 'EDC' or 'EDY' , which means 'Equity', 'Domestic' ,'Cap-based' or 'Style'. 
	Exclude 'EDYH' and 'EDYS', which means hedge fund and short fund.

	note that we can get nearly identical result from these two different selection criteria.

### (2) select mutual funds that invest no less than 80% in common stocks:
	from crsp.fund_summary, per_com: Amount of fund invested in common stocks

### (3) exclude index,sector,etf,target-date fund:
	from crsp.fund_names: index_fund_flag=None,
	screen funds' names, and exclude funds' whose names contains {'index','s&p','idx','dfa','program','international',
	'balanced','bond','sector','etf','exchange traded','exchange-traded','target'} etc.

### (4) further exclusion (optional): 
	i) exclude funds with less than x months observations;
	ii) exclude fund observations before a fund passes the $x million threshold for assets under management (AUM).
	   All subsequent observations, including those that fall under the $x million AUM threshold in the future, are included.
	iii) exclude funds with which over n% of its AUM less than $x million 
	iv) exclude funds with TNA of less than $x million on average
	v) exclude first x months observations or exclude fund return observations reported prior to the year of fund organization to 
	    mitigate incubation bias.


## 3. Get CRSP fund data:
### (1) calculate fund flow:
	we define funds monthly flow equals to 
	(TNA - lag1_TNA*(1+RET))/lag1_TNA

### (2) compute gross return:
	since CRSP report the fund return net of fee, we calculate the gross return (raw return) as funds' return + exp_ratio/12

### (3) aggregate share classes
	Most funds have multiple share classes, which typically differ only in the fee structure 
	and the target clientele and have same holdings.
	So we should aggregate several share classes (identified by crsp_fundno) into the 
	same fund (identified by wficn Wharton Financial Institution Center Number (WFICN)). 

	In particular, we calculate the TNA of each fund as the sum of TNAs of its share classes and calculate fund age 
	as the age of its oldest share class. For all other fund characteristics(total net assets, returns, expense_ratio, turnover_ratio, flow)
	we use the one-month-lagged-TNA weighted average over the share classes. 



## 4. Get Thomson mutual funds holding data
### (1) First get s12type1, which includes fundno, rdate(funds actual holding date), fdate(report date)
### (2) Use mflink to link FUNDNO with WFICN 
### (3) Get s12type3 fund holdings, and merge with crsp monthly stock file, to get a table with wifcn-date-permno-share-price

#


# Variable description:

	'wficn': Wharton Financial Institution Center Number

	'date': date

	'mret': monthly net return (after fee) 

	'rret': gross return (before fee) 

# fund charateristics:
	
	'mtna': monthly total net assets 
	
	'turnover': turnover ratio 

	'exp_ratio': expense ratio 

	'age':fund age in months

	'flow':fund flow 

	'vol': fund return volatility 

	'flow_vol': fund flow volatility

	'mgr_tenure': fund manager's tenure in months

	'capm_rvar': CAPM residual varience

	'capm_alpha': CAPM regression intercept

	'capm_beta': CAPM regression slope

	'fundmom1m': fund 1 month momentum 

	'fundmom12m': fund 12 month momentum (from 2 - 12)

	'fundmom36m': fund 36 month momentum (from 12 - 36)

	'Rsquare': R-squared from rolling-window regression on FF5+MOM factors for previous 36 months

	