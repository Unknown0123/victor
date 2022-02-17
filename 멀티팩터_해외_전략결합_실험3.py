import requests
import bs4
import finterstellar as fs
import pandas as pd
import FinanceDataReader as fdr
import time
from pandas import Series
import requests
import pandas as pd
import time
import bs4
import collections
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager, rc
import matplotlib
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from numpy import matrix, ndarray
import pymysql
import datetime
import matplotlib as mpl
import os
from dateutil.relativedelta import relativedelta
import math
from typing import Tuple, Any, Union
from pandas_datareader import data as web
import random
from scipy.optimize import minimize
from collections import Counter
import scipy.optimize as sco
from datetime import datetime, date, timedelta
from openpyxl import Workbook
from pykrx import stock
from dateutil.relativedelta import *
import requests as rq
from tqdm import tqdm
import yahoo_fin.stock_info as si
from bs4 import BeautifulSoup as bs
import csv
import urllib
from yahoofinancials import YahooFinancials
import pickle
import OpenDartReader
import dart_fss as dart

##################### 날짜 만들어주기 #############################
def date_range(start, end):
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    dates = [date.strftime("%Y-%m-%d") for date in pd.date_range(start, periods=(end - start).days + 1)]
    return dates
def date_range2(start2):
    start2 = datetime.strptime(start2, "%Y-%m-%d")
    return start2
def date_range3(end2):
    end2 = datetime.strptime(end2, "%Y-%m-%d")
    return end2
# 여기서 날짜 조정하기 #
start2 = date_range2("2019-03-30")
end2 = date_range3("2021-06-21")
dates = date_range("2019-07-13", "2021-07-18")
print(dates)
print(type(dates))

t = datetime.today()
y = str(t.year)
m = str(t.month).zfill(2)
d = str(t.day).zfill(2)
# mm = t.month.zfill(2)
yesterday = (t - timedelta(1))
yesterday = yesterday.strftime('%Y-%m-%d')

mmmmm = str(t.month - 3).zfill(2)
print(mmmmm)

nday= t - timedelta(1)
totoday = y+m+d
DD = y+'-'+m+'-'+d
# N일전부터 지금까지의 종목! = 즉, 상폐될꺼 상폐되고 상장할꺼 상장된 종목들
Mdays = t - relativedelta(months = 3)
MM = Mdays.strftime('%Y-%m-%d')
MMMMMM = Mdays.strftime('%d-%m-%Y')
Ydays = t - relativedelta(years= 2)
DDDays = t - relativedelta(days = 1)
YY = Ydays.strftime('%Y-%m-%d')
YYY = Ydays.strftime('%m-%d-%Y')
DDD = y+', '+m+', '+d
MMM = MM.replace("-",", ")
MMMMMMM = MMMMMM.replace("-", "/")
DDDays =  DDDays.strftime('%m-%d-%Y')
MMDays = Mdays.strftime('%m-%d-%Y')
startt = y+', '+mmmmm+', '+d
endd = y+', '+m+', '+d
print("날짜확인3", startt, endd)

starttt = MMMMMMM
enddd = d+'/'+m+'/'+y
reitsstart = YYY
reitsend = m+'-'+d+'-'+y
reitsdivstart = MMDays
reitdatastart = DDDays
################################################################################################
def save_csv(데이터프레임, 파일명):  # 데이터프레임을 csv로 저장한다.
    데이터프레임.to_csv(f'{파일명}.csv', encoding='ms949')
    print(f'{파일명}.csv 저장완료')
def read_csv(파일명): # csv를 데이터 프레임으로 읽어온다.
    데이터프레임 = pd.read_csv(f'{파일명}.csv', encoding='ms949' ,thousands = ',') # thousands는 읽어올 파일에서 천의 자릿수 ,를 제거해준다.
    컬럼리스트 = list(데이터프레임.columns)  # 칼럼을 리스트로 가져오기
    데이터프레임 = 데이터프레임.set_index(컬럼리스트[0])  # 첫번째 칼럼을 인덱스로 지정해주기 (csv를 읽어오면 기본 인덱스는 번호이기 때문)
    컬럼리스트 = list(데이터프레임.columns)

    새컬럼리스트 = []

    for 컬럼명 in 컬럼리스트: # 컬럼중에는 스팩과 Unnamed라는 단어가 포함된 결측값이 존재하는것으로 파악되었다.
        if "스팩" in 컬럼명:
            # print(f'{컬럼명} 제외')
            pass
        elif "Unnamed" in 컬럼명:
            # print(f'{컬럼명} 제외')
            pass
        else:
            새컬럼리스트.append(컬럼명)
    데이터프레임 = 데이터프레임[새컬럼리스트] # 결측값 제외된 컬럼명만 사용하여 데이터프레임 재구성
    return 데이터프레임
from datetime import datetime
def date_range(start, end):
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    dates = [date.strftime("%Y-%m-%d") for date in pd.date_range(start, periods=(end - start).days + 1)]
    return dates
def date_range2(start2):
    start2 = datetime.strptime(start2, "%Y-%m-%d")
    return start2
def date_range3(end2):
    end2 = datetime.strptime(end2, "%Y-%m-%d")
    return end2

####################################################################################
######### @@@@@@@@@@@@@@@@  3. 여기서부터 백테스트 코딩 ######### @@@@@@@@@@@@@@@@
####################################################################################
def get_returns_df(df, N=1, log=False):
    if log:
        return np.log(df / df.shift(N)).iloc[N - 1:].fillna(0)
    else:
        return df.pct_change(N, fill_method=None).iloc[N - 1:].fillna(0)

def get_cum_returns_df(return_df, log=False):
    if log:
        return np.exp(return_df.cumsum())
    else:
        return (1 + return_df).cumprod()  # same with (return_df.cumsum() + 1)

def get_CAGR_series(cum_rtn_df, num_day_in_year=252):
    # cagr_series = cum_rtn_df.iloc[-1] ** (num_day_in_year / (len(cum_rtn_df))) - 1
    year_series = len(cum_rtn_df) / num_day_in_year
    year_change = 1 / year_series
    # cagr2 = cum_rtn_df.iloc[-1]**year_change - 1
    cagrcheck1 = cum_rtn_df.iloc[-1, [-1]]
    cagrcheck2 = cum_rtn_df.iloc[0, [-1]]
    cagrcheck3 = cagrcheck1 / cagrcheck2 - 1
    cagrcheck4 = (1 + cagrcheck3) ** year_change - 1
    cagr_series = cagrcheck4
    print("#### 기간 :", year_series, "년")
    print("#### 계산을 위한 기간 날짜변환 1 / 날짜 :", year_change)
    print("#### 'P0' 1일차 INDEX 값 :", cagrcheck2)
    print("#### 'P1' 막일차 INDEX 값 :", cagrcheck1)
    print("#### 기간수익률 :", cagrcheck3)
    print("#### 연 환산 수익률 :", cagrcheck4)
    print("#### CAGR 구하기 :", cagr_series)
    return cagr_series

def get_port_CAGR_series(cum_rtn_df, num_day_in_year=252):
    # cagr_series = cum_rtn_df.iloc[-1] ** (num_day_in_year / (len(cum_rtn_df))) - 1
    year_series = len(cum_rtn_df) / num_day_in_year
    year_change = 1 / year_series
    # cagr2 = cum_rtn_df.iloc[-1]**year_change - 1
    cagrcheck1 = cum_rtn_df.iloc[-1]
    cagrcheck2 = cum_rtn_df.iloc[0]
    cagrcheck3 = cagrcheck1 / cagrcheck2 - 1
    cagrcheck4 = (1 + cagrcheck3) ** year_change - 1
    cagr_series = cagrcheck4
    print("#### 기간 :", year_series, "년")
    print("#### 계산을 위한 기간 날짜변환 1 / 날짜 :", year_change)
    print("#### 'P0' 1일차 INDEX 값 :", cagrcheck2)
    print("#### 'P1' 막일차 INDEX 값 :", cagrcheck1)
    print("#### 기간수익률 :", cagrcheck3 * 100, "%")
    print("#### 연 환산 수익률 :", cagrcheck4 * 100, "%")
    print("#### CAGR 구하기 :", cagr_series * 100, '%"')
    return cagr_series

def get_sharpe_ratio(log_rtn_df, yearly_rfr=0):
    excess_rtns = log_rtn_df.mean() * 252 - yearly_rfr
    return excess_rtns / (log_rtn_df.std() * np.sqrt(252))

def get_std(log_rtn_df):
    std = log_rtn_df.std() * np.sqrt(252)
    return std

def get_drawdown_infos(cum_returns_df):
    # 1. Drawdown
    cummax_df = cum_returns_df.cummax()
    dd_df = cum_returns_df / cummax_df - 1

    # 2. Maximum drawdown
    mdd_series = dd_df.min()

    # 3. longest_dd_period
    dd_duration_info_list = list()
    max_point_df = dd_df[dd_df == 0]
    for col in max_point_df:
        _df = max_point_df[col]
        _df.loc[dd_df[col].last_valid_index()] = 0
        _df = _df.dropna()

        periods = _df.index[1:] - _df.index[:-1]

        days = periods.days
        max_idx = days.argmax()

        longest_dd_period = days.max()
        dd_mean = int(np.mean(days))
        dd_std = int(np.std(days))

        dd_duration_info_list.append(
            [
                dd_mean,
                dd_std,
                longest_dd_period,
                "{} ~ {}".format(_df.index[:-1][max_idx].date(), _df.index[1:][max_idx].date())
            ]
        )

    dd_duration_info_df = pd.DataFrame(
        dd_duration_info_list,
        index=dd_df.columns,
        columns=['drawdown mean', 'drawdown std', 'longest days', 'longest period']
    )
    drawdowndf = pd.concat([dd_df, mdd_series, dd_duration_info_df], axis=0)
    save_csv(drawdowndf, y + m + d + "모멘텀_드로우다운")
    return dd_df, mdd_series, dd_duration_info_df

def get_rebal_dates(price_df, period="month"):
    _price_df = price_df.reset_index()
    if period == "month":
        groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.month]
    elif period == "quarter":
        groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.quarter]
    elif period == "halfyear":
        groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.month // 7]
    elif period == "year":
        groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.year]
    rebal_dates = pd.to_datetime(_price_df.groupby(groupby)['Date'].last().values)
    return rebal_dates

from functools import reduce
def calculate_portvals(price_df, weight_df):
    cum_rtn_up_until_now = 100
    individual_port_val_df_list = []
    assets_add = 10  ### 10만원씩 매월 투입할 금액.
    prev_end_day = weight_df.index[0]
    print("전월 마지막 날짜", prev_end_day)
    for end_day in weight_df.index[1:]:
        print("현재 월 마지막 날짜", end_day)
        sub_price_df = price_df.loc[prev_end_day:end_day]
        print("'전월' ~ '현재 월' 까지의 평균일수_그리고 가격", sub_price_df)
        sub_asset_flow_df = sub_price_df / sub_price_df.iloc[0]
        print("sub_asset_flow_df=개별종목들의 일별 수익률", sub_asset_flow_df)
        weight_series = weight_df.loc[prev_end_day]
        indi_port_cum_rtn_series = (sub_asset_flow_df * weight_series) * cum_rtn_up_until_now
        print("indi_port_cum_rtn_series", indi_port_cum_rtn_series)
        individual_port_val_df_list.append(indi_port_cum_rtn_series)
        total_port_cum_rtn_series = indi_port_cum_rtn_series.sum(axis=1)

        ############## 아래 매월 10만원 납입하는거 추가할지 정하기 ################

        cum_rtn_up_until_now = total_port_cum_rtn_series.iloc[-1] #+ assets_add  # 매월 마지막일에 10만원 투입
        print("####매월마지막일에 10만원 투입####", cum_rtn_up_until_now)

        prev_end_day = end_day

    individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
    individual_port_val_df['Portfolio'] = individual_port_val_df.sum(axis=1)
    save_csv(individual_port_val_df, y + m + d + "리스크패러티테스팅")
    return individual_port_val_df


####################################################################################
######### @@@@@@@@@@@@@@@@  4. 여기서부터 멀티팩터 모델링 ######### @@@@@@@@@@@@@@@@
####################################################################################

## OTP
OTP = '16445493521001111081'

# 분기 날짜 변경
Quarter1 = 'Q1'
Quarter2 = 'Q2'
Quarter3 = 'Q3'
Quarter4 = 'Q4'

# 시작일과 끝일 + 분기날짜 변경 전
startday = '2020'
endday = '2021'

# 시작일과 끝일 Put 값
Msday = startday+Quarter3
Meday = endday+Quarter4

# 여기서 날짜 바뀜
terms = fs.set_terms(trade_start=Msday, trade_end=Meday)

data = {}
for t in terms:
    data[t] = fs.fn_consolidated(otp=OTP, term=t)
for t in terms:
    # Set previous terms
    prev_t = fs.quarters_before(terms, t, 4)
    # Company size
    data[t]['Market Cap'] = data[t]['Price_M3'] * data[t]['Shares']
    # Value
    data[t]['PER'] = data[t]['Price_M3'] / data[t]['EPS']
    data[t]['PBR'] = data[t]['Price_M3'] / (data[t]['Shareholders Equity']/data[t]['Shares'])
    data[t]['PSR'] = data[t]['Price_M3'] / (data[t]['Revenue'] / data[t]['Shares'])
    data[t]['PCR'] = data[t]['Price_M3'] / ( ( data[t]['Net Income'] + data[t]['Depreciation'] ) / data[t]['Shares'] )
    # Profitability
    data[t]['Avg Assets'] = ( data[t]['Total Assets'] + data[prev_t]['Total Assets'] ) / 2
    data[t]['Avg Equity'] = ( data[t]['Shareholders Equity'] + data[prev_t]['Shareholders Equity'] ) / 2
    data[t]['ROA'] = data[t]['Net Income'] / data[t]['Avg Assets']
    data[t].loc[(data[t]['Net Income']<0) | (data[t]['Avg Assets']<0) | (data[t]['Total Assets']<0), 'ROA'] = float('nan')
    data[t]['ROE'] = data[t]['Net Income'] / data[t]['Avg Equity']
    data[t].loc[(data[t]['Net Income']<0) | (data[t]['Avg Equity']<0) | (data[t]['Shareholders Equity']<0), 'ROE'] = float('nan')
    data[t]['GP/A'] = data[t]['Gross Profit'] / data[t]['Avg Assets']
    data[t].loc[(data[t]['Gross Profit']<0) | (data[t]['Avg Assets']<0) | (data[t]['Total Assets']<0), 'GP/A'] = float('nan')
    data[t]['GP/E'] = data[t]['Gross Profit'] / data[t]['Avg Equity']
    data[t].loc[(data[t]['Gross Profit']<0) | (data[t]['Avg Equity']<0) | (data[t]['Shareholders Equity']<0), 'GP/E'] = float('nan')
    # Stability
    data[t]['Liability/Equity'] = data[t]['Total Liabilities'] / data[t]['Shareholders Equity']
    data[t].loc[(data[t]['Shareholders Equity']<0), 'Liability/Equity'] = float('nan')
    data[t]['Debt/Equity'] = (data[t]['Long Term Debt'] + data[t]['Current Debt']) / data[t]['Shareholders Equity']
    data[t].loc[(data[t]['Shareholders Equity']<0), 'Debt/Equity'] = float('nan')
    data[t]['Current Ratio'] = data[t]['Current Assets'] / data[t]['Current Liabilities']
    data[t]['Share Increase'] =  data[t]['Shares'] / data[prev_t]['Shares']
    # Efficiency
    data[t]['Gross Margin'] = data[t]['Gross Profit'] / data[t]['Revenue']
    data[t]['Asset Turnover'] = data[t]['Revenue'] / data[t]['Avg Assets']
    data[t]['Equity Turnover'] = data[t]['Revenue'] / data[t]['Avg Equity']
    # Momentum
    data[t]['Price Growth'] =  data[t]['Price_M3'] / data[prev_t]['Price_M3']

s1 = {}
s2 = {}
s3 = {}
s4 = {}
s5 = {}
s6 = {}
s7 = {}
s8 = {}
s9 = {}
s10 = {}
s11 = {}
s12 = {}
s13 = {}

s = {}
ssssss = {}
signal = {}

# *가치*[4대장] + *퀄*[ROE + GP/A + 부채R + 유동R] + *사*소형주 + *증*증자X + *모*모멘텀
# @@@ 해석 =
# 1. 컴바인스코어링은 df= df1[t] + df2[t] 로 해결하고,
# 2. 나머지 fn_filter 는 그대로 해결하면 됨.
# 3. 그렇게 해서 combine_signal 안에 집어넣으면 됨.
# 4. 이렇게 하는 이유는, comebine_scoring 이 2개이상 존재 할 수 없음. 하나의 포문에서
# 5. 즉, combine_signal() 이 ()안에 combine_scoring은 하나여야 하기 때문임.
# 6. 따라서, combine_scoring은  1. 에서 설명한 바와 같이, df = df1[t] + df2[t] 으로 해결하면 됨.

# 이거 활용해볼까 생각 중.
def combine_signal(*signals, how='and', n=None):
    '''
    :param signals: Data set storing trading signal
    :param how: The joining method. Select 'and' for intersection, 'or' for union
    :param n: The size of result data
    :return: Combination of signals
    '''
    how_dict = {'and':'inner', 'or':'outer'}
    signal = signals[0]
    for s in signals[1:]:
        signal = signal.join(s, how=how_dict[how])
    return signal[:n]

for t in terms:
    s1[t] = fs.fn_score(data[t], by='PER', method='relative', floor=1, asc=True)
    s2[t] = fs.fn_score(data[t], by='PBR', method='relative', floor=.1, asc=True)
    s3[t] = fs.fn_score(data[t], by='PSR', method='relative', floor=.1, asc=True)
    s4[t] = fs.fn_score(data[t], by='PCR', method='relative', floor=.1, asc=True)
    s5[t] = fs.fn_score(data[t], by='ROE', method='relative', floor=0, asc=False)
    s6[t] = fs.fn_score(data[t], by='GP/E', method='relative', floor=0, asc=False)
    s7[t] = fs.fn_score(data[t], by='Debt/Equity', method='relative', floor=0, asc=True)
    s8[t] = fs.fn_score(data[t], by='Current Ratio', method='relative', floor=0, asc=False)

    # 사이즈 팩터 @ 1개
    s9[t] = fs.fn_filter(data[t], by='Market Cap', floor=0, n=1, asc=True)
    print("사이즈팩터 스크리닝 5개 비교",type(s9), s9)

    # 증자X 팩터 @ 1개
    s10[t] = fs.fn_filter(data[t], by='Share Increase', floor=.9, cap=1, n=1, asc=True)
    print("증자팩터 스크리닝 5개 비교",s10)

    # 모멘텀 팩터 @ 1개
    s11[t] = fs.fn_filter(data[t], by='Price Growth', floor=1, n=1, asc= True)
    print("모멘텀팩터 스크리닝 5개 비교",s11)
    s12[t] = fs.combine_score(s1[t], s2[t], s3[t], s4[t], n=1) # 가치 팩터 스코어
    print("가치 팩터 스크리닝 5개",s12) # 이렇게 가능함.
    s13[t] = fs.combine_score(s5[t], s6[t], s7[t], s8[t], n=1) # 퀄리티 팩터 스코어
    print("퀄리티 팩터 스크리닝 5개",s13) # 이렇게 가능함.

    ## 가치 팩터 + 퀄리티 팩터 @ 2개
    s[t] = s12[t] + s13[t]
    print("@@ 가치 + 퀄 = 합쳐지나? 10개", s)

    # ***** 여기다 넣어야겠지 ? *******

    # @@@@@ 가치 + 퀄 + 사이즈 + 증자X + 모멘텀 # 팩터 결합 @@@@@ =  총 5개 종목
    ssssss[t] = fs.combine_signal(s[t], s9[t],s10[t],s11[t], how='or', n=5)
    signal[t] = list(ssssss[t].index)
    print("각분기보유종목 개수",ssssss[t],signal[t])

    # 여기서 중복되는 종목으로 4개 종목이 나올 수 있으니 .join or .merge or concat으로 중복안되게 할 것.
    # 위에 def combine_signal(*,) 이걸 통해서!!


stocks = set()
for k, v in signal.items():
    for sto in v:
        stocks.add(sto)
print("해결책을위한과정1",stocks)
trades = pd.DataFrame()
for k, v in signal.items():
    for sto in stocks:
        trades.loc[k, sto] = 'l' if sto in list(v) else 'z'
print("해결책을위한과정2",trades)
prev = trades.shift(periods=1, axis=0)
prev.fillna('z', inplace=True)
print("해결책을위한과정3",prev)

TDF = pd.DataFrame(trades)
print( "해결책을위한과정2-2",type(TDF), TDF)
save_csv(TDF, y+m+d+"팩포제발되어라1")

position = prev + trades
position = pd.DataFrame(position)
print("해결책을위한과정4", position)
save_csv(position, y+m+d+"팩포제발되어라2")
## @@@ 설명 : position 에서 zz, zl, lz, ll 이 나오는데 이걸로 진입시점, 청산시점 알 수 있음.

inclusive = position.copy()
inclusive.replace('zz', 0, inplace=True)
inclusive.replace('zl', 0.2, inplace=True)
inclusive.replace('ll', 0.2, inplace=True)
inclusive.replace('lz', 0, inplace=True)

##  @@@ 해결해야 하는 부분 : combine_signal() 에서 각 팩터의 중복되는 종목에 대해 중첩이 안되어버림.

print(inclusive)
save_csv(inclusive, y+m+d+"팩포제발되어라3")

# Q4를 시작으로 하면 Q3 때 좋았던 종목을 Q4 때 삼 => 즉, 9월 3분기 발표때 좋았던 종목을 12월에 매수하는거임. = 12월 31일.
# [시작일]
# Q1 설정 => 전 Q4에 좋은 종목 Q1인 3월에 매수 => Q2에 매도
# Q2 설정 => 전 Q1에 좋은 종목 Q2인 6월에 매수 => Q3인 9월에 매도
# Q3 설정 => 전 Q2에 좋은 종목 Q3인 9월에 매수 => Q4인 12월에 매도
# Q4 설정 => 전 Q3에 좋은 종목 Q4인 12월에 매수 => 다음 해 Q1인 3월에 매도

df = fs.backtest(signal=signal, data=data, m=3, cost=.001)

print(df)
print(s9,s10,s11,s12,s13)
print("시그널 생성 전 종목합",type(ssssss),ssssss)
print("시그널 생성",type(signal),signal)
# dfdict1 = pd.DataFrame(ssssss)
# dfdict2 = pd.DataFrame(signal)

save_csv(df, y+m+d+"팩포_4대장")
# save_csv(dfdict1, y+m+d+"팩포_딕1_데프로")
# save_csv(dfdict2, y+m+d+"팩포_딕2_데프로")

df2 = df.drop(['term_rtn','acc_rtn','dd', 'mdd', 'term_exs_rtn'], axis=1)
print("필요없는 컬럼 잘 삭제되었는지 확인",df2)
save_csv(df2, y+m+d+"분기별_팩터_종목_저장_데이터프레임")

# 데이터프레임의 컬럼명을 리스트로
columnslist = df2.columns.tolist()
print(columnslist)

# 날짜 Q -> 월말 날짜로 바꾸기
Q1 = Quarter1.replace("Q1", "-03-20")
Q2 = Quarter2.replace("Q2", "-06-20")
Q3 = Quarter3.replace("Q3", "-09-20")
Q4 = Quarter4.replace("Q4", "-12-20")
print(Q1, Q2, Q3, Q4)
# Quarter1 = 'Q1'
# Quarter2 = 'Q2'
# Quarter3 = 'Q3'
# Quarter4 = 'Q4'
# startday = '2020'
# endday = '2021'
# Msday = startday+Quarter3
# Meday = endday+Quarter4
print("변경 전 값", type(startday), startday)
startday2 = startday + Q3
print("변경 후 값", type(startday2),startday2)

datetime_format = '%Y-%m-%d'
startday3 = datetime.strptime(startday2, datetime_format)
print("변경 후 값2", type(startday3),startday3)


# 각 종목의 주가데이터 다운받기
item_list = []
for item_code in columnslist:
    print('오늘 날짜는 :.', startday3, "종목은 :", item_code)
    close = fdr.DataReader(item_code, startday3)['Close']
    print(close)
    item_list.append(close)

df1 = pd.concat(item_list, axis=1)
df1.columns = columnslist
price_df = df1
price_df = price_df.fillna(method='pad')
price_df = price_df.dropna()
print("주식_가격데이터", price_df)
rebal_dates = get_rebal_dates(price_df, period = "quarter")
print(rebal_dates)

















