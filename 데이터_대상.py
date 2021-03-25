#!/usr/bin/env python
# coding: utf-8

# # Deasang Time serise 
# 
# 대상 데이터를 활용해 아래와 같은 결과 값을 구현하였습니다.
# 
# 
# **목적**
# 1. 각 아이템의 과거 데이터로 2020년도 QTY 예측
# 2. 신규 아이템에 대한 증감 예측
# 3. 월평균으로 2020년도의 QTY 예측
# 
# **시계열 모델정보**
# - _Autom_Arima_
# - _Autom_Arima_ 처음 실행시   **!pip install pmdarima** 코드 실행 필요
# 
# **Input 정보**
# - 데이터 셋 파일명: DAESANG_DATA_prepared.csv
# - 설정 옵션 파일명: input_시계열모델_설정옵션.csv
# - 분석 조건 값 파일명: input_시계열모델_조건설정값.csv
# - 데이터 유형 파일명: input_시계열모델_데이터유형.csv
# 
# **Output 정보**
# - 예측 결과
# - 예측 모델 정보
# - 예측 모델 값
# - 변수 중요도
# 
# 
# 
# ※ 자세한 Input정보는 같은 폴더에 있는 Readme를 참고 해주세요.

# ## 1. 개요
# 
# ### 시나리오 1. 시계열 분석
# - 예측 모델의 입출력 값
#   - Input(x)는 QTY의 과거 값
#   - Output(y)는 QTY의 미래 예측 값 
# 
# 
# ### 시나리오 2. 유사제품 예측 (차분 & 랜덤포레스트 분석)
# - 예측 모델의 입출력 값
#   - Input(x)은 신규 아이템에 대한 속성(이전에 **_없던_** 속성 값)
#   - Output(y)는 신규 아이템에 대한 QTY값(예측대상)의 증감
#   
#   시나리오 2.1 월간예측 (차분 & Auto_ML 사용)
#    - 계절성 확인 목적
#  
# ### 시나리오 3. 월평균 예측 (선형분석)
# - 예측 모델의 입출력 값
#   - Input(x)은 신규 아이템에 대한 속성(이전에 **_있던_** 속성 값)
#   - Output(y)는 기존 아이템에 대한 평균 QTY 예측값 

# In[ ]:


# !pip install pmdarima # have to install for Library # 처음 실행시 필요


# In[ ]:


# 필요 라이브러리
import csv
import pandas as pd
import numpy as np
import ast

from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose 
from pandas import DataFrame
from pandas import read_csv
from pandas import datetime

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA

import pmdarima
from pmdarima import auto_arima

from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


# 구현 함수
def get_data_type(data_type_file):
    data_type_df = pd.read_csv(data_type_file,encoding='cp949')
    
    x_val = data_type_df.loc[data_type_df['Role']=='x', 'col_name'].tolist() # x변수 다중리스트형태
    y_val = data_type_df.loc[data_type_df['Role']=='y', 'col_name'].tolist()[0] # y변수는 단일
    predic_period = data_type_df.loc[data_type_df['예측주기']=='P', 'col_name'].tolist()[0] # y변수는 단일
    
    dummy_list = data_type_df.loc[data_type_df['col_info']=='STR', 'col_name'].tolist()# y변수는 단일
    
    item_col = data_type_df.loc[data_type_df['col_info']=='STR_KEY', 'col_name'].tolist()# y변수는 단일
    
    return x_val, y_val, dummy_list,item_col[0],predic_period

def get_variables(setting_file_name, condition_file_name):
    """
    Input: setting_file_name(모델설정값파일이름), condition_file_name(조건설정값파일이름)
    
    Input 설정 파일에서 지정된 컬럼 명으로 설정 값 및 조건 값들을 불러옴
    
    return: Date지정 컬럼명, 타겟 컬럼명, 주기 값, 조건 값(리스트 )
    """
    setting_df = pd.read_csv(setting_file_name,encoding='cp949')
    date_col_name = setting_df.at[0, 'Date_col'] 
    target_name = setting_df.at[0, 'Target'] 
    p = setting_df.at[0, 'Period'] 
    
    condition_df = pd.read_csv(condition_file_name,encoding='cp949')
    keys = condition_df['condition_col_name'].tolist()
    values = condition_df['condition'].tolist()
    
    condition_list = get_conditions(keys,values)
    
    return date_col_name, target_name, p, condition_list

def change_col_to_int (df, col_name_change_to_int): 
    # 데이터 유형 정보가 담긴 row1이 string이라 계산값 (타겟) int전환 필요
    df[col_name_change_to_int] = df[col_name_change_to_int].astype('int32')

def get_conditions(keys,values):
    """
    Input: keys(조건 컬럼명), values(조건)
    
    return: Dictionary 형태로 조건 컬럼명과 조건 
    """
    values = [tryeval(x) for x in values]
    values2 = [[_] for _ in values]
    conditions = dict(zip(keys, values2))
    return conditions

def tryeval(val): # change str into int in a list 
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val

def get_dataset(data_file_name,target_name):
    """
    Input: data_file_name(데이터셋 파일이름),target_name(설정값에서 불러온 타겟명)
    
    파일읽고 df에 저장, 타겟값은 int로 지정, Train dataset, TestDataset 분리
    """
    df = pd.read_csv(data_file_name, encoding='cp949') # encoding은 한국어 load 
    df = df.drop([0]) # 데이터 유형은 실제 데이터 학습에 필요하지 않음으로 드롭
    df.fillna(0, inplace=True) # RD에 있는 NAN 데이터를 0으로 바꿈 => 데이터 유형을 정수로 표현하기 위해 (데이터 셋 분리과정)
    change_col_to_int(df,target_name)
    
    # TD, RD 분리 (학습 데이터 및 실험 데이터 생성)
    train = df[df['DATA_TYPE']== 'TD']
    test = df[df['DATA_TYPE']== 'RD']
    
    return df, train, test

def get_model_var(df,Model_ver):  #모델버전 생성 
    df = df.reset_index(drop=False)
    Model_ver_list =  [Model_ver] * len(df)
    Model_ver_list = pd.DataFrame(Model_ver_list, columns =['모델 버전'])
    
    updated_df = pd.concat([Model_ver_list,df] ,axis=1)
    return updated_df

def c_columns(df):
    cols = []
    cols.append('모델')
    for i in range(len(df.columns)-1):
        col = 'c'+str(i)
        cols.append(col)
    return cols

def export_results(df_model): #시계열 식 결과 export into excel
    model_results = df_model.summary()

    model_info = model_results.tables[0].as_html()
    model_info = pd.read_html(model_info, header=0, index_col=0)[0]
    
    model_result = model_results.tables[1].as_html()
    model_result = pd.read_html(model_result, header=0, index_col=0)[0] # Excel 내보내기
    
    Model_ver = model_info['y'][0] + "_" + model_info['y'][1] + "_"+model_info['y'][2]
   
    model_info_df = get_model_var(model_info,Model_ver)
    model_info_df.columns = pd.MultiIndex.from_tuples(zip(c_columns(model_info_df), model_info_df.columns)) 
    
    model_result_df = get_model_var(model_result,Model_ver)
    model_result_df.columns = pd.MultiIndex.from_tuples(zip(c_columns(model_result_df), model_result_df.columns)) 
    
    return model_info_df,model_result_df,Model_ver

def export_results_reg(df_model,y_pred_year,y_test,f1,f2,f3): #시계열 식 결과 export into excel
    model_results = df_model.summary()

    model_info = model_results.tables[0].as_html()
    model_info = pd.read_html(model_info, header=0, index_col=0)[0]
    
    model_result = model_results.tables[1].as_html()
    model_result = pd.read_html(model_result, header=0, index_col=0)[0] # Excel 내보내기
    
    Model_ver = model_info[y_val][0] + "_" + model_info[y_val][1] + "_"+model_info[y_val][2]
   
    model_info_df = get_model_var(model_info,Model_ver)
    model_info_df.columns = pd.MultiIndex.from_tuples(zip(c_columns(model_info_df), model_info_df.columns)) 
    
    model_result_df = get_model_var(model_result,Model_ver)
    model_result_df.columns = pd.MultiIndex.from_tuples(zip(c_columns(model_result_df), model_result_df.columns)) 
    
    prediction_df = pd.DataFrame(data=y_pred_year)
    prediction_df.columns = ['예측값']
    
    results_pred_df = get_model_var(prediction_df,Model_ver)
    results_pred_df_= results_pred_df.set_index('PART').join(y_test)
    
    outputfile(results_pred_df_,f1) # 예측값 엑셀로 내보내기
    outputfile(model_info_df,f2) # 모델 정보 엑셀로 내보내기
    outputfile(model_result_df,f3) # 모델 식 엑셀로 내보내기
    
    return model_info_df,model_result_df,Model_ver,results_pred_df_

def outputfile(result1,output_file_name): # Export result into excel 
    result1.to_csv(output_file_name,encoding='utf-8-sig')
    print("\n▼ 폴더에서",output_file_name,"파일을 확인하세요")
    
def export_to_excel(df_model,pred_df,f1,f2,f3):
    model_info , model_result, Model_ver= export_results(df_model)
    pred_df_ = get_model_var(pred_df,Model_ver)
    outputfile(pred_df_,f1) # 예측값 엑셀로 내보내기
    outputfile(model_info,f2) # 모델 정보 엑셀로 내보내기
    outputfile(model_result,f3) # 모델 식 엑셀로 내보내기
    
def index_match_prediction(y_pred_df):
    indexs = pd.DataFrame(y_pred_df.index.str.split('-',1).tolist(),
                                 columns = ['RD_index','TD_index'])
    y_pred_df_ = y_pred_df.reset_index(drop=True)
    result = pd.concat([indexs,y_pred_df_], axis=1)
    return result    
    
def filter_df(df, filter_values):
    """
    Input: df(데이터셋), filter_values(Dic 형태의 조건 리스트: condition_list)
    
    Filter df by matching targets for multiple columns.
    지정된 조건에 df 필터
    
    return 조건에 맞는 데이터셋 
    """
    if filter_values is None or not filter_values:
        return df
    return df[
        np.logical_and.reduce([
            df[column].isin(target_values) 
            for column, target_values in filter_values.items()
        ])
    ]

def find_similar_Item_ID(new_grouped1,item_col):
    index_list = []
    for i in range(len(new_grouped1)):
        index = new_grouped1.index[i].split('-')[1]
        index_list.append(int(index))
        
    item_list_by_index = []
    for i in index_list:
        item = df[df.index == i][item_col].values[0]
        item_list_by_index.append(item)
        
    index_list_for_Predict = []
    for i in range(len(new_grouped1)):
        index = new_grouped1.index[i].split('-')[0]
        index_list_for_Predict.append(int(index))
        
    item_list_for_Predict = []
    for i in index_list_for_Predict:
        item = df[df.index == i][item_col].values[0]
        item_list_for_Predict.append(item)
    
    return index_list,item_list_by_index,index_list_for_Predict,item_list_for_Predict


def runAutoArima_Prediction(train,test, target_name, date_col_name, condition, p):
    """
    Input: train(학습용 데이터 셋), test(예측용 데이터 셋),
    target_name (예측변수 - 타겟), date_col_name (Time 컬럼
    ) ,
    condition: 조건 리스트, p(예측 주기)
    
    Output = 시계열 모델식, 예측 데이터 셋
    """
    df_train = filter_df(train, condition)
    df_test = filter_df(test, condition)
    
    Q_train = df_train[[target_name,date_col_name]]
    Q_test = df_test[[target_name,date_col_name]]
    
    Q_train[[target_name] ] = Q_train[[target_name]].astype('int32')
    Q_test[[target_name] ] = Q_test[[target_name]].astype('int32')
    
    Q_train = Q_train.set_index(date_col_name)
    Q_test = Q_test.set_index(date_col_name)
    
    n_p = len(df_test) # 데이터에 RD가 없는 경우 확인
    
    if n_p < 0: # If there are RD
        print("예측기간이 정해지지 않았습니다.")
        
    else:
        stepwise_model_series = auto_arima(Q_train,m=p,seasonal = True)
        stepwise_model_series.fit(Q_train)
        future_forecast_ = stepwise_model_series.predict(n_periods= n_p)
        future_forecast_ = pd.DataFrame(future_forecast_,index = Q_test.index,columns=[target_name])
        if len(condition_list) <= 1:
            pd.concat([Q_train,future_forecast_],axis=1).plot(figsize=(20,5))
            prediected_df = pd.concat([Q_train,future_forecast_]) 
        
        df_test[target_name] = future_forecast_[[target_name]].values
        
    return stepwise_model_series, df_test

# 각 아이템 별 시계열 분석
def Prediction_per_item(train,test,item_col,item, target_name, date_col_name, p):
    """
    Input: train(학습용 데이터 셋), test(예측용 데이터 셋), itme_col (아이템 컬럼 리스트), item(아이템 명)
    target_name(Y 데이터 컬럼 명), date_col_name(시계열 컬럼 명), p(예측 주기)
    """
    df_train = train[train[item_col]== item]
    df_test = test[test[item_col]== item]
    
    Q_train = df_train[[target_name,date_col_name]]
    Q_test = df_test[[target_name,date_col_name]]
    
    Q_train[[target_name] ] = Q_train[[target_name]].astype('int32')
    Q_test[[target_name] ] = Q_test[[target_name]].astype('int32')
    
    Q_train = Q_train.set_index(date_col_name)
    Q_test = Q_test.set_index(date_col_name)
    
    n_p = len(df_test) # 데이터에 RD가 없는 경우 확인
    
    if n_p < 0: # If there are RD
        print("예측기간이 정해지지 않았습니다.")
        
    else:
        stepwise_model_series = auto_arima(Q_train,m=p, max_D =1 ,seasonal = True)
        stepwise_model_series.fit(Q_train)
        future_forecast_ = stepwise_model_series.predict(n_periods= n_p)
        future_forecast_ = pd.DataFrame(future_forecast_,index = Q_test.index,columns=[target_name])
        pd.concat([Q_train,future_forecast_],axis=1).plot(figsize=(20,5))
        prediected_df = pd.concat([Q_train,future_forecast_]) 
        print('아이템 컬럼: ',item_col,", 아이템 명: ",item)
        print(prediected_df)
        
    return prediected_df,stepwise_model_series

# new data (X_Predict) 와 X_Train의 차분 계산
def subtract (new_test_df,train_df):
    dff = pd.DataFrame()
    for i in range(len(new_test_df.columns)):
        for j in range(len(train_df.columns)):
            var = str(new_test_df.columns[i]) + '-' + str(train_df.columns[j])
            dff[var] = abs(new_test_df.iloc[:,i].astype('int32')-train_df.iloc[:,j].astype('int32'))
    
    temp_train_x = pd.DataFrame(data=dff)
    
    return temp_train_x.T

def get_subtract_sample(x):
    x = x.transpose()
    dff = pd.DataFrame()
    for i in range(len(x.columns)):
        for j in range(i+1,len(x.columns)):
            var = str(x.columns[i]) + '-' + str(x.columns[j])
            dff[var] = abs(x.iloc[:,i].astype('int32')-x.iloc[:,j].astype('int32'))
            
    temp_train_x = pd.DataFrame(data=dff)
    #temp_train_x_0 = temp_train_x.fillna(0)
    
    return temp_train_x.T


# ## 2.  데이터 불러오기 및 데이터 준비  (공통)
# #### 2.1 설정 값 파일 읽기 (사용 된 함수)
#    - get_variables() : 날자 컬럼, 타겟 컬럼, 주기, 조건 리스트
#    -  get_dataset()  : Train & Test 데이터 세트 준비
#    - get_data_type() : x변수, y변수, 명목변수, key이름, 주기 준비

# In[ ]:


# Input 파일 
data_file_name = 'DAESANG_DATA_prepared.csv'
setting_file_name = 'input_시계열모델_설정옵션.csv'
data_type_file = 'input_시계열모델_데이터유형.csv'
condition_file_name = 'input_시계열모델_조건설정값.csv'


# ## 시나리오 1. 시계열 분석
# runAutoArima_Prediction(): Auto Arima 실행 
#    - 조건별 RD 예측
#        - 시나리오 1.1 아이템별 y값 예측
#        - 시나리오 1.2 Feature 조건별 y값 예측
#        
#        
# Output▶ 예측값 및 모델 정보 출력
# 

# In[ ]:


date_col_name, target_name, p, condition_list = get_variables(setting_file_name, condition_file_name)
df, train, test = get_dataset(data_file_name,target_name) #df: raw data, train:TD test:RD
x_val, y_val, dummy_list,item_col,predic_period = get_data_type(data_type_file) # x변수, y변수, 명목변수, key이름, 주기


# ### 시나리오 1.1 아이템 별 y 값 예측

# In[ ]:


# PART: 1019260 예측
df_model, pred_df = runAutoArima_Prediction(train,test, target_name, date_col_name, condition_list, p)
pred_df


# In[ ]:


file_name_1 = "output_시계열모델_예측결과.csv" # 예측값 엑셀로 내보내기
file_name_2 = "output_시계열모델_정보.csv"     # 모델 정보 엑셀로 내보내기
file_name_3 = "output_시계열모델_결과.csv"     # 모델 식 엑셀로 내보내기
# 결과 엑셀로 내보내기
export_to_excel(df_model,pred_df, file_name_1,file_name_2,file_name_3) # 예측값,모델 정보, 모델 식 


# ### 시나리오 1.2  조건별 y값 예측
# - 조건: 3월 냉동식품 B2C A급

# In[ ]:


condition_file_name = 'input_시계열모델_조건설정값2.csv' 
date_col_name, target_name, p, condition_list = get_variables(setting_file_name, condition_file_name)
df, train, test = get_dataset(data_file_name,target_name)
train_month = filter_df(train, condition_list)
test_month = filter_df(test, condition_list)


# In[ ]:


df_model, pred_df = runAutoArima_Prediction(train_month,test_month,target_name, date_col_name, condition_list, p)
pred_df


# In[ ]:


file_name_1 = "output_시계열모델_A급냉동3월_예측결과.csv" # 예측값 엑셀로 내보내기
file_name_2 = "output_시계열모델_A급냉동3월_정보.csv"     # 모델 정보 엑셀로 내보내기
file_name_3 = "output_시계열모델_A급냉동3월_결과.csv"     # 모델 식 엑셀로 내보내기
# 결과 엑셀로 내보내기
export_to_excel(df_model,pred_df, file_name_1,file_name_2,file_name_3) # 예측값,모델 정보, 모델 식 


# ## 시나리오 2. 유사제품 예측 (차분 & 랜덤포레스트 분석)
# 
# 
# - 2월 제품 중 상온이지만 판매 방식, 등급이 다른 4가지 제품의 QTY 예측
# - 비슷한 제품 찾기 (비슷제품 1)
# 
# **가정:**   QTY를 예측하고 싶은 제품A는 기존 제품B와 특성이 비슷
# 
# **∴ _제품A_ 는 _기존 제품B_ 의 예측값이랑 비슷할 것이다**
# - 시나리오 2.1 데이터 준비 및 더미화
# - 시나리오 2.2 모델 실행 및 유사성 예측값 출력
# - (참고1) 유사제품 순위
# - (참고2) 유사제품 시계열 예측

# ##### 1. 데이터 준비 및 더미화
# - 지정된 x와 y로 데이터셋 구성
#     - TD: x_train, y_train 
#     - RD: x_predic, y_pred(예측값)

# In[ ]:


# 실험을 위한 파라미터 
samplesize = 200
num_of_prediction = 5


# In[ ]:


condition_file_name = 'input_시계열모델_조건설정값3.csv'
date_col_name, target_name, p, condition_list = get_variables(setting_file_name, condition_file_name)
df, train, test = get_dataset(data_file_name,target_name)
dummies = [x for x in dummy_list if x in x_val] 
mon = condition_list.get(predic_period)[0]


# In[ ]:


# 더미화
try_df = pd.get_dummies(data=df[dummies])
try_df_ = pd.concat([try_df, df[[predic_period]],df[['DATA_TYPE']],df[[y_val]]],axis=1)

x_train_dummified = try_df_.loc[(try_df_[predic_period]==mon) & (try_df_['DATA_TYPE']=='TD')]
x_test_dummified = try_df_.loc[(try_df_[predic_period]==mon) & (try_df_['DATA_TYPE']=='RD')]


# In[ ]:


# 실험용
x_train_dummified = x_train_dummified[x_train_dummified['FEATURE_1_상온'] == 1] # 빠른 체크를 위한 상온만 필터
x_test_dummified = try_df_.loc[(try_df_[predic_period]==mon) & (try_df_['DATA_TYPE']=='RD')].head(num_of_prediction)


# ##### 2. 학습용 데이터셋 준비 (차분과정)

# In[ ]:


x_train = x_train_dummified.drop(['MONTH','DATA_TYPE'], axis=1) # drop month and data_type (QTY는 유지)

x_pred = x_test_dummified.drop(['MONTH','DATA_TYPE'], axis=1)
y_pred = x_test_dummified[[y_val]]


# In[ ]:


# 실험용
x_train = x_train.sample(n=samplesize)


# ##### 2.1 학습용 차분
# 
# **get_subtract_sample()** 로 차분된 데이터셋 생성

# In[ ]:


train_dummied_sub_self = get_subtract_sample(x_train)


# In[ ]:


train_dummied_sub_self.head() # 차분된 데이터 셋 


# In[ ]:


# 모델링을 위한 데이터 구분
x_train_dummied_sub_self = train_dummied_sub_self.drop([y_val], axis=1)
y_train_dummied_sub_self = train_dummied_sub_self[[y_val]]


# ##### 2.2 예측용 데이터셋 차분
# 
# **subtract()** 로 예측 데이터와 학습 데이터 차분
# - x_pred 사용

# In[ ]:


x_predict_data_sub =subtract(x_pred.T, x_train.T)
x_predict_data_sub_ = x_predict_data_sub.drop([y_val], axis=1)


# ### 시나리오 2.3 모델 실행 및 유사성 예측값 출력
# **RandomForestRegressor(X_Train_df,y_Train_df)** 랜덤포레스트 모델 적용

# In[ ]:


RF_model = RandomForestRegressor(max_depth=2, random_state=0).fit(x_train_dummied_sub_self, y_train_dummied_sub_self)
y_pred= RF_model.predict(x_predict_data_sub_)
y_pred_df = pd.DataFrame(data=y_pred)
y_pred_df.columns = ['Predicted']
y_pred_df.index = x_predict_data_sub_.index
y_pred_df_ =  index_match_prediction(y_pred_df) # 결과값
y_pred_df_ #RD_index would be smilar to TD_index, presucted by subtracted distance 


# In[ ]:


file_name_1 = "output_랜덤포레스트_차분_유사성결과.csv" # 유사값 엑셀로 내보내기

# 결과 엑셀로 내보내기
outputfile(y_pred_df_,file_name_1)


# #### (참고) 유사성 랭크 순위 

# In[ ]:


group_list = list(range(num_of_prediction)) # size of RD
res =  [ele for ele in group_list for i in range(samplesize)]
res1 =  pd.DataFrame(res, columns = ['Group'], index = y_pred_df.index) 
Grouped = pd.concat([res1,y_pred_df ],axis=1) 
# 제품간의 차분 거리가 가장 작은 top1 
result_top1 = Grouped.drop_duplicates().groupby('Group', group_keys=False).apply(lambda x: x.sort_values('Predicted', ascending=True)).groupby('Group').head(1)
result_top1


# In[ ]:


# 제품간의 차분 거리가 가장 작은 top5 
result_top5 = Grouped.drop_duplicates().groupby('Group', group_keys=False).apply(lambda x: x.sort_values('Predicted', ascending=True)).groupby('Group').head(5)
result_top5 


# #### (참고) 유사제품의 시계열 예측 

# In[ ]:


# mapping PART with index
index_list,item_list_by_index,index_list_for_Predict,item_list_for_Predict = find_similar_Item_ID(result_top1,item_col)


# In[ ]:


for i in range(len(item_list_by_index)):
    item = item_list_by_index[i]
    y_pred_by_Item, df_model_by_item = Prediction_per_item (train,test,item_col,item, y_val, date_col_name, p)
    print("예측 item:",item_list_for_Predict[i], ",   기존 item", item_list_by_index[i])


# ## 시나리오 3. 월평균 예측 (선형분석)
# 
#     Y= c + a1.X1 + a2.X2 + a3.X3 + a4.X4 +a5X5 +a6X6 ...

# #### 아이템별 월평균 데이터 셋 생성

# In[ ]:


change_col_to_int(train, 'YEAR')
x_train = pd.pivot_table(train, index=item_col,values=x_val, aggfunc='first')
x_train_dumm = pd.get_dummies(data=x_train)
y_train = train[['PART','QTY']].groupby('PART').mean()

y_test = pd.pivot_table(test, index=item_col,values=x_val, aggfunc='first')


# ####  선형 모델실행

# In[ ]:


lin_reg = sm.OLS(y_train, x_train_dumm).fit()

x_predict = pd.pivot_table(test, index=item_col,values=x_val, aggfunc='first')
x_predict_dumm = pd.get_dummies(data=x_predict)

y_pred_year = lin_reg.predict(x_predict_dumm)
lin_reg.summary()


# In[ ]:


file_name_1 = "output_월평균_선형_예측.csv" # 예측값 엑셀로 내보내기
file_name_2 = "output_월평균_선형_예측모델_정보.csv"     # 모델 정보 엑셀로 내보내기
file_name_3 = "output_월평균_선형_예델모델_결과.csv"     # 모델 식 엑셀로 내보내기
model_info_df,model_result_df,Model_ver,results_pred_df = export_results_reg(lin_reg,y_pred_year,y_test,file_name_1,file_name_2,file_name_3)


# ### (참고: 중요도 및 상관분석)

# In[ ]:


train_= x_train_dumm.join(y_train)


# #### Univariate Selection : 통계 테스트를 통해 변수들 간의 관계 이해 

# In[ ]:



# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(train_.iloc[:,:-1],train_.iloc[:,0])
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(train_.iloc[:,:-1].columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
featureScores_df = featureScores.sort_values(by=['Score'], ascending=False)
featureScores_df 


# #### Feature importance : 변수 중요도 확인

# In[ ]:


model = ExtraTreesClassifier()
fit = model.fit(train_.iloc[:,:-1],train_.iloc[:,0])
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=train_.iloc[:,:-1].columns)
clrs = ['grey' if (x < max(model.feature_importances_)) else 'blue' for x in model.feature_importances_ ]
feat_importances.nlargest(10).plot(kind='barh',color=clrs)
plt.show()


# In[ ]:


feature_importances_df = pd.DataFrame(model.feature_importances_, index=train_.iloc[:,:-1].columns)
feature_importances_df.columns = ['Score']
feature_importances_df


# #### x 변수들의 상관분석

# In[ ]:


corrmat = train_.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(13,13))
#plot heat map
g=sns.heatmap(train_[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


train_.corr()


# In[ ]:





# In[ ]:





# In[ ]:




