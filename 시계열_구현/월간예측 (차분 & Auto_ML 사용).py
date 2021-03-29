#!/usr/bin/env python
# coding: utf-8

# **Input 정보**
# - 데이터 셋 파일명: new_DAESANG_DATA.csv
# - 설정 옵션 파일명: input_AutoML_설정옵션.csv
# - 데이터 유형 파일명: input_AutoML_데이터유형.csv
# 
# **모델정보**
# - auto_modelling
# - auto_modelling 처음 실행시 **!pip install auto_modelling** 코드 실행 필요
# 
# 
# **코드설명**
# - 시나리오 2-1. 월간예측 (차분 & Auto_ML 사용)
#     - 월별 RD 예측 (ID별예측, feature 조건별 예측)
# 
# **Output 정보**
# - 예측 결과
# - 예측 모델 정보

# In[1]:


import csv
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn import preprocessing
from auto_modelling.classification import GoClassify
from auto_modelling.regression import GoRegress
from auto_modelling.preprocess import DataManager
from auto_modelling.stack import Stack
import logging

from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose 
from pandas import DataFrame

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA

import pmdarima
from pmdarima import auto_arima


# ## step1. Read data & data preparation

# In[2]:


read_data_file = 'new_DAESANG_DATA.csv'
read_col_info_file = 'input_AutoML_데이터유형.csv'
read_model_info_file = 'input_AutoML_설정옵션.csv'


# ### 1.1 기본 정보
# 
# - 데이터 파일 읽기
# - 모델링에 필요한 데이터 유형, 역할, 예측 주기 파라미터 값 사용자 지정 파일에서 불러오기
# 

# 1. 파일명 입력 -> 데이터 읽기/사용자 지정 조건들 읽기 (파라미터값들)
#  
#  
# 2. 차분만 이용/ 회귀식 선택시 -> Split Train & Test data set
# 
# 
#     2.1 Run 회귀식/데이터모델
# 
#     2.2 결과 출력
# 
# 
# 3. 월간 예측 전처리 (1.2) 필요/ 차분 회귀식 선택  
# 
# 
#     3.1 피봇/더미화 (전처리)
#  
#     3.2 분석모델 실행
#  
#     3.3 결과 출력 

# In[3]:


def read_data_info (read_data_file, read_col_info_file, read_model_info_file):
    # input data
    data = pd.read_csv(read_data_file) # 년도와 월을 split해서 new data 생성

    # col info
    col_info = data.iloc[0] # col 정보 ex. int / str / month_no / ...
    dummy_list = col_info.loc[col_info=='STR'].index.tolist() # str부분 더미화처리해야함
    id_val = col_info.loc[col_info=='STR_KEY'].index.tolist()[0] # key는 자동으로 id 인식

    # input role info
    role_info = pd.read_csv(read_col_info_file, encoding ='cp949') # 모델링 할 때 사용할 x, y, month
    x_val = role_info.loc[role_info['Role']=='x', 'col_name'].tolist() # x변수 다중리스트형태
    y_val = role_info.loc[role_info['Role']=='y', 'col_name'].tolist()[0] # y변수는 단일
    month_val = role_info.loc[role_info['예측주기']=='p', 'col_name'].tolist()[0] # month값은 단일

    # input model info 모델과 예측달 정하기
    model_info = pd.read_csv(read_model_info_file)
    model_name = model_info['Model'][0] # 기본으로 auto 지정
    month_name = model_info['예측월'][0] # 예측하고 싶은 월 예:) 3월 QTY예측 / 10월 QTY 예측
    # 맨처음 month로 불러온 것과 동일하게 이름작성 필수 ex. Mar (o) / March (X) / 3 (X)

    # data split
    Train = data[data['DATA_TYPE']=='TD'] # 학습할 데이터
    Train[[y_val]] = Train[[y_val]].fillna(0).astype('int32') # 학습할 데이터
    Predict = data[data['DATA_TYPE']=='RD'] # 예측해야할 데이터 
    Predict[[y_val]] = Predict[[y_val]].fillna(0).astype('int32')
    
    return dummy_list, x_val, y_val, id_val, month_val, Train, Predict, model_name, month_name


# ### 1.2 전처리 과정 
# - 월간예측을 위한 전처리 과정
# - 예측하고 싶은 달에 대한 예측값 출력을 위한 전처리 과정
# - 완성된 Train, Predict data를 각각 role에 따라 분리한 후 col정보에 따른 더미화 진행

# In[4]:


def make_model_df (dummy_list, x_val, y_val, id_val, month_val, Train, Predict, month_name):
    
    # part(id)에 따른 X, y train 할 값들 - 피봇이용해서 만들기
    X_Train = pd.pivot_table(Train, index=id_val,values=x_val, aggfunc='first') # id에 따른 x 값 # 제품간 feature는 동일하므로 첫번째 값 출력
    y_Train = pd.pivot_table(Train, index=id_val, columns=month_val, values=y_val, aggfunc = np.mean, fill_value = 0) # id에 따른 월별 예측을 위해 월별 평균/ 결측값은 0처리
    X_Predict = pd.pivot_table(Predict, index=id_val,values=x_val, aggfunc='first') # Train과 col은 항상 일치해야함
    #  y_Predict는 현재 없음, 있다면 실측값과 예측값을 비교해서 정확도 및 mse 확인 가능
    y_Predict = pd.pivot_table(Predict, index=id_val, columns=month_val, values=y_val, aggfunc = np.mean, fill_value = 0)
    
    # 월별 qty를 위해 분리 # 모델불러올 때 예측하고 싶은 달 불러 올 것 - 아래참고
    mons = []
    for i in range(len(y_Train.columns)):
        mon = pd.DataFrame(y_Train.iloc[:,i])
        mons.append(mon)
    for month in mons:
        if month.columns == month_name:
            month_df = pd.DataFrame(month)
    # 즉, month가 예측하고 싶은 달이 됨

    # 따라서 y_Train을 예측하고 싶은 것으로 재설정
    y_Train = month_df # 월 QTY
    
    # y_Predict를 위해 같은 과정을 한번 더 해줌
    mons2 = []
    for i in range(len(y_Predict.columns)):
        mon2 = pd.DataFrame(y_Predict.iloc[:,i])
        mons2.append(mon2)
    for month2 in mons2:
        if month2.columns == month_name:
            month2_df = pd.DataFrame(month2)

    y_Predict = month2_df # 월 QTY
    
    # 더미화 할 col정보
    ele = [x for x in dummy_list if x in X_Train] # 더미화가 필요한 col중에 train에 들어가지 않는 것이 있을 수 있으므로 진행

    # 더미화 형태의 X로 바꿈
    X_Train = pd.get_dummies(data=X_Train, columns=ele)
    X_Predict = pd.get_dummies(data=X_Predict, columns=ele)

    return X_Train, y_Train, X_Predict, y_Predict


# ## step2. 빠른 연산을 위한 단계 (선택사항) -> 전체 데이터로 모델링할시 step3로 이동
# 
# - 파일 정보를 불러와 일부샘플을 뽑은 뒤 차분계산

# In[5]:


# 빠른 결과값을 위해 일부로만 샘플진행
train_num = 300
test_num = 50


# In[6]:


def small_sample(train_num, test_num, X_Train, X_Predict, y_Train, y_Predict):
    
    # 빠른 결과값을 위해 일부로만 샘플진행
    X_Train = X_Train.head(train_num) # 300개
    X_Predict = X_Predict[round(X_Predict.shape[0]/2) : round(X_Predict.shape[0]/2) + test_num] # 50개

    # 예측하고 싶은 월을 y로 둠
    y_Train = y_Train.head(train_num)
    y_Predict = y_Predict[round(y_Predict.shape[0]/2) : round(y_Predict.shape[0]/2) + test_num] 

    return X_Train, y_Train, X_Predict, y_Predict


# ## step3. 차분계산

# In[7]:


# Train 에 대한 차분계산
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


# In[8]:


# new data (X_Predict) 와 X_Train의 차분 계산
def subtract (new_test_df,train_df):
    dff = pd.DataFrame()
    for i in range(len(new_test_df.columns)):
        for j in range(len(train_df.columns)):
            var = str(new_test_df.columns[i]) + '-' + str(train_df.columns[j])
            dff[var] = abs(new_test_df.iloc[:,i].astype('int32')-train_df.iloc[:,j].astype('int32'))
    
    temp_train_x = pd.DataFrame(data=dff)
    
    return temp_train_x.T


# In[9]:


# 차분 계산
def get_diff_df(train_num, test_num, X_Train, X_Predict, y_Train, y_Predict):
    
    # [차분 과정1.] 회귀식을 위한 데이터 준비 
    X_Train_df = get_subtract_sample(X_Train) # 학습 데이터 차분 (subtracting inside) (row1 - row0)
    X_Predict_df = subtract(X_Predict.T, X_Train.T) # 예측 데이터셋을 위한 차분  (df1[row1] - df2[row1])
    y_Train_df = get_subtract_sample(y_Train) # 학습 데이터 y 에 대한 차분 (y[row1] - y[row0])
    y_Predict_df = subtract(y_Predict.T, y_Train.T)
    # 알아보기 쉽게 이름 재 구성 _df를 제외하고 사용해야 일반회귀모델링과 같은 main함수 사용가능
    
    return X_Train_df, y_Train_df, X_Predict_df, y_Predict_df


# ## step4. 파일 저장경로 및 파일 저장정보

# In[10]:


def get_model_var(df,Model_ver):  #모델버전 생성 
    df = df.reset_index(drop=False)
    Model_ver_list =  [Model_ver] * len(df)
    Model_ver_list = pd.DataFrame(Model_ver_list, columns =['모델 버전'])
    
    updated_df = pd.concat([Model_ver_list,df] ,axis=1)
    return updated_df


# In[11]:


# 모델 저장
def outputfile(sheet1): 
    output_file_name = input("▼ 결과 파일명을 입력해주세요 xx.csv: \n") 
    print("결과 파일명:",output_file_name) 
    
    sheet1.to_csv(output_file_name, encoding ='utf-8-sig')
    print("\n폴더에서",output_file_name,"파일을 확인하세요")


# In[12]:


def index_match_prediction(y_pred_df):
    indexs = pd.DataFrame(y_pred_df.index.str.split('-',1).tolist(),
                                 columns = ['RD_index','TD_index'])
    y_pred_df_ = y_pred_df.reset_index(drop=True)
    result = pd.concat([indexs,y_pred_df_], axis=1)
    return result   


# In[13]:


def get_simple_results(model, X_Predict_df, y_val): # 식 결과 export into excel
    model_results = model.summary()

    model_info = model_results.tables[0].as_html()
    model_info = pd.read_html(model_info, header=0, index_col=0)[0]
    
    model_result = model_results.tables[1].as_html()
    model_result = pd.read_html(model_result, header=0, index_col=0)[0] # Excel 내보내기
    
#     return model_info, model_result
    Model_ver = model_info[y_val][0] + "_" + model_info[y_val][1] + "_"+model_info[y_val][2]
   
    model_info_df = get_model_var(model_info,Model_ver)
    model_result_df = get_model_var(model_result,Model_ver)
    
    prediction = model.predict(X_Predict_df) # 예측값구하는 식
    prediction_df = pd.DataFrame(data=prediction)
    prediction_df.columns = ['Predicted']
    prediction_df.index = X_Predict_df.index
    
    
    results_pred_df = index_match_prediction(prediction_df)
    # results_pred_df_= get_model_var(prediction_df, Model_ver)

    outputfile(results_pred_df) # 예측값 엑셀로 내보내기
    outputfile(model_info_df) # 모델 정보 엑셀로 내보내기
    outputfile(model_result_df) # 모델 식 엑셀로 내보내기
    
    return model_info_df,model_result_df,Model_ver,results_pred_df


# In[14]:


# 예측값 출력
def get_model_results (model, X_Predict_df):
    
    Model_ver = pd.DataFrame([str(model)], columns=['모델정보'])
    
    prediction = model.predict(X_Predict_df) # 예측값구하는 식
    prediction_df = pd.DataFrame(data=prediction)
    prediction_df.columns = ['Predicted']
    prediction_df.index = X_Predict_df.index
    
    results_pred_df = index_match_prediction(prediction_df)
    # results_pred_df_= get_model_var(prediction_df, Model_ver)    
    
    outputfile(Model_ver)
    outputfile(results_pred_df)
        
    return Model_ver, results_pred_df


# # RUN Model 

# In[15]:


# model select
def main():
    
    # 데이터 분리
    dummy_list, x_val, y_val, id_val, month_val, Train, Predict, model_name, month_name = read_data_info (read_data_file, read_col_info_file, read_model_info_file)
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = make_model_df (dummy_list, x_val, y_val, id_val, month_val, Train, Predict, month_name)
    
    # 샘플뽑아 진행 (생략가능)
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = small_sample(train_num, test_num, X_Train_df, X_Predict_df, y_Train_df, y_Predict_df)
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = get_diff_df(train_num, test_num, X_Train_df, X_Predict_df, y_Train_df, y_Predict_df)
    
    # 세가지는 일반 모델 직접 불러올 수 있도록 샘플링
    # 예시
    if model_name == 'logit':
        model = sm.Logit(y_Train_df, X_Train_df).fit() 
        get_simple_results(model, y_Predict_df, X_Predict_df, y_val)

    elif model_name == 'OLS':
        model = sm.OLS(y_Train_df, X_Train_df).fit()
        get_simple_results(model, y_Predict_df, X_Predict_df, y_val)
        
    elif model_name == 'MNlogit':
        model = sm.MNLogit(y_Train_df, X_Train_df).fit() 
        get_simple_results(model, y_Predict_df, X_Predict_df, y_val)
    
    elif model_name == 'Random_fore':
        model = RandomForestRegressor(max_depth=2, random_state=0).fit(X_Train_df, y_Train_df) 
        get_model_results (model, X_Predict_df)
        
    # 현재 우리가 필요한 문제 auto_reg로 자동화 회귀모델링
    # auto 모델의 경우 predict를 할 수 있는 reg와 분류작업을 위한 classifi를 직접 지정받아야하는 부분입니다. 
    elif model_name == 'Auto_classi':
        model =  GoClassify(n_best=1).train(X_Train_df, y_Train_df)
        get_model_results (model, X_Predict_df)
        
    elif model_name == 'Auto_reg':
        model =  GoRegress(n_best=1).train(X_Train_df, y_Train_df)
        get_model_results (model, X_Predict_df)
        
    else: 
        print('Please select your data model')


# In[16]:


if __name__ == "__main__":  
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




