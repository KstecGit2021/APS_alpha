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

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation,Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler

import pickle
import joblib
from keras.models import model_from_json
import re


# ## step1. Read data & data preparation

# In[2]:


read_data_file = 'new_DAESANG_DATA.csv'
read_col_info_file = 'input_AutoML_데이터유형.csv' 
read_model_info_file = 'input_AutoML_설정옵션.csv' # 1: Auto , 2: OLS , 3: N.N


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
    
    # 차분을 더미화가 아닌 같다 아니다로 판별할 것이기에 필요 없음. 만약 TD의 column과 RD의 column이 일치하면 두가지방법(더미화 / 같음다름구별) 중 선택
#     # 더미화 할 col정보
#     ele = [x for x in dummy_list if x in X_Train] # 더미화가 필요한 col중에 train에 들어가지 않는 것이 있을 수 있으므로 진행

#     # 더미화 형태의 X로 바꿈
#     X_Train = pd.get_dummies(data=X_Train, columns=ele)
#     X_Predict = pd.get_dummies(data=X_Predict, columns=ele)

    return X_Train, y_Train, X_Predict, y_Predict


# ## step2. 빠른 연산을 위한 단계 (선택사항) -> 전체 데이터로 모델링할시 step3로 이동
# 
# - 파일 정보를 불러와 일부샘플을 뽑은 뒤 차분계산

# In[5]:


# 빠른 결과값을 위해 일부로만 샘플진행
train_num = 300
test_num = 50


# In[6]:


def small_sample(train_num, test_num, Train, Predict):
    # 빠른 결과값을 위해 일부로만 샘플진행
    Train = Train.sample(n=train_num) # 300개
    Predict = Predict.sample(n=test_num) # 50개

    return Train, Predict


# ## step3. 차분계산

# In[7]:


def subtract_string(df):
    df = df.transpose()
    new_df = pd.DataFrame()
    for i in range(len(df.columns)):
        for j in range(i+1,len(df.columns)):
            var = str(df.columns[i]) + '-' + str(df.columns[j])
            dff_df = df.iloc[:,i].isin(df.iloc[:,j]).astype(int) # difference check 0 if True, 1 False
            new_df[var] = np.logical_xor(dff_df,1).astype(int)  # XOR 다르면 1 같으면 0 for distance differentiation 
    
    new_df_ = pd.DataFrame(data=new_df)
    return new_df_.T


# In[8]:


def subtract_integer(x):
    x = x.transpose()
    dff = pd.DataFrame()
    for i in range(len(x.columns)):
        for j in range(i+1,len(x.columns)):
            var = str(x.columns[i]) + '-' + str(x.columns[j])
            dff[var] = abs(x.iloc[:,i].astype('int32')-x.iloc[:,j].astype('int32'))
            
    temp_train_x = pd.DataFrame(data=dff)    
    return temp_train_x.T


# In[9]:


def subtract_btw_df_STR (new_test_df,train_df): # string 
    dff = pd.DataFrame()
    
    new_test_df = new_test_df.T
    train_df = train_df.T
    for i in range(len(new_test_df.columns)):
        for j in range(len(train_df.columns)):
            var = str(new_test_df.columns[i]) + '-' + str(train_df.columns[j])
            dff_df = new_test_df.iloc[:,i].isin(train_df.iloc[:,j]).astype(int) # difference check 0 if True, 1 False
            dff[var] = np.logical_xor(dff_df,1).astype(int)  # XOR 다르면 1 같으면 0 for distance differentiation 
    
    temp_train_x = pd.DataFrame(data=dff) 
    return temp_train_x.T


# In[10]:


# new data (X_Predict) 와 X_Train의 차분 계산
def subtract_btw_df_INT (new_test_df,train_df):
    dff = pd.DataFrame()
    
    new_test_df = new_test_df.T
    train_df = train_df.T
    for i in range(len(new_test_df.columns)):
        for j in range(len(train_df.columns)):
            var = str(new_test_df.columns[i]) + '-' + str(train_df.columns[j])
            dff[var] = abs(new_test_df.iloc[:,i].astype('int32')-train_df.iloc[:,j].astype('int32'))
    
    temp_train_x = pd.DataFrame(data=dff)
    
    return temp_train_x.T


# In[11]:


# 차분 계산
def get_diff_df(X_Train, X_Predict, y_Train, y_Predict):
    
    # [차분 과정1.] 회귀식을 위한 데이터 준비 
    X_Train_df = subtract_string(X_Train) # 학습 데이터 차분 (subtracting inside) (row1 - row0)
    X_Predict_df = subtract_btw_df_STR(X_Predict, X_Train) # 예측 데이터셋을 위한 차분  (df1[row1] - df2[row1])
    y_Train_df = subtract_integer(y_Train) # 학습 데이터 y 에 대한 차분 (y[row1] - y[row0])
    y_Predict_df = subtract_btw_df_INT(y_Predict, y_Train)
    # 알아보기 쉽게 이름 재 구성 _df를 제외하고 사용해야 일반회귀모델링과 같은 main함수 사용가능
    
    return X_Train_df, y_Train_df, X_Predict_df, y_Predict_df


# ## step4. 파일 저장경로 및 파일 저장정보

# In[12]:


def get_model_var(df,Model_ver):  #모델버전 생성 
    df = df.reset_index(drop=False)
    Model_ver_list =  [Model_ver] * len(df)
    Model_ver_list = pd.DataFrame(Model_ver_list, columns =['모델 버전'])
    
    updated_df = pd.concat([Model_ver_list,df] ,axis=1)
    return updated_df


# In[13]:


# 모델 저장
def outputfile(sheet1,output_file_name):     
    sheet1.to_csv(output_file_name, encoding ='utf-8-sig')
    print("\n폴더에서",output_file_name,"파일을 확인하세요")


# In[14]:


def index_match_prediction(y_pred_df):
    indexs = pd.DataFrame(y_pred_df.index.str.split('-',1).tolist(),
                                 columns = ['RD_index','TD_index'])
    y_pred_df_ = y_pred_df.reset_index(drop=True)
    result = pd.concat([indexs,y_pred_df_], axis=1)
    return result   


# In[15]:


def get_simple_results(model, X_Predict_df, y_val, Predict): # 식 결과 export into excel
    model_results = model.summary()

    model_info = model_results.tables[0].as_html()
    model_info = pd.read_html(model_info, header=0, index_col=0)[0]
    
    model_result = model_results.tables[1].as_html()
    model_result = pd.read_html(model_result, header=0, index_col=0)[0] # Excel 내보내기
    
#     return model_info, model_result
    Model_ver = model_info[y_val][0] + "_" + model_info[y_val][1] + "_"+model_info[y_val][2]
   
    model_info_df = get_model_var(model_info,Model_ver)
    model_info_df.columns = pd.MultiIndex.from_tuples(zip(c_columns(model_info_df), model_info_df.columns)) 
    
    model_result_df = get_model_var(model_result,Model_ver)
    model_result_df.columns = pd.MultiIndex.from_tuples(zip(c_columns(model_result_df), model_result_df.columns)) 
    
    filename = re.sub('[<>.]','',str(model).split()[3])
    model_file="{model}.pkl".format(model=filename)
    
    joblib.dump(model, model_file)
    clf_from_joblib = joblib.load(model_file)  
    
    prediction = clf_from_joblib.predict(X_Predict_df) # 예측값구하는 식
    prediction_df = pd.DataFrame(data=prediction)
    prediction_df.columns = ['Predicted']
    prediction_df.index = X_Predict_df.index
    
    #prediction_df.index = range(len(prediction_df.index))
    
    results_pred_df= get_model_var(prediction_df, Model_ver)
    results_pred_df_= results_pred_df.set_index('index').join(Predict)
    results_pred_df_[y_val] = results_pred_df_['Predicted']
    results_pred_df_ = results_pred_df_.drop(columns=['Predicted'])
    
    outputfile(results_pred_df_,output_file_name1) # 예측값 엑셀로 내보내기
    outputfile(model_info_df,output_file_name2) # 모델 정보 엑셀로 내보내기
    outputfile(model_result_df,output_file_name3) # 모델 식 엑셀로 내보내기
    
    return model_info_df,model_result_df,Model_ver,results_pred_df_


# In[16]:


# 예측값 출력
def get_model_results (model, X_Predict_df):
    
    string_model = re.sub("\n","",str(model)).replace(" ","")
    suffix = pd.datetime.now().strftime("%y%m%d_%H%M%S") # 파일이 돌아가기 시작한 시간을 기준으로 이름 생성
    model_ = "_".join([suffix, string_model])
    Model_ver = pd.DataFrame([model_], columns=['모델정보'])
    
    filename = model_.split('(',1)[0]
    model_file="{model}.pkl".format(model=filename)
    
    joblib.dump(model, model_file)
    clf_from_joblib = joblib.load(model_file)  
    
    prediction = clf_from_joblib.predict(X_Predict_df) # 예측값구하는 식
    prediction_df = pd.DataFrame(data=prediction)
    prediction_df.columns = ['Predicted']
    prediction_df.index = X_Predict_df.index
    
    results_pred_df = index_match_prediction(prediction_df)
    # results_pred_df_= get_model_var(prediction_df, Model_ver)    
    
    outputfile(Model_ver,output_file_name1)
    outputfile(results_pred_df,output_file_name2)
        
    return Model_ver, results_pred_df


# In[17]:


def get_neural_results (model, X_Predict_df):
    
    Model_ver = pd.DataFrame([str(model)], columns=['모델정보'])
    
    filename = re.sub('[<>.]','',str(model).split()[3])
    model_file1 = "{model}.json".format(model=filename)
    model_file2 = "{model}.h5".format(model=filename)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file1, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_file2)

    prediction = model.predict(X_Predict_df) # 예측값구하는 식
    prediction_df = pd.DataFrame(data=prediction)
    prediction_df.columns = ['Predicted']
    prediction_df.index = X_Predict_df.index # scale하면 ndarray형태로 바껴서 index는 없음. (참고사항)
    
    results_pred_df = index_match_prediction(prediction_df)
    # results_pred_df_= get_model_var(prediction_df, Model_ver)    
    
    outputfile(Model_ver,output_file_name1)
    outputfile(results_pred_df,output_file_name2)
        
    return Model_ver, results_pred_df


# # RUN Model 

# In[18]:


# model select
def main():
    
    # 데이터 분리
    dummy_list, x_val, y_val, id_val, month_val, Train, Predict, model_name, month_name = read_data_info (read_data_file, read_col_info_file, read_model_info_file)
    
    # 샘플뽑아 진행 (생략가능)
    Train, Predict = small_sample(train_num, test_num, Train, Predict)
    
    # make dataset 
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = make_model_df (dummy_list, x_val, y_val, id_val, month_val, Train, Predict, month_name)
    
    # 차분공식
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = get_diff_df(X_Train_df, X_Predict_df, y_Train_df, y_Predict_df)
    
    # -------------- 모델 선택 -------------------
    if model_name == 'logit': # 로지스틱 
        
        # 현재 dataset은 logit에 맞는 형태가 아니기에 임의로 변경해서 확인하는 작업입니다. 
        y_Train_df.loc[y_Train_df[y_val] > np.mean(y_Train_df[y_val]), y_val]=1
        y_Train_df.loc[y_Train_df[y_val] > 1, y_val]=0
    
        model = sm.Logit(y_Train_df, X_Train_df).fit() 
        get_simple_results(model, X_Predict_df, y_val, Predict)
        
    elif model_name == 'MNlogit': # 다중 로지스틱
        model = sm.MNLogit(y_Train_df, X_Train_df).fit() 
        get_simple_results(model, X_Predict_df, y_val, Predict)

    elif model_name == 'OLS': # 선형회귀
        model = sm.OLS(y_Train_df, X_Train_df).fit()
        get_simple_results(model, X_Predict_df, y_val, Predict)
        
    elif model_name == 'Random_fore': # 랜덤포레스트
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
        
    # 신경망 (Deep learning)
    elif model_name == 'Neural_net':

        # scaling 하는 또다른 방법. 적용하였으면 추후 재 되돌리는 코드 필요. LSTM 코드 참조
        #sc = StandardScaler()
        #X_Train_df = sc.fit_transform(X_Train_df)
        #y_Train_df = sc.fit_transform(y_Train_df)
        #X_Predict_df = sc.fit_transform(X_Predict_df)
        #X_Predict_df = sc.fit_transform(y_Predict_df)

        # Initialising the ANN
        model = Sequential()

        # Adding the input layer and the first hidden layer
        model.add(Dense(10, activation = 'relu', kernel_initializer='normal',  input_dim = X_Train_df.shape[1]))
        
        # Adding the second hidden layer
        model.add(Dense(units = 8, activation = 'relu'))
        # model.add(Dropout(0.5))
        
        # Adding the third hidden layer
        # model.add(Dense(units = 4, activation = 'relu'))   #  레이어 추가
        # model.add(Dropout(0.5))
        
        # Adding the output layer
        model.add(Dense(units = 1, activation='relu'))
        model.compile(optimizer = 'rmsprop',loss = 'mean_squared_error', metrics=['accuracy'])
        model.fit(X_Train_df, y_Train_df, batch_size = 10, epochs = 150, verbose=0) # callback 안함. 필요시 LSTM 코드 참조 추가
        
        get_neural_results (model, X_Predict_df)
        
          
    else: 
        print('Please select your data model')


# In[19]:


# Auto_ML
if __name__ == "__main__":  
    read_model_info_file = 'input_AutoML_설정옵션.csv'
    output_file_name1 = 'output_AutoML_수요증감예측_모델.csv'
    output_file_name2 = 'output_AutoML_월간수요증감예측_예측값.csv'
    main()


# ## 저장된 모델 불러오기

# In[20]:


def load_model_keras(jsonfile, h5file, new_RD):
    json_file = open(jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5file)
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    
    pred = loaded_model.predict(new_RD) # 예측값구하는 식
    pred_df = pd.DataFrame(data=pred)
    pred_df.columns = ['Predicted']
    pred_df.index = new_RD.index
    
    results_pred_df = index_match_prediction(pred_df)
    
    outputfile(results_pred_df, output_file_name)
    
    return results_pred_df


# In[21]:


def load_model_sklearn(filename, new_RD):
    clf_from_joblib = joblib.load(filename) 
    pred = clf_from_joblib.predict(new_RD)
    pred_df = pd.DataFrame(data=pred)
    pred_df.columns = ['new_Predicted']
    pred_df.index = new_RD.index
    
    results_pred_df = index_match_prediction(pred_df)

    outputfile(results_pred_df, output_file_name)
    
    return results_pred_df


# ## new_RD 임의 생성 및 모델 저장 확인

# In[22]:


def main2():

    dummy_list, x_val, y_val, id_val, month_val, Train, Predict, model_name, month_name = read_data_info (read_data_file, read_col_info_file, read_model_info_file)
    
    # 샘플뽑아 진행 (생략가능)
    Train, Predict = small_sample(train_num, test_num, Train, Predict)
    
    # make dataset 
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = make_model_df (dummy_list, x_val, y_val, id_val, month_val, Train, Predict, month_name)
    
    # 차분공식
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = get_diff_df(X_Train_df, X_Predict_df, y_Train_df, y_Predict_df)
    new_RD = X_Predict_df
    
    if model_name == 'logit' or model_name == 'OLS' or model_name == 'Auto_reg':
        load_model_sklearn (file, new_RD)
        
    elif model_name == 'Neural_net' or model_name == 'Random_fore':
        load_model_keras(jsonfile, h5file, new_RD)
        
    else: 
        print('Please select your data model')


# In[24]:


# auto
if __name__ == "__main__":
    read_model_info_file = 'input_AutoML_설정옵션.csv'
    file = "210331_111416_AdaBoostRegressor.pkl"
    output_file_name = 'output_AutoML_재사용_월별수요증감_예측값.csv'
    main2()


# In[ ]:





# In[ ]:





# In[ ]:




