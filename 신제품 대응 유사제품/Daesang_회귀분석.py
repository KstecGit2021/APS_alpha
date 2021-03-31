#!/usr/bin/env python
# coding: utf-8

# In[1]:


# step0. import library
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

# Auto model
from auto_modelling.classification import GoClassify
from auto_modelling.regression import GoRegress
from auto_modelling.preprocess import DataManager
from auto_modelling.stack import Stack
import logging

# Neural Network
import keras
import ast
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional, Activation, Dropout
from keras import backend as K
from keras.optimizers import RMSprop

# save file
import pickle
import joblib
from keras.models import model_from_json
import re

pd.set_option('display.max_rows', 500) # 많은 데이터를 한눈에 볼 수 있도록


# # 회귀모델 (Auto_ML, 선형(OLS), 이항선형회귀(Logistic))
# 
# 아이템 별 월평균 데이터 셋 생성
# 
# **_모델 종류_**
# - OLS: 선형회귀
# - MNLogit: 이항회귀
# - RandomForestRegressor
# - GoRegress: Auto_ML 
# - GoClassify: Auto_ML
# - Neural_net: 신경망,Neural Network (분류)
# 
# 
# - 예측 모델의 입출력 값
#   - Input(x)은 신규 아이템에 대한 속성(이전에 _있던_ 속성 값)
#   - Output(y)는 기존 아이템에 대한 평균 QTY 예측값
#   
# 
# ### 모델을 파일로 저장
# 
# - keras(케라스)를 쓰는 경우는 json / h5 파일로 모델 저장
# - sklearn를 쓰는 경우는 pickle 파일로 모델 저장

# In[2]:


# step1. Read data & data preparation
read_data_file = 'new_DAESANG_DATA.csv'
read_col_info_file = 'input_LSTM_데이터유형.csv' # 시계열로 변환이 필요한 예측주기는 일반회귀에선 사용 안함.


# In[3]:


# Functions 
def read_data_info (read_data_file, read_col_info_file, read_model_info_file):
    # input data
    data = pd.read_csv(read_data_file) # 년도와 월을 split해서 new data 생성

    # input role info
    role_info = pd.read_csv(read_col_info_file, encoding ='cp949') # 모델링 할 때 사용할 x, y, month
    x_val = role_info.loc[role_info['Role']=='x', 'col_name'].tolist() # x변수 다중리스트형태
    y_val = role_info.loc[role_info['Role']=='y', 'col_name'].tolist()[0] # y변수는 단일
    predic_period = role_info.loc[role_info['예측주기']=='P', 'col_name'].tolist()[0]
    
    # col info
    dummy_list = role_info.loc[role_info['col_info']=='STR', 'col_name'].tolist() 
    
    # input model info 모델과 예측달 정하기
    model_info = pd.read_csv(read_model_info_file)
    model_name = model_info['Model'][0] # 기본으로 auto_reg 지정
    
    # data split
    Train = data[data['DATA_TYPE']=='TD'] # 학습할 데이터
    Train[[y_val]] = Train[[y_val]].fillna(0).astype('int32') # 학습할 데이터
    Predict = data[data['DATA_TYPE']=='RD'] # 예측해야할 데이터 
    Predict[[y_val]] = Predict[[y_val]].fillna(0).astype('int32')
    
    return dummy_list, x_val, y_val, predic_period, Train, Predict, model_name


def make_model_df (dummy_list, x_val, y_val, Train, Predict):
    
    # Train, Predict에서 role에 따른 값을 각각 X, y로 둠
    X_Train = Train[x_val]
    y_Train = Train[[y_val]].astype(int) # y_val은 값만 불러왔기에 이중리스트형태로 사용해야 dataframe형태로 출력
    
    X_Predict = Predict[x_val]
    y_Predict = Predict[[y_val]]
    #  y_Predict는 현재 없음, 있다면 실측값과 예측값을 비교해서 정확도 및 mse 확인 가능
    
    # 더미화 할 col정보
    ele = [x for x in dummy_list if x in X_Train] # 더미화가 필요한 col중에 train에 들어가지 않는 것이 있을 수 있으므로 진행

    # 더미화 형태의 X로 바꿈
    X_Train = pd.get_dummies(data=X_Train, columns=ele)
    X_Predict = pd.get_dummies(data=X_Predict, columns=ele)

    return X_Train, y_Train, X_Predict, y_Predict

# 빠른 결과값을 위해 일부로만 샘플진행
train_num = 300
test_num = 50


def small_sample(train_num, test_num, Train, Predict):
    # 빠른 결과값을 위해 일부로만 샘플진행
    Train = Train.sample(n=train_num) # 300개
    Predict = Predict.sample(n=test_num) # 50개

    return Train, Predict

def c_columns(df):
    cols = []
    cols.append('모델')
    for i in range(len(df.columns)-1):
        col = 'c'+str(i)
        cols.append(col)
    return cols

def get_model_var(df,Model_ver):  #모델버전 생성 
    df = df.reset_index(drop=False)
    Model_ver_list =  [Model_ver] * len(df)
    Model_ver_list = pd.DataFrame(Model_ver_list, columns =['모델 버전'])
    
    updated_df = pd.concat([Model_ver_list,df] ,axis=1)
    return updated_df


# In[4]:


# 모델 저장
def outputfile(sheet1,output_file_name):     
    sheet1.to_csv(output_file_name, encoding ='utf-8-sig')
    print("\n폴더에서",output_file_name,"파일을 확인하세요")

# 회귀 모델(선형, 로지스틱) 예측값 및 모델 파일로 출력
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
    
    prediction = clf_from_joblib.predict(X_Predict_df)
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


# In[5]:


# 회귀 모델(Auto_ML, 랜덤포레스트) 예측값 및 모델 파일로 출력
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
    
    results_pred_df= get_model_var(prediction_df, filename)    
    
    outputfile(Model_ver,output_file_name1)
    outputfile(results_pred_df,output_file_name2)
        
    return Model_ver, results_pred_df


# In[6]:


# 신경망 분석 모델 결과 출력
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
    
    results_pred_df= get_model_var(prediction_df, filename)    
    
    outputfile(Model_ver,output_file_name1)
    outputfile(results_pred_df,output_file_name2)
        
    return Model_ver, results_pred_df


# ## 모델 선택 및 실행

# In[7]:


# model select
def main():
    
    # 데이터 분리
    dummy_list, x_val, y_val, predic_period, Train, Predict, model_name = read_data_info (read_data_file, read_col_info_file, read_model_info_file)
    
    # 샘플뽑아 진행 (생략가능)
    Train, Predict = small_sample(train_num, test_num, Train, Predict)
    
    # make dataset 
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = make_model_df (dummy_list, x_val, y_val, Train, Predict)


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


# In[9]:


# 선형
if __name__ == "__main__":  
    read_model_info_file = 'input_AutoML_설정옵션2.csv'
    output_file_name1 = 'output_선형_예측값.csv'
    output_file_name2 = 'output_선형_정보.csv'
    output_file_name3 = 'output_선형_결과.csv'
    main()


# In[10]:


# 로지스틱
if __name__ == "__main__":  
    read_model_info_file = 'input_AutoML_설정옵션4.csv'
    output_file_name1 = 'output_로지스틱_예측값.csv'
    output_file_name2 = 'output_로지스틱_정보.csv'
    output_file_name3 = 'output_로지스틱_결과.csv'
    main()


# In[11]:


# Auto_ML
if __name__ == "__main__":  
    read_model_info_file = 'input_AutoML_설정옵션.csv'
    output_file_name1 = 'output_AutoML_모델.csv'
    output_file_name2 = 'output_AutoML_예측결과.csv'
    main()


# In[12]:


# 신경망 
if __name__ == "__main__":  
    read_model_info_file = 'input_AutoML_설정옵션3.csv'
    output_file_name1 = 'output_신경망_모델.csv'
    output_file_name2 = 'output_신경망_예측값.csv'
    main()


# ## 파일에 저장한 모델 불러오기

# In[13]:


def load_model_keras(jsonfile, h5file, new_RD): # 신경망
    json_file = open(jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5file)
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    
    pred = loaded_model.predict(new_RD) # 예측값구하는 식
    pred_df = pd.DataFrame(data=pred)
    pred_df.columns = ['new_Predicted']
    pred_df.index = new_RD.index
    
    outputfile(pred_df,output_file_name)
    
    return pred_df


# In[14]:


def load_model_sklearn(filename, new_RD): # 모든 회귀모델 
    clf_from_joblib = joblib.load(filename) 
    pred = clf_from_joblib.predict(new_RD)
    pred_df = pd.DataFrame(data=pred)
    pred_df.columns = ['new_Predicted']
    pred_df.index = new_RD.index
  
    outputfile(pred_df,output_file_name)
    
    return pred_df


# ## 저장된 파일로 모델 실행

# In[15]:


def main2():
    dummy_list, x_val, y_val, predic_period, Train, Predict, model_name = read_data_info (read_data_file, read_col_info_file, read_model_info_file)


    # 샘플뽑아 진행 (생략가능)
    Train, Predict = small_sample(300, 50, Train, Predict) # random하게 data를 뽑아와 모델 재사용해봄
    # 샘플사이즈는 동일하지만, index는 임의적으로 지정되어 데이터 구성, 즉, 새로운 데이터를 만든 것

    # make dataset 
    X_Train_df, y_Train_df, X_Predict_df, y_Predict_df = make_model_df (dummy_list, x_val, y_val, Train, Predict)
    new_RD = X_Predict_df
    
    if model_name == 'logit' or model_name == 'OLS' or model_name == 'Auto_reg':
        load_model_sklearn (file, new_RD)
        
    elif model_name == 'Neural_net' or model_name == 'Random_fore':
        load_model_keras(jsonfile, h5file, new_RD)
        
    else: 
        print('Please select your data model')


# In[17]:


# 선형
if __name__ == "__main__":
    read_model_info_file = 'input_AutoML_설정옵션2.csv'
    file = '0x000001DAD4B9E8E0.pkl'
    output_file_name = 'output_선형_모델재사용_예측값.csv'
    main2()


# In[18]:


# 로지스틱
if __name__ == "__main__":
    read_model_info_file = 'input_AutoML_설정옵션4.csv'
    file = '0x000001DAC1BBF2E0.pkl'
    output_file_name = 'output_로지스틱_모델재사용_예측값.csv'
    main2()


# In[19]:


# Auto_ML
if __name__ == "__main__":
    read_model_info_file = 'input_AutoML_설정옵션.csv'
    file = '210324_145522_ExtraTreesRegressor.pkl'
    output_file_name = 'output_AutoML_모델재사용_예측값.csv'
    main2()


# In[20]:


# 신경망
if __name__ == "__main__":
    read_model_info_file = 'input_AutoML_설정옵션3.csv'
    jsonfile = '0x000001DAC49700A0.json' # 파일 이름 
    h5file = '0x000001DAC49700A0.h5'
    output_file_name = 'output_신경망_모델재사용_예측값.csv'
    main2()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




