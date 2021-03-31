#!/usr/bin/env python
# coding: utf-8

# # 1. 시계열 딥러닝 (LSTM: Long-short Term Memory)
# 
# 
# 최근 데이터 예측 방법으로 인공신경망(Artificial Neural Network, ANN)분야에 대한 관심이 높아졌으며, 
# 그 중 시계열 데이터 예측에 특화된 **LSTM(Long Short-Term Memory)모형**은 수문 시계열자료의 예측방법으로도 활용되고 있다.
# 
# http://elearning.kocw.net/contents4/document/lec/2013/Konkuk/Leegiseong/5.pdf
# 시계열 정보 참고
# 
#  - 1.1 단변량 (just y)
#  - 1.2 다변량 (related with X)
#  
#  
#  
# ※ 설정옵션 파일이름 변경으로 단변량, 다변량 테스트
# - input_AutoML_설정옵션5.csv *단별량 (LSTM)*
# - input_AutoML_설정옵션6.csv *다별량 (Multi_Lstm)*

# In[1]:


# Import libary 
import csv
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt

import keras
import ast
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional, Activation, Dropout
from keras import backend as K
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import model_from_json
import re

pd.set_option('display.max_rows', 500) # 많은 데이터를 한눈에 볼 수 있도록


# In[2]:


# ------ 필요 함수

# LSTM을 위함/ 조건을 불러오는 방법, 일반화하기 위함
def get_variables(condition_file_name):
    condition_df = pd.read_csv(condition_file_name,encoding='cp949')
    keys = condition_df['condition_col_name'].tolist()
    values = condition_df['condition'].tolist()
    
    condition_list = get_conditions(keys,values)
    
    return condition_list

def get_conditions(keys,values):
    """
    함수설명
    --------
    조건 컬럼명과 조건 값을 Disctionary 형태로 변경
    > 조건을 한번에 필터 하기 위한 데이터 변경 
    
    parameters
    ----------
    keys(조건 컬럼명)
    values(조건)
    
    return
    ------
    conditions: Dictionary 형태로 조건 컬럼명과 조건 
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


def filter_df(df, filter_values):
    """
    함수설명
    --------
    Filter df by matching targets for multiple columns.
    
    parameters
    ----------
    df: 데이터 셋
    filter_values: Dictionary 형태의 조건 리스트: condition_list (조건 설정 파일에서 불러옴)
    
    return
    ------
    df 조건에 따라 필터된 데이터셋 
    """
        
    if filter_values is None or not filter_values:
        return df
    return df[
        np.logical_and.reduce([
            df[column].isin(target_values) 
            for column, target_values in filter_values.items()
        ])
    ]

def read_data_info (read_data_file, read_col_info_file, read_model_info_file, condition = None): 
    """
    함수설명
    --------
    데이터 파일 읽기, 모델링에 필요한 데이터 유형, 역할, 예측 주기 파라미터 
    값 사용자 지정 파일에서 불러오기 기능 
    
    parameters
    ----------
    read_data_file: 데이터 셋 파일명
    read_col_info_file: 데이터 정보 파일명
    read_model_info_file: 사용될 모델 지정 파일명
    condition = None: 조건값 자체를 사용안될 경우 고려  
        => read_data_info (read_data_file, read_col_info_file, read_model_info_file) 도 작동
    
    return
    ------
    features, 더미화 된 x변수의 columns 이름
    x_val,  사용자가 지정한 x변수
    y_val, predic_period,  사용자가 지정한 y변수(타겟)
    Train,  학습용 데이터
    Predict,  예측용 데이터
    model_name,  사용할 모델
    train_p,  학습용 데이터의 날짜의 총갯수
    test_p  예측용 데이터의 날짜의 총갯수
    
    """
    
    
    # input data
    data = pd.read_csv(read_data_file, index_col=0) #이름이 지정되지 않은 첫 컬럼은 새로 데이터를 만들면서 index가 자동으로 생성되었기에 제거
    data.fillna(0, inplace=True) # RD에 있는 NAN 데이터를 0으로 바꿈 => 데이터 유형을 정수로 표현하기 위해 (데이터 셋 분리과정)

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

    # 데이터 재구성 : 다변량에서도 잘 작동하는지 확인하기 위해 임의작업 - random한 값
    X_data = data[x_val]
    ele = [x for x in dummy_list if x in X_data]
    dummied_X = pd.get_dummies(data=X_data, columns = ele)
    features = dummied_X.columns
    dummied_df = pd.get_dummies(data=data, columns = ele)
    new_data = np.random.random( size=(len(dummied_df),len(features)))
    new_df = pd.DataFrame(new_data, columns =features , index = dummied_df.index)
    new_generated_df = dummied_df.drop(features,axis=1)
    new_generated_df = pd.concat([new_generated_df,new_df], axis=1)

    # 시계열 형식으로 변환 (datetime)
    new_generated_df[predic_period] = pd.to_datetime(new_generated_df[predic_period], format= '%y-%b') # predic_period를 datetime 으로 변환

    # 조건에 해당하는 set으로 재구성
    new_generated_df = filter_df(new_generated_df, condition) # 필터를 이용해 조건에 해당하는 값으로 df 재구성 # 조건이 없다면 condition없이 작성해도 무방

    # data split
    Train = new_generated_df[new_generated_df['DATA_TYPE']=='TD'] # 학습할 데이터
    Predict = new_generated_df[new_generated_df['DATA_TYPE']=='RD'] # 예측해야할 데이터 

    # 시계열 데이터로 구성하기 위해 필요한 단계 : 기존의 달 수, 예측할 달 수 
    train_p = len(Train[predic_period].unique()) # train의 날짜의 총갯수
    test_p = len(Predict[predic_period].unique()) # predict의 날짜의 총갯수
    
    return features, x_val, y_val, predic_period, new_generated_df, model_name, train_p, test_p


# In[3]:


# ----- LSTM을 위한 전처리 & lag 생성
def make_model_df (df,features, y_val, predic_period, train_p, test_p, model_name):
    """
    함수설명
    --------
    LSTM을 위한 전처리 & lag 생성
    
    parameters
    ----------
    df: Input 데이터 셋 
    features : 더미화된 컬럼이름
    y_val: y 컬럼 이름 
    predic_period: 예측 주기 (12= 1년)
    train_p: train의 날짜의 총 갯수
    test_p: predict의 날짜의 총 갯수
    model_name: 단별량, 다변량 구별
    
    return
    ------
    grouped,  조건에 의해 새로 구성한 데이터셋 - 각 월별 평균으로 생성
    x_train,  학습용 데이터 x변수
    y_train,  학습용 데이터 y변수
    x_test,   예측용 데이터 x변수
    y_test,   예측용 데이터 y변수
    scaler: 표준화 된 데이터 값 (Arrary)
  
    """
        
    target = [y_val]
    Nec = [predic_period, y_val]
    features_list = features.tolist()
    
    grouped = df.groupby(predic_period, as_index=False).mean()  # 시간을 기준으로 part들의 qty의 평균으로 데이터 그룹을 만드는 작업
    
    if model_name=='multi_LSTM':
        Nec.extend(features_list)
        grouped = grouped[Nec] # 필요한 컬럼은 시간과 qty + 변수들
    
    elif model_name=='LSTM':
        grouped = grouped[Nec] # 필요한 컬럼은 시간과 qty뿐.
        
    scaler = MinMaxScaler() # MinMaxScaler: 0과 1사이의 값으로 재구성 => 빠른 계산을 위함
    grouped[target] = scaler.fit_transform(grouped[target]) # 위의 scaler방법으로 scaler 시킴
    
    if model_name=='multi_LSTM':
        target.extend(features_list)
        y_hat = grouped[target]
    
    elif model_name=='LSTM':
        y_hat = grouped[target] 
        
    y_hat = np.array(y_hat) 
    x,y = generateX(y_hat,test_p) # lag생성 
    # x= test_p만큼 짤라서 한칸씩 미뤄서 구성, y= y_hat에서 test_p만큼 삭제된 값
    # 위에 구성된 x, y를 재구성
    x = x.reshape(-1,test_p,y_hat.shape[1]) ; y = y.reshape(-1,1) # lstm에 넣기 위한 3차원으로 변환 작업
    
    # 위에서 구성된 x,y를 test_p의 갯수를 이용해 split 
    x_train = x[:(train_p-test_p),:,:] ; y_train = y[:(train_p-test_p),:]
    x_test = x[(train_p-test_p):,:,:] ; y_test = y[(train_p-test_p):,:]
    
    return grouped, x_train, y_train, x_test, y_test, scaler

# make model => lag 생성
def generateX(a, n):
    """
    함수설명
    --------
    시계열에 쓰이는 lag 생성
    : 시계열 자료를 분석할때 관측시점들 간의 시차(time lag) 생성
    
    parameters
    ----------
    a: array 형태
    n: window 사이즈 
    
    return
    ------
    np.array(x_train):  
    np.array(y_train):  lstm 모델에 넣기 위해 array형태로 변환하는 과정
    """
    x_train = []
    y_train = []
    
    for i in range(len(a)):
        x=a[i:(i+n)]
        if(i+n) < len(a):
            x_train.append(x)
            y_train.append(a[i+n])
            
        else:
            break
    return np.array(x_train), np.array(y_train)


# In[4]:


# ----- 모델 결과 저장 및 엑셀파일로 내보내기 
# 모델 저장
def outputfile(sheet1,output_file_name):     
    sheet1.to_csv(output_file_name, encoding ='utf-8-sig')
    print("\n폴더에서",output_file_name,"파일을 확인하세요")
    
# 결과 값 도출
def get_LSTM_results (model, grouped, y_val, y_train, x_test, scaler, train_p, test_p):
    """
    함수설명
    --------
    1. 모델 정보 파일로 내보내기
    2. 시계열 그래프 
    3. 결과값 Dataframe에 저장 및 엑셀로 내보내기 
    
    parameters
    ----------
    model      : 모델정보
    grouped    : 조건에 의해 만들어진 본래의 데이터셋
    y_val      : 타겟변수 - 예측하고 싶은 것
    y_train    : 기존 시간에 대한 데이터의 추세 - 그림으로 확인하기 위함
    x_test     : 예측값 계산을 위해 만든 예측용 데이터 x
    scaler     : 표준화 된 데이터 값 _ 다시 되돌려 본래의 값으로 출력하기 위해 불러옴
    train_p    : train의 날짜의 총 갯수
    test_p     : predict의 날짜의 총 갯수
    
    return
    ------
    Model_ver: 모델 버젼 (Dataframe 형태)
    grouped: 과거 y 데이터와 예측값 y (Dataframe 형태)
    """
    
    Model_ver = pd.DataFrame([str(model)], columns=['모델정보'])
    filename = re.sub('[<>.]','',str(model).split()[3]) # 생성된 모델정보의 숫자,문자 형태를 파일 이름으로 지정
    model_file1 = "{model}.json".format(model=filename) 
    model_file2 = "{model}.h5".format(model=filename)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file1, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_file2)

    prediction = model.predict(x_test, batch_size=1) # 예측값
    
    # 결과 plot line graph 생성 
    a_axis = np.arange(0,len(y_train))
    b_axis = np.arange(len(y_train), len(y_train) + len(prediction))
    plt.figure(figsize=(10,6))
    plt.plot(a_axis, y_train.reshape((train_p-test_p),), 'o-')
    plt.plot(b_axis, prediction.reshape(test_p,), 'o-', color='red', label='Predicted')
    #plt.plot(b_axis, y_test1.reshape(5,), 'o-', color='green', alpha=0.2, label='Actual') # 과거데이터를 나눠서 학습할때 테스트용 
    plt.legend()
    plt.show()
    
    # pred나온 값을 원래의 데이터에 다시 배치
    grouped[y_val][(-test_p):] = sum(prediction.tolist(), [])
    # scaler된 값 복구
    grouped[[y_val]] = scaler.inverse_transform(grouped[[y_val]]) 
    
    outputfile(Model_ver, output_file_name1)
    outputfile(grouped, output_file_name2)
        
    return Model_ver, grouped


# ----- 모델을 파일로 내보내기 및 저장 
def load_model_lstm(jsonfile, h5file, new_RD, grouped, y_val, test_p, scaler):
    """
    함수설명
    ________
    json, h5 파일 형태로 저장된 시계열 모델 불러오기 
    
    
    Parameters
    __________
    jsonfile, 
    h5file,
    new_RD,    새로 예측할 데이터
    grouped,   새로 예측할 데이터를 조건에 의한 데이터셋으로 구성
    y_val,     타겟변수
    test_p     예측할 월의 갯수
    
    
    return
    ______
    grouped   조건에 의해 구성한 데이터셋에 예측값을 넣어 한번에 출력
        
    """
    
    json_file = open(jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5file)
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    
    pred = loaded_model.predict(new_RD, batch_size=1) # 예측값구하는 식
    
    grouped[y_val][(-test_p):] = sum(pred.tolist(), [])
    # scaler된 값을 다시 원래의 값으로 되돌림
    grouped[[y_val]] = scaler.inverse_transform(grouped[[y_val]])

    outputfile(grouped, output_file_name)
    
    return grouped


# In[5]:


# Read data & data preparation
read_data_file = 'daesang.csv' # 총 TD:60, RD:5개의 시간이 있는 데이터. 즉, 결측값이 없는 데이터로 재구성
read_col_info_file = 'input_LSTM_데이터유형.csv' # 시계열로 변환이 필요한 예측주기는 일반회귀에선 사용 안함.
read_condition_file = 'input_LSTM_조건설정값3.csv' # part별


# In[6]:


# model select
def main():
       
    condition = get_variables(read_condition_file) # 조건값
    features, x_val, y_val, predic_period, new_generated_df, model_name, train_p, test_p = read_data_info (read_data_file, read_col_info_file, read_model_info_file, condition)
    
    grouped, x_train, y_train, x_test, y_test, scaler = make_model_df (new_generated_df, features, y_val, predic_period, train_p, test_p, model_name)
  
    if model_name == 'LSTM' or model_name == 'multi_LSTM':
        # model
        xInput = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))  # input
        xLstm_1 = LSTM(10, return_sequences = True)(xInput)  # lstm 모델 기법
        xLstm_2 = Bidirectional(LSTM(10))(xLstm_1)   #  정확도 향상을 위한 양방향 rnn사용
        xOutput = Dense(1)(xLstm_2) # output

        model = Model(xInput, xOutput) 
        model.compile(loss='mse', optimizer='adam')

        # loss값 계산하며 파일 저장
        basename = "model.h5" # 중간중간 저장함으로 추후 코드끊겼을 때 유용
        suffix = pd.datetime.now().strftime("%y%m%d_%H%M%S") # 파일이 돌아가기 시작한 시간을 기준으로 이름 생성
        path_checkpoint = "_".join([suffix, basename])
        es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=1000) #1000번동안 loss가 변함없다면 stop

        # callback 함수 : #1000번동안 loss가 변함없다면 stop 모델 체크하면서
        modelckpt_callback = keras.callbacks.ModelCheckpoint(
            monitor="loss",
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            mode='auto'
        )

        model.fit(
            x_train, y_train,
            epochs=100000, batch_size=20,
            callbacks=[es_callback, modelckpt_callback],
        )

        get_LSTM_results (model, grouped, y_val, y_train, x_test, scaler, train_p, test_p)
        
    else: 
        print('Please select your data model')


# In[7]:


# PART 별 시계열 예측 
if __name__ == "__main__":  
    read_model_info_file = 'input_AutoML_설정옵션5.csv' # 단변량
    output_file_name1 = 'output_단변량_모델.csv'
    output_file_name2 = 'output_단변량_예측값.csv'
    main()


# In[8]:


# PART 별 시계열 예측 다변량 (x와의 상관성 의미를 둠)
if __name__ == "__main__":
    read_model_info_file = 'input_AutoML_설정옵션6.csv' # 다변량
    output_file_name1 = 'output_다변량_모델.csv'
    output_file_name2 = 'output_다변량_예측값.csv'
    main()


# ## json 파일에 저장된 모델 불러오기 Test
# - 생성된 모델을 재사용
# - load_model_lstm() 함수사용

# In[9]:


def main2():
    condition = get_variables(read_condition_file) # 조건값
    features, x_val, y_val, predic_period, new_generated_df, model_name, train_p, test_p = read_data_info (read_data_file, read_col_info_file, read_model_info_file, condition)
    grouped, x_train, y_train, x_test, y_test, scaler = make_model_df (new_generated_df, features, y_val, predic_period, train_p, test_p, model_name)
    new_RD = x_test
    
    load_model_lstm(jsonfile, h5file, new_RD, grouped, y_val, test_p, scaler)


# In[12]:


# 단변량
if __name__ == "__main__":
    jsonfile = '0x000001323F430160.json'
    h5file = '0x000001323F430160.h5'
    read_model_info_file = 'input_AutoML_설정옵션5.csv' # 다변량
    output_file_name = 'output_모델재사용_단변량_예측값.csv'
    main2()


# In[13]:


# 다변량
if __name__ == "__main__":
    jsonfile = '0x00000132817D5E20.json'
    h5file = '0x00000132817D5E20.h5'
    read_model_info_file = 'input_AutoML_설정옵션6.csv' # 다변량
    output_file_name = 'output_모델재사용_단변량_예측값.csv'
    main2()


# In[ ]:




