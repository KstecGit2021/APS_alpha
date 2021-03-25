#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
fm.get_fontconfig_fonts()

read_data_file = 'new_DAESANG_DATA.csv'
read_col_info_file = 'input_LSTM_데이터유형.csv' # 시계열로 변환이 필요한 예측주기는 클러스터에서 사용하지 않지만, 필요시 사용 가능.
read_model_info_file = 'input_AutoML_설정옵션.csv'


# ## 시나리오 4. 유사제품 예측(클러스터링)
# 
# - 제품 유사도 산출
# - input  : 전체데이터셋 (TD/RD 구별 없음)
# - output : abs_sum은 0 ~ 1까지 (소수단위)
# 
#   *0으로 갈 수록 제품들의 속성이 유사, 1과 가까울 수록 제품들의 속성이 다름*
#   
#   **=> 0이면 제품의 속성이 전부 일치**

# In[2]:


def read_data_info (read_data_file, read_col_info_file, read_model_info_file):
    
    # input data
    data = pd.read_csv(read_data_file) # 년도와 월을 split해서 new data 생성

    # input role info
    role_info = pd.read_csv(read_col_info_file, encoding ='cp949') # 모델링 할 때 사용할 x, y, month
    x_val = role_info.loc[role_info['Role']=='x', 'col_name'].tolist() # x변수 다중리스트형태
    y_val = role_info.loc[role_info['Role']=='y', 'col_name'].tolist()[0] # y변수는 단일
    predic_period = role_info.loc[role_info['예측주기']=='P', 'col_name'].tolist()[0] # month값은 단일
    
    # col info
    dummy_list = role_info.loc[role_info['col_info']=='STR', 'col_name'].tolist() 
    id_val = role_info.loc[role_info['col_info']=='STR_KEY', 'col_name'].tolist()[0] # key는 자동으로 id 인식
    
    # input model info 모델과 예측달 정하기
    model_info = pd.read_csv(read_model_info_file)
    id_name = model_info['예측id'][0]

    return data, dummy_list, x_val, y_val, id_val, predic_period, id_name


# In[3]:


def make_datasample (data, dummy_list, x_val, y_val, id_val, id_name):
    
    new_data = data[1:] ; X = new_data[x_val] ; y_df = new_data[[y_val]].fillna(0).astype('int32') # 전처리 
    # col속성을 제외한 데이터의 값으로만 X,y 값 나눔 필요
    
    # 더미화 할 col정보
    ele = [x for x in dummy_list if x in X] # 더미화가 필요한 col중에 X에 들어가지 않는 것이 있을 수 있으므로 진행
    new_df = new_data.drop_duplicates([id_val]) # 유일한 PART의 값들 나열
    new_df = new_df.reset_index() # 재 index
    new_df = new_df.drop('index', axis=1) # index 드롭
    new_df2 = pd.get_dummies(data=new_df, columns=ele) # 더미화
    
    X_df = pd.get_dummies(data=X, columns=ele) # X의 val전체 더미화
    
    # line이 될 정보 따로 저장
    dummies_col_list = X_df.columns # 더미화 된 컬럼들의 이름
    id_list = new_data[id_val].drop_duplicates() # id의 값들 리스트
    
    test = new_df2[new_df2[id_val] == (id_name) ] # id_list 참고해서 샘플링
    
    return X_df, y_df, new_df2, test, dummies_col_list,id_list


# ## 랜덤포레스트를 이용한 중요도 및 거리 계산

# In[4]:


def make_distance_df (test, new_df2, X_df ,y_df, dummies_col_list, id_list, id_val):
    
    """
    함수설명
    --------
    1. 거리 계산을 통한 유사도 제품 찾기
    2. 랜덤포레스트 회귀모델로 중요도 고려

    parameters
    ----------
    test      : 예측하고자하는 키
    new_df2    : 키 별로 재 구성한 데이터셋
    X_df      : 사용자가 지정한 x변수
    y_df    : 사용자가 지정한 y변수
    dummies_col_list     : 더미화 된 컬럼들의 이름
    id_list     : 키의 리스트 
    id_val    : id. 우리가 사용하는 키 값
    
    return
    ------
    new_df2 : 거리와 중요도 까지 고려되어 새로운 컬럼을 생성하여 재구성된 데이터셋
    """
    
    # 피쳐 중요도 계산
    
    # Feature importance
    # 랜덤포레스트회귀 모형을 활용하여, 라벨들간의 Gini Importance 의 값들을 가지는 
    # Feature Importance array 를 추출 
    # model select

    RFR = RandomForestRegressor(n_jobs=-1, random_state=1)
    RFR.fit(X_df ,y_df )
    feature_im = RFR.feature_importances_
    
    
    # 거리 계산
    # test id = 2026376 의 X 값과 , X_Train 의 X 값의 거리 계산 
    
    '''
    Distance 계산공식 
    sigma{abs(test_Xn - train_Xn)}
    '''
    
    blank_two_dimension = []
    for id in id_list:
        train001 = new_df2[new_df2[id_val] == id]

        blank = []
        for i in dummies_col_list:
            t1 = test[str(i)].values
            t1 = int(t1)

            v1 = train001[str(i)].values
            v1 = int(v1)

            ss = abs(t1-v1)
            blank.append(ss)
          
        # 중요도와 거리를 고려해 weighted 값
        weigthed_blank = [(i*j) for i,j in zip(blank,feature_im)]
        
        blank_result = sum(weigthed_blank)
        blank_two_dimension.append(blank_result)
    
    # 계산된 값 새로운 컬럼으로 저장
    new_df2['abs_sum'] = blank_two_dimension
    new_df2.sort_values('abs_sum') # 정렬
    
    return new_df2


# In[5]:


# 모델 저장
def outputfile(sheet1,output_file_name):     
    sheet1.to_csv(output_file_name, encoding ='utf-8-sig')
    print("\n폴더에서",output_file_name,"파일을 확인하세요")


# ##  특성이 일치하는 PART출력

# In[6]:


# model select
def main():
    
    data, dummy_list, x_val, y_val, id_val, predic_period, id_name = read_data_info (read_data_file, read_col_info_file, read_model_info_file)
    X_df, y_df, new_df2, test, dummies_col_list,id_list = make_datasample (data, dummy_list, x_val, y_val, id_val, id_name)
    
    clust = make_distance_df(test, new_df2, X_df ,y_df, dummies_col_list, id_list, id_val) # 중요도에 따른 거리분포 작성
    
    same_feature = clust['abs_sum'] == clust['abs_sum'].min() # 값 일치하는 것 출력 : 딱 일치하는 것이 없을 가능성 고려 최솟값으로 출력
    print(clust[same_feature]) # 결과
    outputfile(clust[same_feature], output_file_name) # 파일 저장


# In[7]:


if __name__ == "__main__":  
    output_file_name = 'output_유사도.csv'
    main()


# In[ ]:





# In[ ]:




