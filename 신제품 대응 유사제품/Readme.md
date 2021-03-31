# 신 제품 대응 유사제품

**목적**

1.	생산량 예측 
2.	생산량 (Capa) 증감 예측

  2.1. 월간 수요 
3.	월평균 예측 (선형분석)
4.	신 제품과 유사한 기존 제품과의 유사도 분석   

**모델 종류**

- OLS: 선형회귀
- MNLogit: 이항회귀
- RandomForestRegressor
- GoRegress: Auto_ML (선형)
- GoClassify: Auto_ML (분류)
- Neural_net: 신경망,Neural Network (분류)

**모델을 파일로 저장**

- keras(케라스)를 쓰는 경우는 json / h5 파일로 모델 저장
- sklearn를 쓰는 경우는 pickle 파일로 모델 저장

# 시나리오 1

**예측 모델의 입출력 값**

- Input(x)은 신규 아이템에 대한 속성(이전에 있던 속성 값)
- Output(y)는 기존 아이템에 대한 QTY 예측값

**input 정보**

- 데이터 파일 : new_DAESANG_DATA.csv
- 데이터 유형 파일 : input_LSTM_데이터유형.csv

※ 데이터 모델 구분

- _Auto_ML_ 설정 파일 : input_AutoML_설정옵션.csv
- _OLS_ 설정 파일 : input_AutoML_설정옵션2.csv
- _Neural_net_ 설정 파일 : input_AutoML_설정옵션3.csv
- _logit_ 설정 파일 : input_AutoML_설정옵션4.csv

**output 정보**

- 예측 모델 관련 정보
- 예측 결과

**load 정보**

- 저장된 모델에 따른 새로운 예측결과 파일 : load_model_예측값

# 시나리오 2

**Input 정보**

- 데이터 셋 파일명: new_DAESANG_DATA.csv
- 설정 옵션 파일명: input_AutoML_설정옵션.csv
- 데이터 유형 파일명: input_AutoML_데이터유형.csv

**모델정보**

- auto_modelling
  - auto_modelling 처음 실행시 !pip install auto_modelling 코드 실행 필요

**코드설명**

- 월별 RD 예측 (ID별예측, feature 조건별 예측)

**Output 정보**

- 예측 모델 관련 정보
- 예측 결과

# 시나리오 3

**코드설명**

- 제품 유사도 산출

**예측 모델의 입출력 값**

- Input(x)은 전체데이터셋 (TD/RD 구별 없음)
- Output(y)는 abs_sum은 0 ~ 1까지 (소수단위)

  - 0으로 갈 수록 제품들의 속성이 유사, 1과 가까울 수록 제품들의 속성이 다름

▶ 0이면 제품의 속성이 전부 일치

**input 정보**

- 데이터 파일 : new_DAESANG_DATA.csv
- 데이터 유형 파일 : input_LSTM_데이터유형.csv
- 예측 Key 설정 파일 : input_AutoML_설정옵션.csv


**output 정보**

- 유사속성 결과
