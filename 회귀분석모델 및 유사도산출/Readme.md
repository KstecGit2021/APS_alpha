# 회귀분석모델 및 유사도 산출

대상 데이터를 활용해 아래와 같은 결과 값을 구현하였습니다.

**목적**
1. 각 아이템의 과거 데이터로 2020년도 QTY 예측
2. 변수의 유사성 분석

**Auto_ML 모델정보**
- _Auto_ML_
- _Auto_ML_ 처음 실행시   **!pip install auto_modeling** 코드 실행 필요

**Input 정보**
- 데이터 셋 파일명: new_DAESANG_DATA.csv
- 데이터 유형 파일명: input_LSTM_데이터유형.csv

※ 설정 옵션 파일이름 변경으로 모델 테스트
- 설정 옵션 파일명: input_AutoML_설정옵션.csv *Auto_ML*
- 설정 옵션 파일명 : input_AutoML_설정옵션2.csv *OLS*
- 설정 옵션 파일명 : input_AutoML_설정옵션3.csv *Neural_net*
- 설정 옵션 파일명 : input_AutoML_설정옵션4.csv *logit*

**Output 정보**
- 예측 결과
- 예측 모델 정보
- 예측 모델 값
- 변수의 유사도 (클러스터링)

**전체 모델 종류**

- OLS: 선형회귀
- MNLogit: 이항선형회귀
- RandomForestRegressor
- GoRegress: Auto_ML (선형)
- GoClassify: Auto_ML (분류)
- Neural_net: 신경망, Neural Network (분류)

**모델을 파일로 저장**
- keras(케라스)를 쓰는 경우는 json / h5 파일로 모델 저장
- sklearn를 쓰는 경우는 pickle 파일로 모델 저장

# 시나리오

### 1. 회귀분석 모델
- 아이템 별 월평균 데이터 셋 생성

**모델 종류**
- Auto_ML
- 일반선형
- 로지스틱
- 신경망(딥러닝)

**예측 모델의 입출력 값**
- Input(x)은 신규 아이템에 대한 속성 (이전에 있던 속성 값)
- Output(y)는 기존 아이템에 대한 평균 QTY 예측값

### 2. 유사제품 예측 (클러스터링)
- 거리계산 공식 사용한 유사도 산출 
- 랜덤포레스트를 이용한 중요도 계산

**예측 모델의 입출력 값**
- input : 각 PART별 속성으로 구성된 전체데이터셋 (TD/RD _구별 없음_ )
- output : abs_sum은 0 ~ 1까지 (소수단위)

0으로 갈 수록 제품들의 속성이 유사, 1과 가까울 수록 제품들의 속성이 다름

=> 0이면 제품의 속성이 전부 일치

*Author: KSTEC 연구원 성초연*   
*Last edited: 30-03-2021*
