# 회귀분석모델 및 유사도 산출

대상 데이터를 활용해 아래와 같은 결과 값을 구현하였습니다.

## 목적
1. 각 아이템의 과거 데이터로 2020년도 QTY 예측
2. 신규 아이템에 대한 증감 예측
3. 계절성이 있는 월간 예측

회귀모델 (Auto_ML, 선형(OLS), 이항선형회귀(Logistic))
아이템 별 월평균 데이터 셋 생성

## 모델 종류

- OLS: 선형회귀
- MNLogit: 이항회귀
- RandomForestRegressor
- GoRegress: Auto_ML
- GoClassify: Auto_ML
- Neural_net: 신경망,Neural Network (분류)

## 예측 모델의 입출력 값
- Input(x)은 신규 아이템에 대한 속성(이전에 있던 속성 값)
- Output(y)는 기존 아이템에 대한 평균 QTY 예측값

## 모델을 파일로 저장
- keras(케라스)를 쓰는 경우는 json / h5 파일로 모델 저장
- sklearn를 쓰는 경우는 pickle 파일로 모델 저장

# 시나리오

### 1. 회귀분석 모델
- Auto_ML
- 일반선형
- 로지스틱
- 신경망(딥러닝)

### 1.1 월간예측 (차분 & Auto_ML 사용)
- 계절성 확인 목적

### 2. 유사도 산출
- 거리계산 공식 사용한 유사도 산출 


*Author: KSTEC 연구원 성초연*   
*Last edited: 29-03-2021*
