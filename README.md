# APS_alpha
 


## 목적
머신 러닝, 최적화 기준 정보, 수요, CAPA 예측 및 최적화 Platform 구현 프로젝트    
분석을 위한 데이터를 추출 가공하고, 예측 결과 산출 및 업무적 의사 결정에 필수적인 예측 수치와 근거를 제시 한다.   

## 데이터 정보

| PRJ_ID   |  DATA_TYPE | PART | MONTH_NO | QTY  |  FEATURE_1 |  FEATURE_2  | 	FEATURE_3  |	FEATURE_4 | 	FEATURE_5 |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
|  DAESANG  |  TD   | 1000180 | 15-Jan    | 710  | 상온     | 100110100 | B2B | X  | C |
|  DAESANG  |  RD   | 2026375 | 20-Mar     | 0   | 냉동     | 997680140 | B2B | X | A |


## Overview

### 1. 시계열_구현 (Auto Arima) 및 차분 전처리 & 선형 회귀
 - 시계열 데이터 패턴에 맞는 모델 자동 추천 (Auto Arima)
 - 증감 CAPA 예측 
 - Reference 유사도 분석 
 - ATTRIBUTE WEIGHT  (Reference 내 각 속성이 Target Value 결정에 미친 영향도 분석 결과)
### 2. 시계열 딥러닝 (LSTM)
 - 단별량 시계열
 - 다별량 시계열
 - 개발 된 모델 파일로 저장 
### 3. 회귀분석모델 및 유사도산출
 - 데이터 속성에 맞는 모델 자동 추천 (Auto_ML)
 - 


KSETEC 홈페이지: <http://kstec.co.kr/>   
   
      
     
     

Author: KSTEC 연구원 이선의, 성초연     
Last edited: 29-03-2021
