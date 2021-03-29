# APS_alpha
 


# 목적
머신 러닝, 최적화 기준 정보, 수요, CAPA 예측 및 최적화 Platform 구현    
분석을 위한 데이터를 추출 가공하고, 예측 결과를 산출   
업무적 의사 결정에 필수적인 예측 수치와 근거를 제시 한다.   

# 데이터 정보

| PRJ_ID   |  DATA_TYPE | PART | MONTH_NO | QTY  |  FEATURE_1 |  FEATURE_2  | 	FEATURE_3  |	FEATURE_4 | 	FEATURE_5 |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
|  DAESANG  |  TD   | 1000180 | 15-Jan    | 710  | 상온     | 100110100 | B2B | X  | C |
|  DAESANG  |  RD   | 2026375 | 20-Mar     | 0   | 냉동     | 997680140 | B2B | X | A |


## Overview
### 1. 일반회귀식 & Auto_ML
### 2. 시계열 & Auto Arima & 시계열 딥러닝 (LSTM)


예상 Target Value
값(수치, 빈도) : value, stdev
제품 판매 예측 량, 예상 작업 시간, 필요 인원, 발생 횟수 etc.
확률(해당 속성 포함 확률) : %, stdev
설비 가능 여부, 공정 포함 여부 etc.

ATTRIBUTE WEIGHT
Reference 내 각 속성이 Target Value 결정에 미친 영향도 분석 결과

Reference 유사도 분석
예측 대상 데이터 기준 Reference와의 상관관계 분석(유사도, - 상관 관계 등)
유사한 판매 패턴을 보일 것으로 예상되는 제품
유사한 가격 흐름을 보일 것으로 예상되는 종목

model 저장 - data model
저장 모델 활용 예측 결과 산출
강화 학습 모델 적용


KSETEC 홈페이지: <http://kstec.co.kr/>
*Author: KSTEC 연구원 이선의, 성초연     
Last edited: 29-03-2021*
