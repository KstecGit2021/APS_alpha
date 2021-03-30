# 제품별 /수요 (트렌드) 예측 

대상 데이터를 활용해 아래와 같은 결과 값을 구현하였습니다.


**목적**
1. 각 아이템의 과거 데이터로 2020년도 QTY 예측
2. 각 아이템의 속성들에 대한 과거 데이터로 2020년도 QTY 예측

**시계열 모델정보**
- _Autom_Arima_
  - _Autom_Arima_ 처음 실행시   **!pip install pmdarima** 코드 실행 필요
- LSTM

**Input 정보**
- 데이터 셋 파일명: DAESANG_DATA_prepared.csv
- 설정 옵션 파일명: input_시계열모델_설정옵션.csv
- 분석 조건 값 파일명: input_시계열모델_조건설정값.csv
- 데이터 유형 파일명: input_시계열모델_데이터유형.csv

**Output 정보**
- 예측 결과
- 예측 모델 정보
- 예측 모델 값
- 변수 중요도

# 시나리오

### 1. 시계열 분석 (Auto_Arima 사용)
- 예측 모델의 입출력 값
  - Input(x)는 QTY의 과거 값
  - Output(y)는 QTY의 미래 예측 값 


### 2. 시계열 분석 딥러닝 (LSTM 사용)
- 예측 모델의 입출력 값
  - Input(x)는 QTY의 과거 값
  - Output(y)는 QTY의 미래 예측 값 
