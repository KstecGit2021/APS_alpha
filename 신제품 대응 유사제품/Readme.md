# 신 제품 대응 유사제품
1.	생산량 예측 
2.	생산량 (Capa) 증감 예측
3.	신 제품과 유사한 기존 제품과의 유사도 분석   

**input 정보**
- 데이터 파일 : new_DAESANG_DATA.csv
- 데이터 유형 파일 : input_LSTM_데이터유형.csv

※ 데이터 모델 구분

- _Auto_ML_ 설정 파일 : input_AutoML_설정옵션.csv
- _OLS_ 설정 파일 : input_AutoML_설정옵션2.csv
- _Neural_net_ 설정 파일 : input_AutoML_설정옵션3.csv
- _logit_ 설정 파일 : input_AutoML_설정옵션4.csv

**output 정보**
- 모델 정보 파일 : output_model_모델 (*model: 사용된 모델 정보*)
- 모델 예측결과 파일 : output_model_예측값

**load 정보**
- 저장된 모델에 따른 새로운 예측결과 파일 : load_model_예측값
