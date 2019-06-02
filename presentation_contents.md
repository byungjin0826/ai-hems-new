# 전체적인 모델 구성
- labeling model (30분 간의 전력 흐름) -> 가전기기 동작 상태
- Usage_daily_prediction_model (일별 전력량) -> 예측하고자 하는 기간의 일별 전력량
- AI schedule (요일, 기존 데이터의 누적) -> schedule
- DR schedule (요일, 시간, 가전기기별 사용 또는 대기 상태의 전력, AI schedule) -> schedule

# 각 model별 estimator, parmeter
- labeling model: RF, 
- Usage_daily_prediction_model:

# 모델 평가


# REST API
