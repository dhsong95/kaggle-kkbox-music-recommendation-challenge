**kaggle-kkbox-music-recommendation-challenge**
==============================================

Kaggle Competition [WSDM - KKBox's Music Recommendation Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge) 에서 제공하는 데이터 세트를 활용한 EDA 및 추천 과제 해결.

---

## 1. EDA
데이터 세트에 대한 EDA 및 시각화. [링크](report/Exploratory%20Data%20Analysis.md)

* 사용자와 노래에 대한 데이터를 분석한다.
* 사용자와 노래에 대한 데이터가 예측값(target)에 어떠한 영향을 주는지 확인한다
* 분석 결과를 Barplot, Boxplot, KDEplot을 통해 시각화한다.

---

## 2. Model - v1
기본적인 Matrix Factorization 기반의 데이터 모델링. [링크](report/Model%20v1.md)

* 테스트 데이터에 대해서 정활률 0.54589
* Sample Submission 0.5000 보다 높지만 KKbox Benchmark 인 0.61337 보단 낮은 결과
* 성능 개선 방법
  1. 현재 모델의 하이퍼파라미터(반복 횟수, Feature 개수) 조정. - Kaggle Kernel에서 수행 권장. [링크](https://www.kaggle.com/dhsong13/model-v1-mf-hyper-parameter-adjustment)
  2. K-NN 기반의 방법 + IDF 가중치를 활용한 방법