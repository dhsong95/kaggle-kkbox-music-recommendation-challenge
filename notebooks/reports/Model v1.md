Model version 1: Basic Matrix Factorization
===========================================

## **1. Train, Validation, and Test**

Kaggle에서 제공하는 학습을 위한 데이터 세트로는 train.csv와 test.csv가 있다. train.csv 로 모델을 학습하고 test.csv에 있는 데이터를 가지고 target 값을 예측한다.

train.csv에는 7,377,418 개의 레코드가 있고, test.csv에는 2,556,790 개의 레코드가 있다. 

학습한 모델 평가를 위해서 train.csv를 학습용 데이터(train)와 검증용 데이터(validation)으로 나눈다. 9:1의 비율로 나눈 결과 학습용 레코드는 6,639,676 개이고, 검증용 레코드는 737,742 개이다.

## **2. Modeling: Matrix Factorization**

### **1. Matrix Factorization used in Collaboratice Filtering Recommender System**
Collaborative Filtering은 추천 시스템 구현에 있어서 자주 사용된다. Collaborative Filtering은 사용자의 아이템 이용 정보를 바탕으로 추천하는 방식이다. 

CF(Collaboratice Filtering) 방식은 사용자 기반의 CF와 아이템 기반의 CF로 구분한다. 사용자 기반 CF는 사용자와 유사한 이용 패턴(아이템 선호 패턴)을 보이는 사용자를 찾고, 유사한 사용자가 선호하는 아이템을 추천한다. 아이템 기반 CF는 사용자가 좋아하는 아이템을 기준으로 그 아이템을 좋아하는 사용자를 찾고, 그 사용자가 선호하는 아이템을 추천하다. 

Matrix Factorization는 Collaborative Filtering 기반의 추천 시스템을 구축하기 위해서 사용한다. Matrix Factorization는 행렬 분해의 일종으로 사용자-아이템 (평가) 행렬을 U, Σ, V로 분해한다. 

분해된 행렬 U, Σ, V는 저마다의 의미를 지닌다. 행렬 U는 사용자-특징(feature) 행렬로 행렬 U의 각 행은 사용자를 표현하는 특징(feature) 벡터이다. 행렬 V는 특징(feature)-아이템 행렬이다. 행렬 연산을 위해서 행렬 V를 전치한 Vt를 일반적으로 사용한다. 전치 행렬 Vt의 각 행은 아이템을 표현하는 특징(feature) 벡터이다. 행렬 Σ는 특징값에 대한 대각 행렬이다. 

Matrix Facorization로 분해된 U, Σ, V는 행렬 곱 연산을 통해 결합되어서 원래의 사용자-아이템과 같은 크기의 행렬을 생성한다. 

원 데이터인 사용자-아이템 행렬은 비어있는 값이 많은(sparse) 행렬이다. 한 명의 사용자가 선호하는 아이템은 전체 아이템의 일부이고, 하나의 아이템을 선호하는 사용자 역시 전체 사용자의 극소에 불과하다. Matrix Factorization 으로 분해된 행렬 U, Σ, V를 결합한 행렬은 비어있는 행렬 성분의 값을 추론한다. 추론된 값은 아이템에 대한 사용자의 평점이라고 할 수 있다. 즉 원래는 비어있었던 행렬 성분이 추론됨에 따라서 사용자가 아이템을 선호할 지 판단하는 기준이 된다.

### **2. Modeling**

학습용 데이터와 검증용 데이터를 분리하였으면 학습용 데이터를 바탕으로 모델을 구축한다. 여기서 모델을 구축하는 것은 학습용 데이터를 사용자-아이템 행렬로 변환하고, Matrix Factorization 으로 분해하는 것을 의미한다. 이후 추천할 때에는 분해된 행렬에서 사용자에 대한 벡터와 아이템에 대한 벡터를 찾아 곱셈한 값을 기준으로 이루어진다.

학습용 데이터를 사용자-아이템 행렬로 만들 때 메모리 효율성을 위해 scipy의 csr_matrix 형식을 사용한다. 변환 결과 고유한 사용자 30,640와 고유한 노래 342,679 의 행렬이 만들어졌다. 행렬의 각 성분은 만약 학습용 데이터의 target 값이다. 

30,640 x 342,679 크기의 사용자-아이템 행렬을 sci-kit learn의 Truncated SVD를 사용해 분해한다. 이 때 특징의 개수는 100 개이고, 총 5번 반복한다. 

Matrix Factorization 결과 30,640 x 100인 행렬 U의 각 행은 사용자에 대한 특징 벡터이고, 342,679 x 100인 행렬 Vt의 각 행은 아이템에 대한 특징 벡터이다. 행렬 Σ는 100 x 100 의 크기를 가진 대각행렬이다. 

분해된 행렬을 바탕으로 추천이 이루어진다. 사용자 u가 아이템 i를 좋아할 지 판단하기 위해서 사용자 u와 아이템 i를 각각 특징 벡터로 변환해야한다. 사용자 u에 해당하는 행을 행렬 U에서 찾고 아이템 i에 해당하는 행을 행렬 Vt에서 찾는다. 

![Threshold Test](../figure/model/model-v1-threshold.png)

사용자 특징 벡터, 특징값 대각 행렬, 아이템 특징 벡터를 행렬 곱한 결과는 사용자 u가 아이템 i를 선호하는 지 판단하는 기준이 된다. 특정 임계값(threshold) 이상인 경우 target은 1이고, 미만인 경우 target은 0이다. 

적절한 임계값을 찾기 위해서 검증용 데이터로 실험을 한다. 실험 결과 임계값 20 에 대해서 가장 높은 정확률을 보였으므로 최종 모델은 임계값 20을 기준으로 구현한다. 

### **3. Submission**

최종 제출을 위해 학습용 데이터와 검증용 데이터를 통합하여서 모델을 생성한다. 모델 생성은 이전과 동일하게 사용자-아이템 행렬을 만들고 Matrix Factorization 으로 분해하는 과정이다. 최종 추천을 위한 임계값은 0.1로 사용한다.

![Submission Result](../figure/model/model-v1-submission.png)

테스트용 데이터(test.csv)에 대해 모델은 사용자가 아이템을 한 달 안에 다시 들을지(target)를 예측한다. 생성된 결과를 Kaggle에 최종 제출하였으며 그 결과는 0.54589 이다. 약 절반의 정확률을 보인 것이다. 

## **3. Conclusionn**

가장 기본적인 Matrix Facrotization 방법은 baseline 이다. 따라서 향후 다양한 모델을 적용하여서 정확률을 높일 것이다.

### **Hyper Parameter Tuning**

현재 모델에서 Hyper Parameter는 TruncatedSVD를 적용할 때 Feature의 개수(n_components), 반복 횟수(n_iter) 그리고 예측 시 기준이 되는 임계값(threshold) 이다. 

Hyper Parameter를 조정하는 과정은 Kaggle Kernel을 사용했다. n_components, n_iter의 변화는 엄청나게 큰 성능의 개선을 보장하지 않는다.

![Model Parameter Tuning](../figure/model/model-v1-hyper_parameter_tuning.png)

n_components와 n_iter를 증가시켰을 때 약간의 성능 개선은 있으므로 이를 기반으로 모델을 수정하여서 제출한 결과 0.54660 의 정확률을 보았다. 

![Model Parameter Tuning](../figure/model/model-v1-submission-tuning.png)

