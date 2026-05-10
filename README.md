# House Prices: Advanced Regression Techniques
> **Feature Engineering Pipeline & Predictive Modeling with XAI**

본 프로젝트는 실제 데이터를 활용하여 머신러닝 성능 향상을 위한 **특성 공학(Feature Engineering) 파이프라인**을 설계하고, 전처리 및 변수 선택 전략에 따른 성능 차이를 비교·분석하는 것을 목표로 한다.

---

## 1. 개요 (Assignment Overview)

본 프로젝트는 데이터 탐색(EDA)부터 결측치 처리, 인코딩, 스케일링, 변수 선택, 모델 학습 및 평가까지 전체 **ML Pipeline**을 직접 구현함으로써 데이터 사이언스의 워크플로우를 이해하고, 모델의 성능을 정량적으로 개선하는 과정을 학습한다.

## 2. 데이터셋 안내 (Dataset Information)

본 프로젝트는 제시된 후보 데이터셋 중 '**House Prices Dataset**'을 선택하여 진행하였다.

- **목적**: 미국 Ames 지역의 주택 특성 데이터를 기반으로 주택 가격(`SalePrice`) 예측 (Regression)

- **주요 특징**: 
    
    - 79개의 설명 변수로 구성된 고차원 데이터셋
    
    - 수치형(Numerical) 및 범주형(Categorical) 변수가 혼합된 실무형 데이터
    
    - 풍부한 결측치와 이상치가 포함되어 특성 공학 실습에 최적화됨

- **Source**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## 3. 과제 수행 목표 (Project Objectives)

다음 과정을 포함한 엔드 투 엔드 머신러닝 파이프라인을 구축하였다.

### STEP 01. 데이터 탐색 (EDA)

- 타겟 변수(`SalePrice`)의 분포 분석 및 로그 변환(`log1p`)을 통한 정규화

- 특성별 결측치 비율 시각화 및 상관관계 행렬(Heatmap) 분석

- 수치형 변수의 분포 및 이상치(Outlier) 탐색

### STEP 02. 데이터 전처리 및 파이프라인 설계

- **Scikit-learn Pipeline** 및 **ColumnTransformer**를 활용한 자동화

- **결측치 처리**: `SimpleImputer`를 통한 평균/중앙값/최빈값 전략 비교

- **인코딩**: `OneHotEncoder` 및 `LabelEncoder` 성능 대조

- **스케일링**: `Standard`, `MinMax`, `RobustScaler`의 기여도 평가

### STEP 03. 특성 공학 (Feature Engineering)

- **파생 변수 생성**: 

    - `TotalSF`: 주택의 총 면적 통합

    - `HouseAge`: 연식 데이터를 활용한 노후도 지표 생성

    - `TotalBath`: 화장실 개수 통합 지표 생성

- 성능 향상을 위한 도메인 지식 기반의 피처 생성 및 선택

### STEP 04 & 05. 모델 학습, 평가 및 최적화

- **사용 모델**: Random Forest Regressor, XGBoost Regressor 등

- **최적화**: `GridSearchCV`를 활용한 하이퍼파라미터 튜닝

- **평가 지표**: $RMSE, MAE, R^2$ Score 기반 정량 분석

---

## 특이 사항

본 프로젝트는 가이드라인에서 제시한 고도화 항목을 모두 수행하였다.

1.  **Pipeline 객체 활용**: 전처리부터 모델 학습까지 전체 과정을 `Pipeline` 객체로 묶어 데이터 누수(Data Leakage)를 원천 차단함.

2.  **GridSearchCV 적용**: XGBoost 모델에 대해 교차 검증 기반 파라미터 튜닝을 수행하여 일반화 성능 최적화.

3.  **SHAP 기반 분석**: **XAI(Explainable AI)** 기법인 SHAP을 도입하여 블랙박스 모델의 예측 근거를 시각적으로 해석함.

4.  **AutoML 비교 실험**: 선형 모델부터 부스팅 모델까지 5종 이상의 알고리즘을 자동 비교 분석하여 최적의 모델군 탐색.

5.  **시각화 고도화**: Feature Importance 및 SHAP Summary Plot 등을 통해 변수 영향력을 심층 시각화함.

---

## 실험 결과 요약 (Experimental Results)

### 전처리 전략별 성능 비교 (5-Fold CV RMSE)
| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection | RMSE (CV) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base** | mean | onehot | standard | X | **0.1360** |
| Exp-1 | mean | onehot | standard | X | 0.1360 |
| Exp-2 | median | label | minmax | O | 0.1476 |
| Exp-3 | most_frequent | onehot | robust | O | 0.1475 |

### 최종 모델 성능 지표 (Optimized XGBoost)

- **$RMSE$**: $16,000 내외 (실제 가격 기준)

- **$R^2$ Score**: **0.9024** (테스트 셋 기준)

---

## 최종 결론

본 프로젝트를 통해 체계적인 데이터 전처리와 파생 변수 생성이 모델 성능에 미치는 정량적인 영향을 검증하였다. 특히 SHAP 분석 결과, 주택의 '**총 면적(TotalSF)**'과 '**품질 등급(OverallQual)**'이 가격 결정의 핵심 지표임을 입증하였으며, 최적화된 파이프라인을 통해 높은 설명력($R^2 > 0.90$)을 가진 예측 모델을 성공적으로 구현하였다.