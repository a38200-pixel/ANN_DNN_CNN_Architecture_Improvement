# Emotion Classification Architecture Improvement
### 어린이 얼굴 감정 분류 모델 구조 개선 및 Keras·PyTorch 비교

어린이 얼굴 이미지를 **8개 감정 클래스**로 분류하는 문제를 대상으로,  
**ANN / DNN / CNN 구조를 직접 설계하고 개선**한 뒤 **Keras(TensorFlow)와 PyTorch**에서 각각 구현하여 성능을 비교한 프로젝트입니다.

단순히 모델별 정확도를 비교하는 데 그치지 않고, **과적합·Underfitting·일반화 성능 저하의 원인을 분석하고 구조와 학습 조건을 단계적으로 수정**하는 데 중점을 두었습니다.

---

## 1. Project Overview

| 항목 | 내용 |
|---|---|
| Task | Facial Emotion Classification |
| Classes | anger, contempt, disgust, fear, happy, neutral, sad, surprise |
| Input Size | 128 × 128 |
| Models | ANN, DNN, CNN |
| Frameworks | Keras(TensorFlow), PyTorch |
| ANN / DNN Input | Grayscale, 1 Channel |
| CNN Input | RGB, 3 Channels |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Metrics | Test Loss, Accuracy, Macro F1, Train–Validation Gap, Class-wise F1 |

### Project Goal

이 프로젝트의 핵심 목표는 다음과 같습니다.

1. ANN, DNN, CNN의 구조적 차이가 이미지 분류 성능에 미치는 영향을 확인
2. 초기 모델에서 발생한 **과적합과 일반화 성능 저하 원인 분석**
3. Dropout, LayerNorm, Data Augmentation, RGB 입력 등 구조 및 학습 조건 개선
4. 개선된 구조를 Keras와 PyTorch에서 각각 구현
5. 프레임워크 차이와 모델 구조 차이가 결과에 미치는 영향 비교

---

## 2. Model Architecture Improvement

### ANN

기준 모델로 사용한 가장 단순한 Fully Connected Network입니다.

```text
Input
  ↓
Flatten
  ↓
Dense / Linear (128)
  ↓
ReLU
  ↓
Output (8)
```

ANN은 비교적 단순하고 Train–Validation 차이도 크지 않았지만,  
이미지를 Flatten하여 처리하기 때문에 **공간적 특징을 충분히 활용하지 못하는 한계**가 있었습니다.

---

### DNN

ANN보다 깊은 구조를 구성하고 정규화 방법을 실험했습니다.

```text
Input
  ↓
Flatten
  ↓
512 → ReLU → LayerNorm
  ↓
256 → ReLU → LayerNorm
  ↓
128 → ReLU
  ↓
Output (8)
```

Keras DNN에서는 다음과 같은 Ablation Experiment를 진행했습니다.

| Experiment | Train Acc | Val Acc | Test Acc |
|---|---:|---:|---:|
| Baseline | 39.90% | 36.37% | 36.09% |
| Dropout only | 24.26% | 29.22% | 27.83% |
| Augmentation only | 35.77% | 34.41% | 33.00% |
| **LayerNorm only** | **40.32%** | **40.15%** | **39.81%** |

### 개선 결과

- Dropout 적용 시 오히려 성능이 크게 감소하여 **Underfitting 가능성** 확인
- 단순 Augmentation만으로는 유의미한 성능 향상이 나타나지 않음
- **LayerNorm only 구성에서 가장 안정적인 성능** 확인
- 최종 DNN에서는 Dropout을 제거하고 LayerNorm을 유지

---

## 3. CNN Improvement

CNN에서는 이미지의 공간적 특징을 직접 추출하도록 Convolution Block을 구성했습니다.

```text
Input
  ↓
Conv(32)  → BatchNorm → ReLU → Pool
  ↓
Conv(64)  → BatchNorm → ReLU → Pool
  ↓
Conv(128) → BatchNorm → ReLU → Pool
  ↓
Conv(256) → BatchNorm → ReLU → Pool
  ↓
Global Average Pooling
  ↓
Dense / Linear (128)
  ↓
Output (8)
```

### Experiment Flow

```text
Baseline CNN
    ↓
Overfitting 확인
    ↓
Data Augmentation 적용
    ↓
Callback 조건 수정
    ↓
Dropout 0.2 실험
    ↓
Underfitting 확인 → Dropout 제거
    ↓
Grayscale / RGB 비교
    ↓
RGB + Augmentation 최종 선택
```

### 주요 개선 과정

- Baseline CNN에서 약 **20%p 수준의 Train–Validation Gap** 확인
- Flip + Rotation 기반 Data Augmentation 적용
- Learning Rate Scheduler / EarlyStopping 조건 조정
- Dropout 0.2 적용 시 Test Accuracy가 **44.85%**로 감소하여 제거
- Grayscale + Augmentation: **52.02%**
- RGB + Augmentation: **52.47%**
- 최종적으로 **RGB + Augmentation + Callback 개선 구조** 선택

---

## 4. Final Results

### Keras

| Model | Test Loss | Accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|---:|
| ANN | 1.4264 | 44.89% | 43.83% | 2,098,312 |
| DNN | 1.4370 | 44.03% | 43.46% | 8,555,912 |
| **CNN** | **1.2620** | **52.47%** | **52.00%** | **423,208** |

### PyTorch

| Model | Test Loss | Accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|---:|
| ANN | 1.4570 | 41.04% | 39.36% | 2,098,312 |
| DNN | 1.4545 | 42.71% | 42.06% | 8,555,912 |
| **CNN** | **1.2832** | **50.70%** | **49.46%** | **422,824** |

---

## 5. Keras vs PyTorch

| Model | Keras Accuracy | PyTorch Accuracy | Difference |
|---|---:|---:|---:|
| ANN | 44.89% | 41.04% | +3.86%p |
| DNN | 44.03% | 42.71% | +1.32%p |
| CNN | **52.47%** | **50.70%** | +1.77%p |

이번 실험에서는 Keras가 세 모델 모두에서 조금 더 높은 성능을 기록했습니다.

그러나 이를 프레임워크 자체의 절대적인 성능 차이로 해석하기보다는  
**초기화 방식, Optimizer 세부 설정, Data Augmentation 구현, BatchNorm / LayerNorm 동작 차이 등 구현 세부사항의 영향**을 함께 고려했습니다.

더 중요한 결과는 두 프레임워크에서 공통적으로 다음 경향이 재현되었다는 점입니다.

> **CNN > DNN ≈ ANN**

즉, 이번 감정 분류 문제에서는 **프레임워크의 차이보다 모델 구조의 차이가 성능에 더 큰 영향을 주었습니다.**

---

## 6. Class-wise Analysis

CNN의 클래스별 F1 Score를 비교했을 때 다음 특징을 확인했습니다.

| Class | Keras CNN F1 | PyTorch CNN F1 |
|---|---:|---:|
| anger | 0.4364 | 0.3970 |
| contempt | 0.5129 | 0.5387 |
| disgust | 0.4369 | 0.3714 |
| fear | 0.3702 | 0.3484 |
| **happy** | **0.8696** | **0.8497** |
| **neutral** | **0.7949** | **0.7923** |
| sad | 0.3348 | 0.2559 |
| surprise | 0.3988 | 0.4034 |

### Error Analysis

- `happy`, `neutral`은 두 프레임워크 모두 높은 F1 Score 기록
- `sad`, `fear`, `disgust`는 상대적으로 낮은 분류 성능
- 감정 간 표정 특징이 유사한 클래스에서는 혼동 가능성이 높다고 판단
- 향후 Confusion Matrix를 활용한 클래스별 오류 분석 필요

---

## 7. Key Findings

### 01. 이미지 분류에서는 공간 정보가 중요하다

ANN과 DNN은 입력 이미지를 Flatten하기 때문에 픽셀 간 공간 관계가 사라집니다.

반면 CNN은 Convolution을 통해 표정의 국소적인 특징과 공간적 패턴을 추출할 수 있었고,  
두 프레임워크 모두에서 가장 높은 Accuracy와 Macro F1을 기록했습니다.

### 02. 복잡한 모델이 항상 더 좋은 것은 아니다

DNN은 ANN보다 훨씬 많은 파라미터를 사용했지만 성능 향상은 제한적이었습니다.

특히 Keras 기준:

```text
ANN : 2.10M Parameters → Accuracy 44.89%
DNN : 8.56M Parameters → Accuracy 44.03%
CNN : 0.42M Parameters → Accuracy 52.47%
```

파라미터 수를 단순히 증가시키는 것보다  
**데이터 특성에 적합한 구조를 선택하는 것이 중요**하다는 점을 확인했습니다.

### 03. Regularization도 데이터와 모델에 맞게 선택해야 한다

Dropout은 일반적으로 과적합 완화에 사용되지만, 이번 실험에서는 일부 모델의 학습 능력을 지나치게 제한했습니다.

따라서 정규화 기법을 관습적으로 적용하기보다  
**Ablation Experiment를 통해 실제 효과를 검증한 뒤 적용**했습니다.

### 04. Framework보다 Architecture의 영향이 더 컸다

Keras와 PyTorch에서 세부적인 성능 차이는 존재했지만,  
두 환경 모두 CNN이 가장 높은 성능을 기록했습니다.

이를 통해 이번 실험에서는 **프레임워크 선택보다 문제에 적합한 모델 구조 설계가 더 중요**하다는 결론을 얻었습니다.

---

## 8. Future Work

추가적으로 다음 실험을 진행할 수 있습니다.

- Conv 1개 → Pool 구조를 **Conv 2개 → Pool 구조**로 변경
- Flip / Rotation / Flip + Rotation 개별 Ablation
- Learning Rate `0.001 → 0.0005` 비교
- BatchNorm / GroupNorm 비교
- 다중 Random Seed 실험을 통한 평균 성능 및 표준편차 비교
- Confusion Matrix 기반 `sad`, `fear`, `disgust` 오류 분석
- Class Imbalance 및 클래스별 데이터 품질 분석

---

## 9. Repository Contents

이 Repository는 전체 학습 코드를 공개하는 형태가 아니라,  
**모델 구조 개선 과정과 실험 결과를 정리한 포트폴리오용 Repository**입니다.

```text
.
├── README.md
├── 감정분류_모델구조_개선_및_프레임워크_비교_통합보고서.docx
├── presentation.pptx
└── trained_model.*
```

### Files

| File | Description |
|---|---|
| `README.md` | 프로젝트 개요 및 핵심 실험 결과 |
| `감정분류_모델구조_개선_및_프레임워크_비교_통합보고서.docx` | 전체 실험 과정과 결과를 정리한 통합 보고서 |
| `presentation.pptx` | 발표용 프로젝트 요약 자료 |
| `trained_model.*` | 최종 학습 모델 |

> `presentation.pptx`와 `trained_model.*`은 실제 GitHub에 업로드할 파일명에 맞게 README의 파일명을 수정하면 됩니다.

---

## 10. Tech Stack

**Deep Learning**

- ANN
- DNN
- CNN

**Framework**

- TensorFlow / Keras
- PyTorch

**Experiment**

- Layer Normalization
- Batch Normalization
- Data Augmentation
- Early Stopping
- Learning Rate Scheduling
- Ablation Experiment

**Evaluation**

- Accuracy
- Macro F1 Score
- Train–Validation Gap
- Class-wise F1 Score

---

## Conclusion

본 프로젝트에서는 ANN, DNN, CNN을 단순 구현하는 데서 끝나지 않고,  
초기 실험에서 발생한 문제를 분석하고 **구조와 학습 조건을 단계적으로 수정하는 실험 과정**을 진행했습니다.

그 결과 Keras와 PyTorch 모두에서 CNN이 가장 높은 성능을 보였으며,  
얼굴 감정 분류와 같은 이미지 문제에서는 **공간적 특징을 보존하고 추출할 수 있는 모델 구조가 핵심**이라는 점을 확인했습니다.

또한 동일한 구조를 두 프레임워크에서 재구현하고 비교함으로써,  
프레임워크 자체의 우열보다 **모델 구조 설계와 실험 조건의 영향이 더 중요할 수 있음을 확인**했습니다.
