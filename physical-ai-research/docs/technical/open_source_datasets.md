# 오픈소스 수술 비디오 데이터셋

## 1. Cholec80 (복강경 담낭 절제술)

### 개요
- **비디오 수**: 80개 수술 비디오
- **해상도**: 1920x1080 @ 25fps
- **총 길이**: ~80시간
- **내용**: 복강경 담낭 절제술 전체 과정
- **어노테이션**: 7가지 수술 단계, 7가지 수술 도구

### 다운로드
- **웹사이트**: http://camma.u-strasbg.fr/datasets
- **논문**: Twinanda et al., "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos" (2017)
- **라이선스**: 연구 목적 사용 가능 (신청 필요)

### 활용 방안
- 수술 단계 인식 학습
- 도구 검출 및 추적
- 비디오 생성 모델의 기준 데이터

---

## 2. JIGSAWS (Da Vinci Surgical Skills Dataset)

### 개요
- **비디오 수**: 39명의 외과의가 수행한 103개 시연
- **태스크**: Suturing, Needle Passing, Knot Tying
- **데이터**: 비디오 + 키네마틱스 (관절 위치, 속도, 그리퍼 각도)
- **로봇**: da Vinci Research Kit (dVRK)

### 다운로드
- **웹사이트**: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
- **논문**: Gao et al., "JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS)" (2014)
- **라이선스**: 오픈 액세스

### 활용 방안
- **1차년도 핵심**: 비디오-키네마틱스 쌍 데이터
- 로봇 동작 학습 및 검증
- 액션 데이터 추출 기술 검증
- da Vinci 로봇 ORBIT-Surgical 시뮬레이션 검증

---

## 3. MICCAI EndoVis Challenge Datasets

### 개요
- **시리즈**: 2015-2024 매년 개최
- **태스크**:
  - Instrument Segmentation
  - Tracking
  - Robotic Scene Segmentation (2017, 2019)
  - Surgical Action Triplet (2021-2022)

### 주요 데이터셋
- **EndoVis 2017**: Robotic Instrument Segmentation (8 videos, 255 frames)
- **EndoVis 2018**: Robotic Scene Segmentation (19 videos)
- **SAR-RARP50**: 50 prostatectomy videos

### 다운로드
- **웹사이트**: https://endovis.grand-challenge.org/
- **라이선스**: Challenge 등록 후 다운로드

### 활용 방안
- 수술 장면 세그멘테이션
- 도구 추적 및 동작 인식
- 합성 데이터 품질 평가 벤치마크

---

## 4. HeiChole (Heidelberg Colorectal Dataset)

### 개요
- **비디오 수**: 30개 복강경 대장 수술
- **어노테이션**: 수술 단계, 해부학적 구조, 도구
- **해상도**: Full HD

### 다운로드
- **웹사이트**: https://www.synapse.org/#!Synapse:syn18824884/
- **라이선스**: 연구 목적 사용 가능

---

## 5. CholecT50 (Surgical Action Triplet)

### 개요
- **비디오 수**: 50개 복강경 담낭 절제술
- **어노테이션**: Triplet (Instrument, Verb, Target) - 100개 action triplets
- **예시**: <grasper, grasp, gallbladder>

### 다운로드
- **웹사이트**: http://camma.u-strasbg.fr/datasets
- **논문**: Nwoye et al., "Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos" (2020)

### 활용 방안
- 수술 동작의 세밀한 이해
- 액션 데이터 표준 포맷 설계 참고

---

## 6. m2cai16-tool (MICCAI 2016 Tool Detection)

### 개요
- **비디오 수**: 15개 훈련, 10개 테스트
- **태스크**: 7가지 수술 도구 실시간 검출
- **프레임**: 10,000+ 어노테이션

### 다운로드
- **웹사이트**: http://ai.stanford.edu/~syyeung/tooldetection.html

---

## 추천 우선순위 (1차년도 PoC용)

### 1순위: **JIGSAWS**
- ✅ **비디오 + 키네마틱스** 모두 포함
- ✅ da Vinci 로봇 (ORBIT-Surgical 호환)
- ✅ 즉시 다운로드 가능
- ✅ 1차년도 목표 직접 부합

### 2순위: **Cholec80**
- ✅ 대용량 수술 비디오
- ✅ 비디오 생성 모델 학습/평가
- ⚠️ 키네마틱스 없음 (추출 필요)

### 3순위: **EndoVis 2017/2018**
- ✅ 로봇 수술 특화
- ✅ 세그멘테이션 어노테이션
- ⚠️ 상대적으로 적은 비디오 수

---

## 다운로드 스크립트 작성 계획

```python
# scripts/download_datasets.py
def download_jigsaws():
    """JIGSAWS 데이터셋 다운로드"""
    pass

def download_cholec80():
    """Cholec80 신청 및 다운로드 안내"""
    pass

def download_endovis():
    """EndoVis 데이터셋 다운로드"""
    pass
```

---

## 참고 링크

- CAMMA Lab (Strasbourg): http://camma.u-strasbg.fr/
- Johns Hopkins CIRL: https://cirl.lcsr.jhu.edu/
- EndoVis Grand Challenge: https://endovis.grand-challenge.org/
- Surgical Data Science Review: https://arxiv.org/abs/2206.02053
