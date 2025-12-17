# Physical AI Research: Medical Motion Content Generation

**과제명**: (2세부) 의료용 Physical AI 학습을 위한 의료 동작 콘텐츠 생성 및 증강 기술 개발

## 프로젝트 개요

NVIDIA 통합 생태계(MONAI, Isaac Sim, Cosmos)를 기반으로 물리적·해부학적 정합성이 보장된 의료 동작 콘텐츠를 생성하고 증강하는 기술 개발

### 최종 목표 (4년)
- 합성 수술 비디오 FVD ≤ 300
- 동작 추론 RMSE ≤ 5° (관절각) 또는 ≤ 5mm (도구 tip)
- 물리 제약 조건 만족률 99%
- 표준 데이터 포맷 v3.0 개발 및 공개

### 1차년도 목표 (9개월)
1. ORBIT-Surgical 기반 의료 시뮬레이션 환경 v1.0
2. 파일럿 합성 비디오 (FVD 기준선 확보)
3. 액션 데이터 표준 포맷 v0.1

## 기술 스택

| 영역 | 기술 | 용도 |
|------|------|------|
| CT 합성 | NVIDIA MAISI (MONAI) | 3D CT 합성 데이터 생성 |
| 비디오 생성 | MONAI Generative Models | 수술 비디오 생성 |
| Sim2Real | NVIDIA Cosmos Transfer | 시뮬레이션→실제 영상 변환 |
| 수술 로봇 시뮬레이션 | ORBIT-Surgical | Isaac Sim 기반 수술 환경 |
| 로봇 학습 | NVIDIA Isaac Lab | 물리 시뮬레이션 및 학습 |
| 모방학습 | LeRobot (Hugging Face) | 로봇 정책 학습 |

## 프로젝트 구조

```
physical-ai-research/
├── README.md                    # 프로젝트 개요
├── docs/                        # 문서
│   ├── research_proposal.pdf    # 연구 제안서
│   ├── progress/               # 진행 상황 리포트
│   └── technical/              # 기술 문서
├── data/                       # 데이터
│   ├── raw/                    # 원본 수술 비디오
│   ├── processed/              # 전처리된 데이터
│   └── synthetic/              # 생성된 합성 데이터
├── src/                        # 소스 코드
│   ├── data/                   # 데이터 처리
│   │   ├── preprocessing.py    # 전처리
│   │   ├── augmentation.py     # 증강
│   │   └── loaders.py          # 데이터 로더
│   ├── generation/             # 생성 모델
│   │   ├── monai_gen/          # MONAI Generative Models
│   │   ├── maisi/              # MAISI CT 합성
│   │   └── cosmos/             # Cosmos Transfer
│   ├── simulation/             # 시뮬레이션
│   │   ├── orbit_surgical/     # ORBIT-Surgical
│   │   └── isaac_sim/          # Isaac Sim
│   ├── training/               # 학습
│   │   ├── isaac_lab/          # Isaac Lab
│   │   └── lerobot/            # LeRobot
│   ├── evaluation/             # 평가
│   │   ├── metrics.py          # FVD, RMSE 계산
│   │   └── validation.py       # 물리 규칙 검증
│   └── utils/                  # 유틸리티
├── notebooks/                  # 실험 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_monai_generation_test.ipynb
│   └── 03_orbit_surgical_setup.ipynb
├── configs/                    # 설정 파일
│   ├── data.yaml
│   ├── training.yaml
│   └── generation.yaml
├── experiments/                # 실험 결과
│   └── logs/
├── tests/                      # 테스트
└── requirements.txt            # 의존성
```

## 환경 설정

### 하드웨어
- GPU: 2x NVIDIA RTX 6000 (48GB each)
- CUDA 지원

### 소프트웨어
```bash
# MONAI 기반 환경
pip install monai[all]
pip install generative-models

# Isaac Sim/Lab (별도 설치 필요)
# ORBIT-Surgical
# LeRobot
```

## 빠른 시작

### Phase 1: PoC - 의료 영상 생성 테스트
```bash
# 1. 오픈소스 수술 데이터셋 다운로드
python scripts/download_datasets.py

# 2. MONAI Generative Models 테스트
jupyter notebook notebooks/02_monai_generation_test.ipynb

# 3. 기준선 메트릭 계산
python src/evaluation/metrics.py --baseline
```

### Phase 2: ORBIT-Surgical 환경 구축
```bash
# Isaac Sim 설치 후
python scripts/setup_orbit_surgical.py
```

## 진행 상황

- [x] 프로젝트 초기 설정
- [x] 연구 제안서 분석
- [ ] 오픈소스 수술 데이터셋 조사
- [ ] MONAI Generative Models PoC
- [ ] ORBIT-Surgical 환경 구축
- [ ] 파일럿 합성 비디오 생성

## 참고 문헌

연구 제안서의 참고문헌 참조 (research_proposal.pdf)

## 라이선스

과기정통부 연구 과제 - 내부 사용
