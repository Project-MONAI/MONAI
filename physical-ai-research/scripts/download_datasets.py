"""
오픈소스 수술 데이터셋 다운로드 스크립트

지원 데이터셋:
1. JIGSAWS - Da Vinci Surgical Skills Dataset (비디오 + 키네마틱스)
2. Cholec80 - Laparoscopic Cholecystectomy (신청 필요)
3. EndoVis - MICCAI Challenge Datasets
"""

import os
import argparse
import urllib.request
from pathlib import Path
from typing import List
import zipfile
import gdown


class DatasetDownloader:
    def __init__(self, data_root: str = "../data/raw"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def download_jigsaws(self):
        """
        JIGSAWS 데이터셋 다운로드

        데이터셋 구성:
        - Suturing (39 trials)
        - Needle Passing (28 trials)
        - Knot Tying (36 trials)

        각 trial:
        - Video (.avi)
        - Kinematics (.txt): joint positions, velocities, gripper angles
        - Transcriptions: gesture labels
        """
        print("=" * 60)
        print("JIGSAWS Dataset Download")
        print("=" * 60)

        jigsaws_root = self.data_root / "JIGSAWS"
        jigsaws_root.mkdir(exist_ok=True)

        # JIGSAWS 공식 다운로드 링크
        base_url = "https://cirl.lcsr.jhu.edu/wp-content/uploads/2015/11/"

        tasks = {
            "Suturing": "Suturing.zip",
            "Needle_Passing": "Needle_Passing.zip",
            "Knot_Tying": "Knot_Tying.zip"
        }

        print("\nJIGSAWS 데이터셋 정보:")
        print("- 출처: Johns Hopkins University CIRL")
        print("- 로봇: da Vinci Research Kit (dVRK)")
        print("- 데이터: 비디오 + 키네마틱스 + Gesture labels")
        print("- 라이선스: 오픈 액세스 (연구 목적)")
        print()

        print("⚠️  주의: JIGSAWS 공식 웹사이트에서 직접 다운로드가 필요합니다.")
        print("    웹사이트: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/")
        print()
        print("다운로드 절차:")
        print("1. 위 링크 방문")
        print("2. Suturing, Needle Passing, Knot Tying 데이터 다운로드")
        print(f"3. 다운로드한 파일을 {jigsaws_root.absolute()}에 압축 해제")
        print()

        # 데이터 구조 예시 생성
        readme_path = jigsaws_root / "README.md"
        with open(readme_path, "w") as f:
            f.write("""# JIGSAWS Dataset Structure

## Expected Directory Layout

```
JIGSAWS/
├── Suturing/
│   ├── video/
│   │   ├── Suturing_S001_T001.avi
│   │   ├── Suturing_S001_T002.avi
│   │   └── ...
│   ├── kinematics/
│   │   ├── Suturing_S001_T001.txt
│   │   └── ...
│   └── transcriptions/
│       ├── Suturing_S001_T001.txt
│       └── ...
├── Needle_Passing/
│   └── ...
└── Knot_Tying/
    └── ...
```

## Data Format

### Video
- Format: AVI
- Resolution: 640x480
- FPS: 30

### Kinematics (.txt)
- Columns: 76 dimensions
  - Master: 38 dimensions (PSM1 + PSM2)
  - Slave: 38 dimensions (PSM1 + PSM2)
  - Per robot: 19 dimensions
    - Position (x, y, z): 3
    - Velocity (x, y, z): 3
    - Rotation matrix: 9
    - Gripper angle: 1
    - Angular velocity: 3

### Gesture Labels (.txt)
- G1: Reaching for needle with right hand
- G2: Positioning needle
- G3: Pushing needle through tissue
- G4: Transferring needle from left to right
- G5: Moving to center with needle in grip
- G6: Pulling suture with left hand
- ...
""")

        print(f"✓ README 생성 완료: {readme_path}")
        print(f"\n데이터 다운로드 후 {jigsaws_root.absolute()}에 압축 해제하세요.")

        return jigsaws_root

    def download_cholec80(self):
        """
        Cholec80 데이터셋 다운로드 안내
        """
        print("=" * 60)
        print("Cholec80 Dataset Download Instructions")
        print("=" * 60)

        cholec_root = self.data_root / "Cholec80"
        cholec_root.mkdir(exist_ok=True)

        print("\nCholec80 데이터셋 정보:")
        print("- 출처: University of Strasbourg (CAMMA Lab)")
        print("- 비디오 수: 80개 복강경 담낭 절제술")
        print("- 해상도: 1920x1080 @ 25fps")
        print("- 총 길이: ~80시간")
        print("- 어노테이션: 7 surgical phases, 7 tool types")
        print()

        print("⚠️  주의: Cholec80은 신청 및 승인 후 다운로드 가능합니다.")
        print()
        print("신청 절차:")
        print("1. 웹사이트 방문: http://camma.u-strasbg.fr/datasets")
        print("2. 'Cholec80' 섹션에서 데이터 신청")
        print("3. 소속, 연구 목적 등 정보 제공")
        print("4. 승인 후 다운로드 링크 수령 (보통 1-2일 소요)")
        print(f"5. 다운로드 후 {cholec_root.absolute()}에 저장")
        print()

        readme_path = cholec_root / "README.md"
        with open(readme_path, "w") as f:
            f.write("""# Cholec80 Dataset

## Dataset Information
- 80 laparoscopic cholecystectomy videos
- Resolution: 1920x1080 @ 25fps
- Total duration: ~80 hours

## Surgical Phases (7)
1. Preparation
2. CalotTriangle Dissection
3. Clipping Cutting
4. Gallbladder Dissection
5. Gallbladder Packaging
6. Cleaning Coagulation
7. Gallbladder Retraction

## Tool Types (7)
1. Grasper
2. Bipolar
3. Hook
4. Scissors
5. Clipper
6. Irrigator
7. Specimen Bag

## Citation
```bibtex
@article{twinanda2017endonet,
  title={EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos},
  author={Twinanda, Andru P and Shehata, Sherif and Mutter, Didier and Marescaux, Jacques and De Mathelin, Michel and Padoy, Nicolas},
  journal={IEEE TMI},
  year={2017}
}
```
""")

        print(f"✓ README 생성 완료: {readme_path}")
        return cholec_root

    def check_datasets(self):
        """다운로드된 데이터셋 확인"""
        print("=" * 60)
        print("Dataset Status Check")
        print("=" * 60)

        datasets = ["JIGSAWS", "Cholec80", "EndoVis"]

        for dataset in datasets:
            dataset_path = self.data_root / dataset
            if dataset_path.exists():
                files = list(dataset_path.rglob("*"))
                video_files = [f for f in files if f.suffix in ['.avi', '.mp4', '.mov']]
                print(f"\n✓ {dataset}:")
                print(f"  경로: {dataset_path}")
                print(f"  총 파일: {len(files)}")
                print(f"  비디오 파일: {len(video_files)}")
            else:
                print(f"\n✗ {dataset}: 없음")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="수술 비디오 오픈소스 데이터셋 다운로드"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["jigsaws", "cholec80", "endovis", "all", "check"],
        default="check",
        help="다운로드할 데이터셋 선택"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/raw",
        help="데이터 저장 경로"
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(data_root=args.data_root)

    if args.dataset == "check":
        downloader.check_datasets()
    elif args.dataset == "jigsaws":
        downloader.download_jigsaws()
    elif args.dataset == "cholec80":
        downloader.download_cholec80()
    elif args.dataset == "all":
        downloader.download_jigsaws()
        downloader.download_cholec80()
        print("\n모든 데이터셋 다운로드 안내 완료!")
        print("각 데이터셋의 README를 참고하여 수동 다운로드를 진행하세요.")


if __name__ == "__main__":
    main()
