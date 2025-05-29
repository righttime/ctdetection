# CBCT 이미지 처리 및 시각화

이 프로젝트는 CBCT (Cone Beam Computed Tomography) 이미지를 처리하고 PyVista를 사용하여 3D 시각화하는 도구입니다.

## 🚀 주요 기능

- **DICOM 파일 로딩**: SimpleITK를 사용한 DICOM 시리즈 읽기
- **3D 볼륨 렌더링**: PyVista를 사용한 실시간 3D 시각화
- **3D 메쉬 생성**: Marching Cubes 알고리즘으로 표면 추출
- **대화형 슬라이서**: 실시간으로 슬라이스를 탐색할 수 있는 뷰어
- **구조물 탐지**: 특정 밀도 범위의 구조물 자동 탐지 및 시각화
- **2D 슬라이스 뷰**: Matplotlib을 사용한 2D 단면 시각화

## 📦 설치

1. 필요한 라이브러리 설치:
```bash
pip install -r requirements.txt
```

## 🎯 사용법

### 기본 실행
```bash
python readct.py
```

### 개별 기능 사용 예제

```python
from readct import CBCTProcessor

# CBCT 프로세서 초기화
processor = CBCTProcessor("CT")

# 1. DICOM 시리즈 로드
volume = processor.load_dicom_series()

# 2. 전처리
processor.preprocess_volume()

# 3. 2D 슬라이스 시각화
processor.visualize_slices([30, 50, 70])

# 4. 3D 볼륨 렌더링
processor.visualize_3d_volume()

# 5. 3D 메쉬 시각화 (임계값 조정 가능)
processor.visualize_3d_mesh(threshold=120)

# 6. 대화형 슬라이서
processor.interactive_slicer()

# 7. 특정 밀도 범위의 구조물 찾기
processor.find_and_visualize_features(min_threshold=100, max_threshold=200)
```

## 🛠️ 주요 라이브러리

- **PyVista**: 3D 시각화 및 볼륨 렌더링
- **pydicom**: DICOM 파일 읽기
- **SimpleITK**: 의료 이미지 처리
- **scikit-image**: 이미지 처리 알고리즘
- **NumPy**: 배열 처리
- **Matplotlib**: 2D 시각화

## 🎨 시각화 기능

### 1. 3D 볼륨 렌더링
- 투명도 조절 가능한 볼륨 렌더링
- 다양한 컬러맵 지원
- 실시간 회전 및 확대/축소

### 2. 3D 메쉬 시각화
- Marching Cubes 알고리즘으로 표면 추출
- 스무딩 기능
- 임계값 조절 가능

### 3. 대화형 슬라이서
- 마우스로 슬라이스 평면 조절
- 실시간 단면 보기
- 3D 공간에서의 직관적인 탐색

### 4. 구조물 탐지
- 밀도 기반 구조물 분할
- 연결된 구성요소 분석
- 다중 구조물 동시 시각화

## 📊 데이터 형식

- 입력: DICOM 파일들 (`.dcm`)
- 지원 형식: CT, CBCT, MRI 등 DICOM 표준을 따르는 모든 의료 이미지

## 🔧 매개변수 조정

### 전처리
- `gaussian_sigma`: 가우시안 필터 시그마 값 (기본값: 1.0)

### 3D 메쉬 생성
- `threshold`: 표면 추출 임계값 (자동 설정 또는 수동 조정)
- `smooth_iterations`: 스무딩 반복 횟수 (기본값: 15)

### 구조물 탐지
- `min_threshold`, `max_threshold`: 탐지할 밀도 범위

## 💡 팁

1. **메모리 사용량**: 큰 볼륨의 경우 메모리 사용량이 클 수 있습니다.
2. **임계값 조정**: 다양한 임계값을 시도해보세요. 뼈는 높은 값, 연조직은 낮은 값을 사용합니다.
3. **대화형 기능**: PyVista의 대화형 기능을 활용하여 다양한 각도에서 관찰하세요.

## 🐛 문제 해결

### 일반적인 오류
1. **DICOM 로딩 실패**: 파일 경로와 DICOM 파일 형식을 확인하세요.
2. **메모리 부족**: 더 작은 볼륨으로 테스트하거나 전처리에서 다운샘플링을 추가하세요.
3. **시각화 창이 나타나지 않음**: 백엔드 설정을 확인하세요.

### 성능 최적화
- 큰 데이터셋의 경우 다운샘플링 고려
- GPU 가속을 위한 VTK 설정 확인
- 메쉬 스무딩 반복 횟수 조정

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

