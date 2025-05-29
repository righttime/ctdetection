import os
import pydicom
import numpy as np
import vtk
from vtk.util import numpy_support

def load_scan(path):
    """
    지정된 경로에서 DICOM 슬라이스들을 로드하고 정렬합니다.
    """
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path) if s.endswith('.dcm')]
    
    # InstanceNumber 또는 SliceLocation을 기준으로 슬라이스 정렬
    # 두 속성 모두 없을 경우 파일 이름으로 정렬 (주의: 정확하지 않을 수 있음)
    if hasattr(slices[0], 'InstanceNumber'):
        slices.sort(key=lambda x: int(x.InstanceNumber))
    elif hasattr(slices[0], 'SliceLocation'):
        slices.sort(key=lambda x: float(x.SliceLocation))
    else:
        # 경고: InstanceNumber 또는 SliceLocation DICOM 태그를 찾을 수 없습니다. 파일명으로 정렬합니다.
        # 순서가 정확하지 않을 수 있습니다.
        print("경고: InstanceNumber 또는 SliceLocation DICOM 태그를 찾을 수 없습니다. 파일 이름으로 정렬합니다.")
        slices.sort(key=lambda x: x.filename)
        
    return slices

def get_pixels_hu(slices):
    """
    슬라이스들로부터 3D 픽셀 데이터(NumPy 배열)를 생성하고,
    필요한 경우 Hounsfield Unit(HU)으로 변환합니다.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16) # 일반적으로 CT 데이터는 int16

    # Rescale Intercept 및 Rescale Slope 적용 (HU 변환)
    # 첫 번째 슬라이스의 값을 기준으로 적용 (일반적으로 동일)
    if hasattr(slices[0], 'RescaleIntercept') and hasattr(slices[0], 'RescaleSlope'):
        intercept = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def segment_teeth(pixel_array_hu, hu_threshold_min=700, hu_threshold_max=3000):
    """
    HU 값을 사용하여 치아 영역을 세그멘테이션합니다.
    """
    try:
        from skimage import measure
        from skimage.morphology import closing, opening, disk, ball
        from scipy.ndimage import binary_fill_holes
    except ImportError:
        print("치아 세그멘테이션을 위해서는 scikit-image와 scipy가 필요합니다.")
        print("pip install scikit-image scipy")
        return None

    print(f"치아 세그멘테이션 시작 (HU 임계값: {hu_threshold_min} ~ {hu_threshold_max})...")
    
    # 1. 임계값 적용
    # 뼈와 치아에 해당하는 HU 범위 설정 (이 값은 데이터에 따라 조절 필요)
    # 일반적으로 치아 에나멜은 뼈보다 훨씬 높은 HU 값을 가짐
    binary_image = (pixel_array_hu >= hu_threshold_min) & (pixel_array_hu <= hu_threshold_max)

    # 2. 형태학적 처리 (노이즈 제거 및 구멍 채우기)
    # 3D 처리를 위해 ball 구조 요소 사용
    # opening: 작은 노이즈 제거
    # closing: 객체 내 작은 구멍 채우기
    print("형태학적 처리 중 (opening, closing, fill_holes)...")
    struct_element_3d = ball(2) # 3x3x3 구형 구조 요소, 크기 조절 가능
    
    # Opening 연산 (작은 객체 제거)
    # binary_image = opening(binary_image, struct_element_3d)
    
    # Closing 연산 (내부 구멍 메우기)
    # binary_image = closing(binary_image, struct_element_3d)

    # 각 2D 슬라이스에 대해 구멍 채우기 (3D fill_holes는 복잡할 수 있음)
    # 더 정교한 방법은 3D 연결 요소 분석 후 작은 객체 제거
    filled_image = np.zeros_like(binary_image)
    for i in range(binary_image.shape[0]): # Z 축 (슬라이스)
        filled_image[i, :, :] = binary_fill_holes(binary_image[i, :, :])
    
    # 3. 연결 요소 분석 (가장 큰 객체 유지 또는 특정 크기 이상의 객체들 유지)
    print("연결 요소 분석 중...")
    # 레이블링: 각 연결된 영역에 고유한 번호 할당
    labeled_image, num_features = measure.label(filled_image, connectivity=3, return_num=True) # 3차원 연결성
    
    if num_features == 0:
        print("세그멘테이션 결과, 객체를 찾지 못했습니다.")
        return None

    print(f"{num_features}개의 연결된 객체 발견.")

    # 객체 속성 분석 시 원본 HU 값을 사용하여 평균 강도 계산
    regions = measure.regionprops(labeled_image, intensity_image=pixel_array_hu)
    
    final_segmentation = np.zeros_like(labeled_image, dtype=bool)
    found_teeth_structures = False

    min_object_size = 10000 # 이 값은 해상도 및 치아 크기에 따라 조절 (사용자 설정값 반영)
    max_object_size = 1000000

    # 치아로 판단할 평균 HU 임계값 (선택적 기준)
    # 예를 들어, 설정된 최소 HU 임계값보다 훨씬 높은 값을 기준으로 설정
    # 또는 (hu_threshold_min + hu_threshold_max) / 2 같은 값을 사용할 수 있음
    # 여기서는 사용자가 설정한 hu_threshold_min 보다 약간 높은 값을 기준으로 설정해봄
    # 실제 치아의 평균 HU는 매우 높을 것이라는 가정.
    mean_hu_threshold_for_teeth = hu_threshold_min #+ (hu_threshold_max - hu_threshold_min) * 0.1 # 최소값에서 범위의 1/4 정도 더 높은 값


    print(f"적용될 최소 객체 크기: {min_object_size}, 적용될 최소 평균 HU: {mean_hu_threshold_for_teeth:.2f}")

    for region in regions:
        # 크기 조건과 함께 평균 HU 값 조건 추가
        if region.area >= min_object_size and region.area <= max_object_size and region.mean_intensity >= mean_hu_threshold_for_teeth:
            final_segmentation[labeled_image == region.label] = True
            found_teeth_structures = True
            print(f"객체 ID {region.label}: 크기 {region.area}, 평균 HU {region.mean_intensity:.2f} (유지)")
        else:
            if (region.area > 100):
                print(f"객체 ID {region.label}: 크기 {region.area}, 평균 HU {region.mean_intensity:.2f} (제거 - 크기 또는 평균 HU ({mean_hu_threshold_for_teeth:.2f}) 미달)")


    if not found_teeth_structures:
        print(f"최소 크기 ({min_object_size} 픽셀) 이상의 객체를 찾지 못했습니다. 임계값 또는 크기 필터를 조절해보세요.")
        # 모든 객체를 반환하거나, 가장 큰 것 하나만 반환하는 등의 fallback 로직 추가 가능
        # 여기서는 발견된 모든 객체를 사용 (크기 필터링 없이)
        # final_segmentation = filled_image
        return None


    print("치아 세그멘테이션 완료.")
    return final_segmentation

def main():
    dicom_dir = 'CT'  # DICOM 파일들이 있는 디렉토리

    # 1. DICOM 파일 로드 및 정렬
    print(f"'{dicom_dir}' 디렉토리에서 DICOM 파일들을 로드 중입니다...")
    patient_slices = load_scan(dicom_dir)
    if not patient_slices:
        print(f"'{dicom_dir}' 디렉토리에서 DICOM 파일을 찾을 수 없습니다.")
        return

    # 2. 픽셀 데이터 추출 (HU 단위)
    print("픽셀 데이터를 추출하고 Hounsfield Unit으로 변환 중입니다...")
    pixel_data_3d = get_pixels_hu(patient_slices)

    # 3. DICOM 기본 정보 출력
    print("\n--- DICOM 정보 ---")
    # 이미지 크기 (Number of slices, Height, Width)
    image_shape = pixel_data_3d.shape
    print(f"이미지 크기 (슬라이스 수, 높이, 너비): {image_shape}")

    # 픽셀 간격 (Slice Thickness, Pixel Spacing row, Pixel Spacing col)
    # SliceThickness는 z 간격, PixelSpacing은 x, y 간격
    slice_thickness = patient_slices[0].get('SliceThickness', 'N/A')
    pixel_spacing = patient_slices[0].get('PixelSpacing', ['N/A', 'N/A'])
    
    # PixelSpacing이 문자열 리스트일 수 있으므로 float으로 변환 시도
    try:
        spacing_x = float(pixel_spacing[0])
        spacing_y = float(pixel_spacing[1])
    except (TypeError, ValueError, IndexError):
        spacing_x = 1.0 # 기본값
        spacing_y = 1.0 # 기본값
        print("경고: PixelSpacing 정보를 정확히 읽을 수 없어 기본값(1.0)을 사용합니다.")

    try:
        spacing_z = float(slice_thickness)
    except (TypeError, ValueError):
        # SliceThickness가 없거나 유효하지 않으면 슬라이스 간의 z 위치 차이로 계산 시도
        if len(patient_slices) > 1 and hasattr(patient_slices[0], 'ImagePositionPatient') and hasattr(patient_slices[1], 'ImagePositionPatient'):
            spacing_z = abs(patient_slices[1].ImagePositionPatient[2] - patient_slices[0].ImagePositionPatient[2])
            if spacing_z == 0: # 만약 ImagePositionPatient z값이 같다면 임의의 값 사용
                spacing_z = 1.0 
                print("경고: SliceThickness를 읽을 수 없고 ImagePositionPatient z값 차이가 0이어서 z 간격을 기본값(1.0)으로 설정합니다.")
        else:
            spacing_z = 1.0 # 기본값
            print("경고: SliceThickness를 읽을 수 없어 z 간격을 기본값(1.0)으로 설정합니다.")


    print(f"픽셀 간격 (x, y, z): ({spacing_x:.2f} mm, {spacing_y:.2f} mm, {spacing_z:.2f} mm)")
    
    # 최소값, 최대값 (HU)
    min_val = np.min(pixel_data_3d)
    max_val = np.max(pixel_data_3d)
    print(f"최소 픽셀 값 (HU): {min_val}")
    print(f"최대 픽셀 값 (HU): {max_val}")
    print("--------------------\n")

    # 4. 치아 세그멘테이션 (새로운 단계)
    # 이 임계값은 실제 데이터와 원하는 결과에 따라 미세 조정이 필요합니다.
    # 예를 들어, 치아 에나멜은 +2000 HU 이상일 수 있습니다.
    # 좀 더 넓은 범위로 시작하여 결과를 보고 조절하는 것이 좋습니다.
    # 예: teeth_mask = segment_teeth(pixel_data_3d, hu_threshold_min=700, hu_threshold_max=4000)
    min_finding_hu = max(min_val + (max_val - min_val) * 0.3, 2000)
    max_finding_hu = max_val
    teeth_mask = segment_teeth(pixel_data_3d, hu_threshold_min=min_finding_hu, hu_threshold_max=max_finding_hu) # 최소 HU 값을 높여서 선택 범위 축소

    # 5. VTK 볼륨 렌더링
    print("VTK 볼륨 렌더링을 설정 중입니다...")

    # VTK ImageData 생성
    data_importer = vtk.vtkImageImport()
    data_string = pixel_data_3d.tobytes() # NumPy 배열을 바이트 문자열로 변환
    data_importer.CopyImportVoidPointer(data_string, len(data_string))
    data_importer.SetDataScalarTypeToShort() # int16 데이터이므로
    data_importer.SetNumberOfScalarComponents(1) # 회색조 이미지

    # 이미지 차원 및 간격 설정 (VTK는 z, y, x 순서로 차원을 기대할 수 있음, 또는 x,y,z 확인 필요)
    # 일반적으로 DICOM은 (슬라이스, 행, 열) 순서이므로 VTK에서는 (열, 행, 슬라이스) 순서로 간주.
    # VTK의 SetDataExtent는 (xmin, xmax, ymin, ymax, zmin, zmax)
    # NumPy 배열은 (z, y, x) 로 가정하고 진행
    extent = [0, image_shape[2] - 1, 0, image_shape[1] - 1, 0, image_shape[0] - 1]
    data_importer.SetDataExtent(extent)
    data_importer.SetWholeExtent(extent)
    
    # 픽셀 간격 설정 (VTK는 x, y, z 순서)
    data_importer.SetDataSpacing(spacing_x, spacing_y, spacing_z)
    # 데이터 원점 (일반적으로 DICOM의 ImagePositionPatient 첫번째 슬라이스의 값)
    # 여기서는 단순화를 위해 (0,0,0)으로 설정. 실제 위치 반영시 이 값 사용
    first_slice_origin = patient_slices[0].get('ImagePositionPatient', [0.0, 0.0, 0.0])
    data_importer.SetDataOrigin(first_slice_origin[0], first_slice_origin[1], first_slice_origin[2])
    data_importer.Update()

    vtk_image_data = data_importer.GetOutput()

    # 볼륨 프로퍼티 (색상 및 불투명도 전달 함수)
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOff() # 음영 효과 끄기 (옵션)
    volume_property.SetInterpolationTypeToLinear()

    # 불투명도 전달 함수 (Piecewise Function)
    # HU 값에 따라 불투명도 조절 (예: 뼈는 불투명하게, 공기는 투명하게)
    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(min_val, 0.0) # 최소값 근처는 투명하게
    opacity_transfer_function.AddPoint(0, 0.0)     # 물(0 HU) 근처
    opacity_transfer_function.AddPoint(500, 0.15)  # 연조직 일부
    opacity_transfer_function.AddPoint(1000, 0.3)  # 뼈 영역 시작
    opacity_transfer_function.AddPoint(max_val, 0.5) # 최대값 근처
    volume_property.SetScalarOpacity(opacity_transfer_function)

    # 색상 전달 함수 (Color Transfer Function)
    # HU 값에 따라 색상 조절 (예: 회색조 또는 특정 조직 강조)
    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0) # 검정
    color_transfer_function.AddRGBPoint(0, 0.5, 0.5, 0.5)     # 회색 (물)
    color_transfer_function.AddRGBPoint(500, 0.8, 0.8, 0.8)   # 밝은 회색 (연조직)
    color_transfer_function.AddRGBPoint(1000, 1.0, 1.0, 0.9)  # 약간 노란 흰색 (뼈)
    color_transfer_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0) # 흰색
    volume_property.SetColor(color_transfer_function)
    
    # 그라데이션 불투명도 (경계면 강조) - 옵션
    # volume_property.SetGradientOpacity(gradient_opacity_function)
    # volume_property.SetDisableGradientOpacity(0)


    # 볼륨 매퍼
    # vtkSmartVolumeMapper는 CPU/GPU를 자동으로 선택 시도
    # 또는 vtkGPUVolumeRayCastMapper (GPU 사용 명시), vtkFixedPointVolumeRayCastMapper (CPU)
    volume_mapper = vtk.vtkSmartVolumeMapper()
    # volume_mapper = vtk.vtkGPUVolumeRayCastMapper() # GPU 사용 시
    # if not volume_mapper.IsRenderSupported(vtk_image_data, volume_property):
    #     print("GPU 볼륨 렌더링이 지원되지 않아 CPU 매퍼로 전환합니다.")
    #     volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()

    volume_mapper.SetInputData(vtk_image_data)
    # volume_mapper.SetBlendModeToComposite() # 기본값, 또는 MIP 등

    # 원본 볼륨 액터
    original_volume = vtk.vtkVolume()
    original_volume.SetMapper(volume_mapper)
    original_volume.SetProperty(volume_property)

    # 렌더러, 렌더 윈도우, 인터랙터
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800) # 윈도우 크기 설정
    render_window.SetWindowName("DICOM Volume Rendering")

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    
    # 스타일 설정 (마우스 상호작용)
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    render_window_interactor.SetInteractorStyle(interactor_style)

    renderer.AddVolume(original_volume) # 원본 볼륨 추가

    # 세그멘테이션된 치아 볼륨 렌더링 (옵션)
    if teeth_mask is not None:
        print("세그멘테이션된 치아 마스크를 VTK로 렌더링합니다...")
        # teeth_mask (boolean)를 vtkImageData로 변환 (uint8, 0 또는 255 값)
        teeth_mask_uint8 = teeth_mask.astype(np.uint8) * 255
        
        teeth_data_importer = vtk.vtkImageImport()
        teeth_data_string = teeth_mask_uint8.tobytes()
        teeth_data_importer.CopyImportVoidPointer(teeth_data_string, len(teeth_data_string))
        teeth_data_importer.SetDataScalarTypeToUnsignedChar()
        teeth_data_importer.SetNumberOfScalarComponents(1)
        
        # 원본 이미지와 동일한 차원, 간격, 원점 사용
        teeth_data_importer.SetDataExtent(extent) # extent는 위에서 원본 데이터에 사용된 값
        teeth_data_importer.SetWholeExtent(extent)
        teeth_data_importer.SetDataSpacing(spacing_x, spacing_y, spacing_z) # 위에서 계산된 값
        teeth_data_importer.SetDataOrigin(first_slice_origin[0], first_slice_origin[1], first_slice_origin[2]) # 위에서 계산된 값
        teeth_data_importer.Update()
        
        segmented_vtk_image_data = teeth_data_importer.GetOutput()

        # 세그멘테이션된 치아 볼륨 프로퍼티 (예: 밝은 단색)
        teeth_volume_property = vtk.vtkVolumeProperty()
        teeth_volume_property.ShadeOff()
        teeth_volume_property.SetInterpolationTypeToLinear()

        teeth_opacity_tf = vtk.vtkPiecewiseFunction()
        teeth_opacity_tf.AddPoint(0, 0.0)    # 배경은 투명하게
        teeth_opacity_tf.AddPoint(255, 0.8)  # 치아는 불투명하게 (0.3~0.8 사이 값으로 조절)
        teeth_volume_property.SetScalarOpacity(teeth_opacity_tf)

        teeth_color_tf = vtk.vtkColorTransferFunction()
        teeth_color_tf.AddRGBPoint(0, 0.0, 0.0, 0.0)      # 배경 (검정/투명)
        teeth_color_tf.AddRGBPoint(255, 0.9, 0.9, 0.2)  # 치아 색상 (예: 밝은 노란색/아이보리색)
        teeth_volume_property.SetColor(teeth_color_tf)

        teeth_volume_mapper = vtk.vtkSmartVolumeMapper()
        teeth_volume_mapper.SetInputData(segmented_vtk_image_data)
        # teeth_volume_mapper.SetBlendModeToComposite()

        teeth_volume = vtk.vtkVolume()
        teeth_volume.SetMapper(teeth_volume_mapper)
        teeth_volume.SetProperty(teeth_volume_property)
        
        renderer.AddVolume(teeth_volume) # 세그멘테이션된 치아 볼륨 추가

    renderer.SetBackground(0.1, 0.2, 0.3) # 배경색 (어두운 파란색 계열)
    renderer.ResetCamera() # 카메라 초기화

    # 카메라 조정 (옵션: 볼륨이 잘 보이도록)
    # camera = renderer.GetActiveCamera()
    # camera.Azimuth(45)
    # camera.Elevation(30)
    # camera.Dolly(1.5) # Zoom out
    # renderer.ResetCameraClippingRange()


    print("렌더링 창을 시작합니다. 창을 닫으면 프로그램이 종료됩니다.")
    render_window.Render()
    render_window_interactor.Start()

if __name__ == '__main__':
    main()
