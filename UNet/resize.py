import os
from PIL import Image
import glob

def crop_and_resize(image_path, output_size):
    """
    이미지를 중앙 기준으로 정사각형 크롭 후 지정된 크기로 리사이즈합니다.
    
    Parameters:
    - image_path (str): 입력 이미지 경로
    - output_size (int): 리사이즈할 정사각형 크기 (예: 256)
    
    Returns:
    - PIL.Image: 크롭 및 리사이즈된 이미지
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # 정사각형 크기: 가로와 세로 중 작은 값
    square_size = min(width, height)
    
    # 중앙 기준으로 크롭 좌표 계산
    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size
    
    # 이미지 크롭
    cropped_img = img.crop((left, top, right, bottom))
    
    # 크롭된 이미지를 지정된 크기로 리사이즈
    resized_img = cropped_img.resize((output_size, output_size), resample=Image.Resampling.LANCZOS)
    return resized_img

def process_images_in_folder(folder_path, output_size):
    """
    폴더에서 이미지를 불러와 정사각형 크롭 후 리사이즈하고 저장합니다.
    
    Parameters:
    - folder_path (str): 이미지 폴더 경로
    - output_size (int): 리사이즈할 정사각형 크기 (예: 256)
    """
    # 이미지 파일 경로 검색 (지원하는 확장자)
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.gif', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print("이미지가 없습니다. 폴더 경로를 확인해주세요.")
        return
    
    for image_path in image_files:
        try:
            # 정사각형 크롭 및 리사이즈 수행
            resized_img = crop_and_resize(image_path, output_size)
            
            # 리사이즈된 이미지를 동일 경로에 저장
            resized_img.save(image_path)
            print(f"Processed and saved: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# 실행 예시
if __name__ == "__main__":
    input_folder = "./datasets/labels"  # 예: "/home/image_dir"
    output_size = 256  # 출력 이미지의 정사각형 크기 (예: 256x256)
    process_images_in_folder(input_folder, output_size)
