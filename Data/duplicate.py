import os
import hashlib
from PIL import Image

def calculate_image_hash(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB') 
        hash_obj = hashlib.sha1(img.tobytes())
    return hash_obj.hexdigest()

def remove_duplicate_images(folder_path):
    image_hashes = {}
    deleted_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                image_hash = calculate_image_hash(file_path)
                if image_hash in image_hashes:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"삭제된 중복 이미지: {file_path}")
                else:
                    image_hashes[image_hash] = filename
            except Exception as e:
                print(f"오류가 발생한 파일: {file_path} - {e}")
                
    print(f"총 {deleted_count}개의 중복 이미지 삭제")

def rename_images_sequentially(folder_path, prefix="image"):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_files.sort() 
    
    for i, filename in enumerate(image_files, start=1):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"{prefix}_{i}.jpg"
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)

folder_path = "./pinterest" # 이미지 저장 폴더 경로 

remove_duplicate_images(folder_path)
rename_images_sequentially(folder_path)
