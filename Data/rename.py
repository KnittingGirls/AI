import os

def rename_images_from_scratch(folder_path, prefix="image"):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_files.sort()

    temp_prefix = "__temp__"
    temp_paths = []

    for filename in image_files:
        old_path = os.path.join(folder_path, filename)
        temp_filename = f"{temp_prefix}_{filename}"
        temp_path = os.path.join(folder_path, temp_filename)
        
        os.rename(old_path, temp_path)
        temp_paths.append(temp_path)  

    for i, temp_path in enumerate(temp_paths, start=1):
        new_filename = f"{prefix}_{i}.jpg"
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(temp_path, new_path)
        print(f"{temp_path} -> {new_path}")

folder_path = "./pinterest_pattern"  # 이미지 저장 폴더 경로
rename_images_from_scratch(folder_path)
