import os
import shutil

def rename_and_move_images(source_folder, destination_folder):
    try:
        # 대상 폴더가 없으면 생성
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # 소스 폴더 내 파일 목록 가져오기
        files = os.listdir(source_folder)
        
        # 파일 순회
        for file_name in files:
            # annotated_로 시작하고 .jpg로 끝나는 파일만 처리
            if file_name.startswith("annotated_") and file_name.endswith(".jpg"):
                # 숫자 부분 추출
                number_part = file_name.replace("annotated_", "").replace(".jpg", "")
                
                # 새 파일 이름 생성
                new_file_name = f"{number_part}.jpg"
                
                # 기존 파일 경로와 새 파일 경로 생성
                old_file_path = os.path.join(source_folder, file_name)
                new_file_path = os.path.join(destination_folder, new_file_name)
                
                # 파일 이동 및 이름 변경
                shutil.copy2(old_file_path, new_file_path)  # 원본 파일 복사
                print(f"Copied and renamed: {file_name} -> {new_file_name}")
        
        print("모든 파일 이름 변경 및 이동 완료.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 소스 폴더 경로와 대상 폴더 경로 설정
source_folder_path = "./datasets/labels"  # 원본 폴더 경로로 변경
destination_folder_path = "./datasets/labels_rename"  # 대상 폴더 경로로 변경

rename_and_move_images(source_folder_path, destination_folder_path)
