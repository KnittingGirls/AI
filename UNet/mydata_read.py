import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 경로 설정
dir_data = './datasets'  # 데이터셋 최상위 디렉토리
dir_label = os.path.join(dir_data, 'labels')  # 라벨 이미지 폴더
dir_input = os.path.join(dir_data, 'inputs')  # 입력 이미지 폴더

## 저장할 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 데이터 불러오기 및 매칭
files_input = sorted([f for f in os.listdir(dir_input) if f.endswith('.jpg')])  # 입력 파일 목록
files_label = sorted([f for f in os.listdir(dir_label) if f.startswith('annotated_') and f.endswith('.jpg')])  # 라벨 파일 목록

# 파일 이름에서 숫자만 추출하여 매칭
input_ids = {os.path.splitext(f)[0]: f for f in files_input}  # {id: filename}
label_ids = {os.path.splitext(f)[0].replace('annotated_', ''): f for f in files_label}  # {id: filename}

# 매칭된 파일 리스트 생성
matched_files = [(input_ids[key], label_ids[key]) for key in input_ids.keys() if key in label_ids]

assert len(matched_files) > 0, "입력과 라벨 데이터의 매칭이 없습니다!"

nframe = len(matched_files)

## 훈련, 검증, 테스트 데이터 비율 설정
nframe_train = int(nframe * 0.8)  # 80% 훈련 데이터
nframe_val = int(nframe * 0.1)    # 10% 검증 데이터
nframe_test = nframe - nframe_train - nframe_val  # 나머지 10% 테스트 데이터

## 데이터 셔플
np.random.shuffle(matched_files)

## 데이터를 .npy 파일로 저장
offset_nframe = 0

# Train 데이터 저장
for i in range(nframe_train):
    input_file, label_file = matched_files[i + offset_nframe]
    img_label = Image.open(os.path.join(dir_label, label_file)).convert('L')
    img_input = Image.open(os.path.join(dir_input, input_file)).convert('L')

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# Val 데이터 저장
offset_nframe += nframe_train
for i in range(nframe_val):
    input_file, label_file = matched_files[i + offset_nframe]
    img_label = Image.open(os.path.join(dir_label, label_file)).convert('L')
    img_input = Image.open(os.path.join(dir_input, input_file)).convert('L')

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# Test 데이터 저장
offset_nframe += nframe_val
for i in range(nframe_test):
    input_file, label_file = matched_files[i + offset_nframe]
    img_label = Image.open(os.path.join(dir_label, label_file)).convert('L')
    img_input = Image.open(os.path.join(dir_input, input_file)).convert('L')

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 마지막 이미지를 확인
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()
