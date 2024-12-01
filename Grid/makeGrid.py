import os  
import matplotlib.pyplot as plt
import ast  
import numpy as np  

# 그리드 설정
rows, cols = 250, 250  # 그리드 사이즈
scale_factor = 4  # 축소 비율 (4x4 블록을 하나로 표현)
scaled_rows, scaled_cols = rows // scale_factor, cols // scale_factor  # 축소된 그리드 사이즈

# RGB 값에 따른 기호 매핑
def rgb_to_symbol(rgb):
    if np.array_equal(rgb, [0, 0, 0]): 
        return None
    elif np.array_equal(rgb, [0, 0, 255]):
        return '※'
    elif np.array_equal(rgb, [255, 255, 0]):
        return '≫'
    else:  # 그 외 색상
        return '.'

def load_pixel_data_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return np.array(ast.literal_eval(file.read().strip()))  # numpy 배열로 반환
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

# 데이터 축소 (scale_factor x scale_factor 블록을 1개로 축소!)
def downscale_pixels(pixel_data, scale_factor):
    new_rows, new_cols = pixel_data.shape[0] // scale_factor, pixel_data.shape[1] // scale_factor
    downscaled = []
    for i in range(new_rows):
        row = []
        for j in range(new_cols):
            block = pixel_data[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor]  # scale_factor x scale_factor 블록 추출
            avg_color = np.mean(block, axis=(0, 1)).astype(int).tolist()  # RGB 평균값 계산
            row.append(avg_color)
        downscaled.append(row)
    return np.array(downscaled)

# 특정 폴더 내의 파일 처리 함수
def process_folder(input_folder, output_folder, scale_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'): 
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_grid.jpg")

            # 픽셀 데이터 로드 및 축소
            pixel_data = load_pixel_data_from_file(input_path)
            if pixel_data is not None:
                scaled_pixel_data = downscale_pixels(pixel_data, scale_factor)  # 데이터 축소

                fig, ax = plt.subplots(figsize=(20, 20))

                for x in range(scaled_cols + 1):
                    ax.plot([x, x], [0, scaled_rows], color='black', linewidth=0.5)
                for y in range(scaled_rows + 1):
                    ax.plot([0, scaled_cols], [y, y], color='black', linewidth=0.5)

                for y in range(scaled_pixel_data.shape[0]):
                    for x in range(scaled_pixel_data.shape[1]):
                        rgb = scaled_pixel_data[y, x]
                        symbol = rgb_to_symbol(rgb)
                        if symbol:  
                            ax.text(x + 0.5, scaled_rows - y - 0.5, symbol, fontsize=8, ha='center', va='center')

                # 축 설정
                ax.axis('off')  # 축 숨기기
                ax.set_xlim(0, scaled_cols)
                ax.set_ylim(0, scaled_rows)

                # 이미지 저장
                plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight')
                print(f"Grid image saved to {output_path}")
                # plt.show()
                plt.close(fig)
            else:
                print(f"Skipping file {filename}: No valid pixel data found.")

input_folder = '/content/drive/MyDrive/start-knitting-girls/UNet/pred_pixels'
output_folder = '/content/drive/MyDrive/start-knitting-girls/UNet/Grid'

process_folder(input_folder, output_folder, scale_factor)
