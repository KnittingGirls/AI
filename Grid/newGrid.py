import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

input_folder = '../data/ajour_label/'
output_folder = '../data/sort_result/'
os.makedirs(output_folder, exist_ok=True)  

scale_factor = 4  # 축소 비율 (4x4 블록을 하나로 표현)

def rgb_to_symbol(rgb):
    mapping = {
        (0, 0, 0): None,
        (255, 0, 0): '※',  # 빨강 - rib
        (0, 255, 0): '◇',  # 초록 - purl
        (255, 20, 147): '⊗',  # 핑크 - moss
        (255, 255, 0): '♣',  # 노란 - ajour
        (0, 255, 255): '≪',  # 하늘 - single_jersy
    }
    for key in mapping:
        if np.allclose(rgb, key, atol=30):  # 유사한 색상 허용
            return mapping[key]
    return '.'  # 지금은 경계선을 구분하는 셈

def load_pixel_data_from_image(file_path): # numpy 배열로 반환
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error parsing file {file_path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 데이터 축소
def downscale_pixels(pixel_data, scale_factor):
    new_size = (pixel_data.shape[1] // scale_factor, pixel_data.shape[0] // scale_factor)
    return cv2.resize(pixel_data, new_size, interpolation=cv2.INTER_NEAREST)

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith('.jpg'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        
        pixel_data = load_pixel_data_from_image(input_path)
        if pixel_data is not None:
            scaled_pixel_data = downscale_pixels(pixel_data, scale_factor)
            scaled_rows, scaled_cols = scaled_pixel_data.shape[:2]
            
            fig, ax = plt.subplots(figsize=(scaled_cols / 5, scaled_rows / 5))
            
            # 그리드 라인 그리기
            for x in range(scaled_cols + 1):
                ax.plot([x, x], [0, scaled_rows], color='black', linewidth=0.5)
            for y in range(scaled_rows + 1):
                ax.plot([0, scaled_cols], [y, y], color='black', linewidth=0.5)
            
            # 각 셀에 기호 표시
            for y in range(scaled_rows):
                for x in range(scaled_cols):
                    rgb = scaled_pixel_data[y, x]
                    symbol = rgb_to_symbol(rgb)
                    if symbol:
                        ax.text(x + 0.5, scaled_rows - y - 0.5, symbol, fontsize=8, ha='center', va='center')
            
            ax.axis('off')
            ax.set_xlim(0, scaled_cols)
            ax.set_ylim(0, scaled_rows)
            plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight')
            plt.close(fig)  