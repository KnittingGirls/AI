import matplotlib.pyplot as plt
import ast  
import numpy as np  

rows, cols = 250, 250  # 원본 그리드 사이즈
scale_factor = 4  # 축소 비율 (4x4 블록을 하나로 표현)
scaled_rows, scaled_cols = rows // scale_factor, cols // scale_factor  # 축소된 그리드 사이즈

file_path = './data/input/sample.txt'
output_path = './data/output/sample_grid.jpg'

# RGB 값(라벨)에 따른 기호 매핑
def rgb_to_symbol(rgb):
    if np.array_equal(rgb, [0, 0, 0]):  
        return None
    elif np.array_equal(rgb, [255, 0, 0]):
        return '※'
    elif np.array_equal(rgb, [0, 255, 0]):
        return '◇'
    elif np.array_equal(rgb, [0, 0, 255]):
        return '∝'
    elif np.array_equal(rgb, [255, 255, 0]):
        return '≫'
    elif np.array_equal(rgb, [0, 255, 255]):
        return '≪'
    elif np.array_equal(rgb, [255, 0, 255]):
        return '⊗'
    else: 
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
            block = pixel_data[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor]  
            avg_color = np.mean(block, axis=(0, 1)).astype(int).tolist()  # RGB 평균값 계산
            row.append(avg_color)
        downscaled.append(row)
    return np.array(downscaled)

pixel_data = load_pixel_data_from_file(file_path)
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
    ax.axis('off')
    ax.set_xlim(0, scaled_cols)
    ax.set_ylim(0, scaled_rows)

    plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
