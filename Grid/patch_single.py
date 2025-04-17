# 자영이가 백엔드에서 deeplab 결과를 보낼 때 

# python convert_grid.py \
#   /path/to/annotated_1.png \
#   /path/to/output/grid_1.png \
#   --scale 4 


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 기호별 3x3 패턴
symbol_patterns = {
    'S': np.array([ # single jersey - 메리야스 뜨기
        ['<', ' ', '<'],
        [' ', '<', ' '],
        ['<', ' ', '<']
    ]),
    'P': np.array([ # purl - 가터뜨기
        ['―', '―', '―'],
        ['|', '|', '|'],
        ['―', '―', '―']
    ]),
    'R': np.array([ # rib - 고무뜨기
        ['|', '|', '|'],
        ['|', '|', '|'],
        ['|', '|', '|']
    ]),
    'A': np.array([ # ajour
        [' ', ' ', ' '],
        [' ', 'O', ' '],
        [' ', ' ', ' ']
    ]),
    'M': np.array([ # moss - 멍석뜨기
        ['―', '|', '―'],
        ['|', '―', '|'],
        ['―', '|', '―']
    ]),
    '.': np.array([ # 경계/기타
        ['.', ' ', '.'],
        [' ', '.', ' '],
        ['.', ' ', '.']
    ])
}

# RGB 값(라벨)에 따른 기호 매핑
def rgb_to_symbol(rgb):
    mapping = {
        (0, 255, 255): 'S',    # 하늘
        (0, 255, 0):   'P',    # 초록
        (255, 0, 0):   'R',    # 빨강
        (255, 255, 0): 'A',    # 노랑
        (255, 20, 147):'M',    # 핑크
    }
    for key, sym in mapping.items():
        if np.allclose(rgb, key, atol=30):
            return sym
    return '.'  # 매핑없음

vectorized_rgb_to_symbol = np.vectorize(rgb_to_symbol, signature='(n)->()')

def process_image(input_path, output_path, scale_factor=4):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {input_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Down-Scaling
    h, w = image.shape[:2]
    down = cv2.resize(image, (w//scale_factor, h//scale_factor), interpolation=cv2.INTER_LINEAR)

    symbol_grid = vectorized_rgb_to_symbol(down.reshape(-1, 3)).reshape(down.shape[:2])
    rows, cols = symbol_grid.shape
    final_grid = np.full((rows*3, cols*3), ' ', dtype='<U1')

    # 3x3 패턴 매핑
    for y in range(rows):
        for x in range(cols):
            pat = symbol_patterns.get(symbol_grid[y, x], symbol_patterns['.'])
            final_grid[y*3:(y+1)*3, x*3:(x+1)*3] = pat

    fig = plt.figure(figsize=(cols/10, rows/10), dpi=300)
    ax = fig.add_axes([0,0,1,1], frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(np.zeros((rows*3, cols*3)), cmap='gray', vmin=0, vmax=1)
    for yy in range(final_grid.shape[0]):
        for xx in range(final_grid.shape[1]):
            ax.text(xx, yy, final_grid[yy, xx], fontsize=3,
                    ha='center', va='center', color='white')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# 실행하는 부분 수정 필요
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="라벨링 이미지를 grid 문자 패턴 이미지로 변환"
    )
    parser.add_argument("input_file", help="원본 라벨 이미지 파일 경로")
    parser.add_argument("output_file", help="저장할 출력 이미지 파일 경로")
    parser.add_argument("--scale", type=int, default=4,
                        help="다운스케일 비율 (기본: 4)")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {args.input_file} → {args.output_file} (scale={args.scale})")
    process_image(args.input_file, args.output_file, scale_factor=args.scale)
    print("완료")
