import matplotlib.pyplot as plt
import numpy as np

rows, cols = 200, 200 # 그리드 사이즈 160x160
plt.figure(figsize=(20, 20)) # 전체 사이즈

for x in range(cols + 1):
    plt.plot([x, x], [0, rows], color='black', linewidth=0.5)
for y in range(rows + 1):
    plt.plot([0, cols], [y, y], color='black', linewidth=0.5)

# 패턴 인식 -> 표현 
# 좌표마다의 색상과 패턴값을 저장
filled_cells = [(1,2), (0,2)] # 여기에다가 패턴 인식 후 좌표 대입

# 색상패턴 표시 
for (y, x) in filled_cells:
    plt.fill([x, x+1, x+1, x], [rows-y-1, rows-y-1, rows-y, rows-y], color='gray', hatch='//') # 좌표마다 색상표시 

# 뜨개방식 표시 

plt.axis('on') # 축의 숫자 표시
plt.xlim(0, cols) # 그래프 테두리  
plt.ylim(0, rows)

plt.show()