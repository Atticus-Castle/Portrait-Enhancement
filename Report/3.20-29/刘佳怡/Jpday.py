import cv2
import numpy as np

def japanese_film_filter(img_path, intensity=0.8,grain_amount=0.09):

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"false,cant read {img_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. 提亮（gamma校正）
    gamma = 0.7
    brightened = np.power(img / 255.0, gamma) * 255.0
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    
    # 2. 降低饱和度
    hsv = cv2.cvtColor(brightened, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.7
    desaturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 3. 偏青/蓝色调
    r, g, b = cv2.split(desaturated)
    b = cv2.addWeighted(b, 0.95, np.zeros_like(b), 0, 5)
    r = cv2.addWeighted(r, 0.9, np.zeros_like(r), 0, 0)
    g = cv2.addWeighted(g, 0.95, np.zeros_like(g), 0, 0)
    cyan_tone = cv2.merge([r, g, b])
    cyan_tone = np.clip(cyan_tone, 0, 255).astype(np.uint8)
    
    # 4. 柔光效果
    blurred = cv2.GaussianBlur(cyan_tone, (15, 15), 0)
    soft = cv2.addWeighted(cyan_tone, 0.90, blurred, 0.15, 0)

    noise = np.random.randn(*soft.shape) * 255 * grain_amount
    grain = soft.astype(np.float32) + noise
    grain = np.clip(grain, 0, 255).astype(np.uint8)
    
    # 5. 混合原图和滤镜
    final = cv2.addWeighted(img, 1 - intensity, soft, intensity, 0)
    
    return cv2.cvtColor(final, cv2.COLOR_RGB2BGR)


img_path = "D:/testFile/Hida.jpg"
# 运行滤镜
result = japanese_film_filter(img_path, intensity=0.8, grain_amount=0.09)

import os

if result is not None:
   
    output_path = "D:/testFile/filtfujier_result.jpg"  
    cv2.imwrite(output_path, result)
    
    print(f"Saved to: {output_path}")
    
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()