import cv2
import numpy as np

def fuji_film_filter(img_path, intensity=0.8):
    """
    富士胶片风格滤镜
    参数映射：
    - 曝光 -10 → 整体变暗
    - 鲜明度 +10 → 对比度增强
    - 高光 +20 → 高光区域提亮
    - 阴影 -15 → 暗部压暗
    - 亮度 +8 → 整体微亮
    - 黑点 +10 → 黑色更深
    - 自然饱和度 -8 → 饱和度降低
    - 色温 -15 → 偏冷（蓝色调）
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read image {img_path}")
        return None
    
    original = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ===== 1. 曝光 -10 + 亮度 +8 = 整体微暗 =====
    # 曝光降低 = gamma > 1 变暗
    gamma = 1.25  # 曝光-10
    brightened = np.power(img / 255.0, gamma) * 255.0
    
    # 亮度+8（轻微补偿）
    brightened = brightened - 2
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    
    # ===== 2. 鲜明度 +10 = 增强对比度 =====
    lab = cv2.cvtColor(brightened, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.addWeighted(l, 1.25, np.zeros_like(l), 0, 0)  # 对比度+10%
    l = np.clip(l, 0, 255).astype(np.uint8)
    lab = cv2.merge([l, a, b])
    contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # ===== 3. 高光 +20 = 提亮高光区域 =====
    hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(contrast_enhanced, cv2.COLOR_RGB2GRAY)
    highlight_mask = gray > 200  # 高光区域
    hsv[highlight_mask, 2] = np.clip(hsv[highlight_mask, 2] * 1.2, 0, 255)
    highlight_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # ===== 4. 阴影 -15 = 压暗暗部 =====
    gray2 = cv2.cvtColor(highlight_enhanced, cv2.COLOR_RGB2GRAY)
    shadow_mask = gray2 < 80  # 阴影区域
    shadow_enhanced = highlight_enhanced.astype(np.float32)
    shadow_enhanced[shadow_mask] = shadow_enhanced[shadow_mask] * 0.85
    shadow_enhanced = np.clip(shadow_enhanced, 0, 255).astype(np.uint8)
    
    # ===== 5. 黑点 +10 = 黑色更深 =====
    gray3 = cv2.cvtColor(shadow_enhanced, cv2.COLOR_RGB2GRAY)
    black_mask = gray3 < 30  # 极暗区域
    black_enhanced = shadow_enhanced.astype(np.float32)
    black_enhanced[black_mask] = black_enhanced[black_mask] * 0.8
    black_enhanced = np.clip(black_enhanced, 0, 255).astype(np.uint8)
    
    # ===== 6. 自然饱和度 -8 = 降低饱和度 =====
    hsv2 = cv2.cvtColor(black_enhanced, cv2.COLOR_RGB2HSV)
    hsv2[:, :, 1] = hsv2[:, :, 1] * 1.08  # 饱和度降低8%
    desaturated = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
    
    # ===== 7. 色温 -15 = 偏冷（蓝色调）=====
    r, g, b = cv2.split(desaturated)
    b = cv2.addWeighted(b, 1.10, np.zeros_like(b), 0, 5)   # 蓝色增强
    r = cv2.addWeighted(r, 0.95, np.zeros_like(r), 0, 0)  # 红色减弱
    g = cv2.addWeighted(g, 0.95, np.zeros_like(g), 0, 0)  # 👈 绿色系数（新增）
    cool_tone = cv2.merge([r, g, b])

    gamma_dark = 1.10
    cool_tone = np.power(cool_tone / 255.0, gamma_dark) * 255.0
    cool_tone = np.clip(cool_tone, 0, 255).astype(np.uint8)
    
    # ===== 8. 添加轻微颗粒感（富士胶片特色）=====
    grain_amount = 0.03
    noise = np.random.randn(*cool_tone.shape) * 255 * grain_amount
    grain = cool_tone.astype(np.float32) + noise
    grain = np.clip(grain, 0, 255).astype(np.uint8)
    
    # ===== 9. 混合原图和滤镜 =====
    final = cv2.addWeighted(original, 1 - intensity, cv2.cvtColor(grain, cv2.COLOR_RGB2BGR), intensity, 0)
    
    return final


# 使用
img_path = "D:/testFile/xyf.jpg"
result = fuji_film_filter(img_path, intensity=0.8)

if result is not None:
    output_path = "D:/testFile/result.jpg"
    cv2.imwrite(output_path, result)
    print(f"Saved to: {output_path}")
    
    cv2.imshow("Fuji Film Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()