import cv2
import numpy as np

def night_softglow_grain(img_path, intensity=0.6, glow_amount=0.2, grain_amount=0.05, contrast_boost=1.15):
    """
    夜景柔光晕染 + 胶片颗粒
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read image {img_path}")
        return None
    
    original = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. 提亮暗部
    gamma = 0.85
    img = np.power(img / 255.0, gamma) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 增强对比度
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.addWeighted(l, contrast_boost, np.zeros_like(l), 0, 0)
    l = np.clip(l, 0, 255).astype(np.uint8)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 2. 柔光晕染（blur_size 必须是奇数）
    blur_size = 5  # 👈 改成奇数：5, 7, 9, 11...
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    glow = cv2.addWeighted(img, 1 - glow_amount, blurred, glow_amount, 0)
    
    # 4. 添加胶片颗粒
    noise = np.random.randn(*glow.shape) * 255 * grain_amount
    grain = glow.astype(np.float32) + noise
    grain = np.clip(grain, 0, 255).astype(np.uint8)
    
    # 5. 轻微提升饱和度
    hsv = cv2.cvtColor(grain, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.8
    grain = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 6. 混合原图
    final = cv2.addWeighted(original, 1 - intensity, cv2.cvtColor(grain, cv2.COLOR_RGB2BGR), intensity, 0)
    
    return final


# 使用
img_path = "D:/testFile/Shinjuku.jpg"
result = night_softglow_grain(img_path, intensity=0.6, glow_amount=0.2, grain_amount=0.05, contrast_boost=1.15)
cv2.imwrite("D:/testFile/night_softglow_grain.jpg", result)
print("Done!")