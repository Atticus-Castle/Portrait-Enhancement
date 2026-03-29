
# import torch
# from model import MattingNetwork
# from inference import convert_video
# model = MattingNetwork('mobilenetv3').eval().cuda()
# model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
# convert_video(
#     model,
#     input_source='test_images',   # ⭐ 改这里
#     output_type='png_sequence',      # ⭐ 推荐输出图片
#     output_composition='com',        # 输出文件夹
#     output_alpha='pha',
#     output_foreground='fgr',
#     downsample_ratio=None
# )

import os
import torch
from model import MattingNetwork
from inference import convert_video

# ====== 1. 输入路径（必须是绝对路径） ======
input_path = 'test_images'   # ⭐ 改成你的

# ====== 2. 输出路径（你已有的文件夹） ======
output_com = 'com'
output_pha = 'pha'
output_fgr = 'fgr'

# ====== 3. 检查路径 ======
print("Input exists:", os.path.exists(input_path))
print("Images:", os.listdir(input_path))

# ====== 4. 加载模型 ======
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MattingNetwork('mobilenetv3').eval().to(device)

model.load_state_dict(torch.load('rvm_mobilenetv3.pth', map_location=device))

print("Model loaded on:", device)

# ====== 5. 执行 ======
convert_video(
    model,
    input_source=input_path,
    output_type='png_sequence',

    output_composition=output_com,
    output_alpha=output_pha,
    output_foreground=output_fgr,

    downsample_ratio=0.25
)

print("Done!")