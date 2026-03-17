import sys
sys.path.insert(0, '.')
from yolo_transformer import YOLOTransformerLowLight
import torch
import cv2

# 创建模型
model = YOLOTransformerLowLight('yolo26n.pt')
state_dict = torch.load('runs/train/test_train_1773657540/best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=False)

# 测试图像
image = cv2.imread('sample_data/high_res/000000001237.jpg')
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

# 测试 forward
with torch.no_grad():
    outputs = model(image_tensor, None, is_training=False)

print('输出形状:', outputs.shape)
print('输出类型:', type(outputs))
print('输出最大值:', outputs.max().item())
print('输出最小值:', outputs.min().item())

# 测试后处理
results = model.custom_head.postprocess(outputs, (image.shape[0], image.shape[1]), conf_thres=0.5, iou_thres=0.7)
print('后处理结果:', results)
if len(results) > 0:
    print('检测框数量:', len(results[0]))
    # 打印前 10 个检测结果
    if len(results[0]) > 0:
        print('前 10 个检测结果:')
        for i, box in enumerate(results[0][:10]):
            print(f'  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}], conf={box[4]:.3f}, class={box[5]:.0f}')
