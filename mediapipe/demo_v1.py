"""
Phase 1 Demo: ResNet-18 + Sign MNIST
demo_v1.py — 纯图像分类，无 MediaPipe

把手放进绿框里，模型直接对框内区域分类。
这一版会暴露 domain gap 问题（背景/光照干扰）。

用法:
    python demo_v1.py

按键:
    q  →  退出
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torchvision.transforms as T

# ══════════════════════════════════════════
#  Config
# ══════════════════════════════════════════
CKPT_PATH      = './checkpoint/ckpt_resnet18.pt'
DEVICE         = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
CONF_THRESHOLD = 0.6
# 固定 ROI 框：(x1, y1, x2, y2)，居中偏右留出显示空间
BOX = (300, 100, 580, 380)


# ══════════════════════════════════════════
#  Model（结构和 train.py 完全一致）
# ══════════════════════════════════════════
def build_model(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    return m


# ══════════════════════════════════════════
#  Preprocessing（和 train.py val_tf 一致）
# ══════════════════════════════════════════
preprocess = T.Compose([
    T.ToTensor(),                  # uint8 (H,W) → float (1,H,W) [0,1]
    T.Resize((28, 28)),
    T.Normalize((0.5,), (0.5,)),   # → [-1, 1]
])


# ══════════════════════════════════════════
#  Main
# ══════════════════════════════════════════
def main():
    # ── 1. 加载模型（增加 try-except 防止模型加载失败） ──
    try:
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        idx_to_char = ckpt['idx_to_char']
        model = build_model(len(idx_to_char)).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f'[模型] 加载成功: val_acc={ckpt["best_val_acc"]:.4%}')
    except Exception as e:
        print(f'[错误] 模型加载失败: {e}')
        return

    # ── 2. 摄像头初始化（去掉特定的 Backend 参数） ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[错误] 无法打开摄像头')
        return

    print('[提示] 正在启动预览，按 q 退出...')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ── 3. 动态计算 ROI（不再硬编码，取画面中心偏右的一块） ──
        # 即使摄像头分辨率变了，这里也不会报错
        x1, y1 = int(w * 0.5), int(h * 0.2)
        x2, y2 = int(w * 0.9), int(h * 0.7)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        try:
            # ── 4. 推理过程 ──
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            inp = preprocess(gray).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(inp)
                probs = torch.softmax(output, dim=1)[0]
                conf, idx = probs.max(0)
                pred_conf = conf.item()
                pred_char = idx_to_char[idx.item()] if pred_conf >= CONF_THRESHOLD else '?'
        except Exception as e:
            # 如果推理出错（比如 MPS 兼容性问题），至少保证画面能出来
            pred_char, pred_conf = "Error", 0.0

        # ── 5. 绘制（确保坐标不会出框） ──
        color = (0, 220, 0) if pred_conf >= CONF_THRESHOLD else (0, 100, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 结果文本
        display_text = f"{pred_char} ({pred_conf:.0%})"
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ── 6. 强制刷新窗口 ──
        cv2.imshow('Sign Language Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # 额外加一个 cv2.waitKey 确保窗口关闭
    for i in range(5):
        cv2.waitKey(1)

if __name__ == '__main__':
    main()