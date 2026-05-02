"""
Phase 1: Sign MNIST + ResNet-18
train.py  (从你的 notebook 精简而来)

数据目录:
    ./data/sign_mnist_train.csv
    ./data/sign_mnist_test.csv
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

# ══════════════════════════════════════════
#  Config（直接改这里）
# ══════════════════════════════════════════
TRAIN_CSV  = './data/sign_mnist_train.csv'
TEST_CSV   = './data/sign_mnist_test.csv'
SAVE_PATH  = './checkpoint/ckpt_resnet18.pt'

DEVICE     = torch.device(
    'cuda'  if torch.cuda.is_available()         else
    'mps'   if torch.backends.mps.is_available() else
    'cpu'
)
EPOCHS      = 10
BATCH_SIZE  = 64
LR          = 1e-3
NUM_CLASSES = 24

# ── 标签映射（demo.py 也要用这个，保持一致） ──
# Sign MNIST: label 0-25，缺 9(J) 和 25(Z)，共 24 类
_ORIG_LABELS = sorted(set(range(26)) - {9, 25})          # [0..8, 10..24]
IDX_TO_CHAR  = {i: chr(ord('A') + lbl) for i, lbl in enumerate(_ORIG_LABELS)}
LABEL_MAP    = {orig: new for new, orig in enumerate(_ORIG_LABELS)}

# print(f'Device: {DEVICE}')
# print(f'Classes: {[IDX_TO_CHAR[i] for i in range(NUM_CLASSES)]}')


# ══════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════
class SignMNIST(Dataset):
    def __init__(self, csv_path: str, transform=None):
        df = pd.read_csv(csv_path)

        # pixels → (N, 1, 28, 28), float32, [0, 1]
        pixels = df.drop('label', axis=1).values.astype(np.float32) / 255.0
        self.images = torch.tensor(pixels).reshape(-1, 1, 28, 28)

        # 标签 remap 到 0-23
        raw_labels = df['label'].tolist()
        self.labels = torch.tensor([LABEL_MAP[l] for l in raw_labels], dtype=torch.long)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ══════════════════════════════════════════
#  Transforms
# ══════════════════════════════════════════
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.Normalize((0.5,), (0.5,)),
])
val_tf = T.Normalize((0.5,), (0.5,))


# ══════════════════════════════════════════
#  Model
# ══════════════════════════════════════════
def get_model() -> nn.Module:
    """
    ResNet-18，适配 28×28 单通道输入：
    - conv1: 7×7 stride-2  →  3×3 stride-1（保留空间信息）
    - maxpool → Identity（28×28 不需要额外下采样）
    - fc → 24 类
    从头训练（weights=None），因为单通道和 ImageNet 域差别大
    """
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


# ══════════════════════════════════════════
#  Train / Eval
# ══════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type,
                            dtype=torch.float16, enabled=(DEVICE.type == 'cuda')):
            out  = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        n        += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        n        += x.size(0)
    return loss_sum / n, correct / n


# ══════════════════════════════════════════
#  Main
# ══════════════════════════════════════════
if __name__ == '__main__':
    # ── 数据 ──
    print(f'Device: {DEVICE}')
    train_ds = SignMNIST(TRAIN_CSV, transform=train_tf)
    val_ds   = SignMNIST(TEST_CSV,  transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
    print(f'Train: {len(train_ds):,}  |  Val: {len(val_ds):,}')

    # ── 模型 / 优化器 ──
    model     = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # ── 训练 ──
    best_acc = 0.0
    t0 = time.time()
    print(f'\n{"Epoch":>5}  {"Train Acc":>10}  {"Val Acc":>10}  {"Best":>10}')
    print('─' * 42)

    for epoch in tqdm(range(1, EPOCHS + 1), desc=f"Training", total=EPOCHS+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        flag = ''
        if va_acc > best_acc:
            best_acc = va_acc
            # 同时保存权重 + 推理时需要的元信息
            torch.save({
                'model_state_dict': model.state_dict(),
                'idx_to_char': IDX_TO_CHAR,
                'label_map': LABEL_MAP,
                'best_val_acc': best_acc,
            }, SAVE_PATH)
            flag = '  ✓'

        print(f'{epoch:>5}  {tr_acc:>10.4f}  {va_acc:>10.4f}  {best_acc:>10.4f}{flag}')

    elapsed = time.time() - t0
    print(f'\n[完成] best_val_acc={best_acc:.4%}  耗时={elapsed:.1f}s')
    print(f'[保存] {SAVE_PATH}')