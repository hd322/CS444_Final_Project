import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

class FeatureDataset(Dataset):
    def __init__(self, json_file, feature_root, split='train', label_map=None):
        with open(json_file) as f:
            full_data = json.load(f)

        self.feature_root = feature_root
        self.split        = split
        self.video_ids    = []
        self.labels       = []

        if label_map is None:
            all_actions = set()
            for info in full_data.values():
                raw   = info.get('action', info.get('label'))
                a_str = str(raw[0] if isinstance(raw, list) else raw)
                all_actions.add(a_str)
            self.action_to_idx = {a: i for i, a in enumerate(sorted(all_actions))}
        else:
            self.action_to_idx = label_map

        for vid_id, info in full_data.items():
            if info.get('subset') != split:
                continue
            npy_path = os.path.join(feature_root, f"{vid_id}.npy")
            if not os.path.exists(npy_path):
                continue
            raw   = info.get('action', info.get('label'))
            a_str = str(raw[0] if isinstance(raw, list) else raw)
            if a_str not in self.action_to_idx:
                continue
            self.labels.append(self.action_to_idx[a_str])
            self.video_ids.append(vid_id)

        print(f"[{split.upper()}] {len(self.video_ids)} videos | "
              f"{len(self.action_to_idx)} classes")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        try:
            data  = np.load(
                os.path.join(self.feature_root, f"{self.video_ids[idx]}.npy"),
                allow_pickle=True
            ).item()
            feat  = data['feature'].astype(np.float32)  # [8, 768]

            if self.split == 'train':
                # TemporalFlip
                if random.random() < 0.3:
                    mask_idx = random.randint(0, feat.shape[0] - 1)
                    feat[mask_idx, :] = 0.0
                # 轻微噪声
                feat = feat + np.random.normal(0, 0.01, feat.shape).astype(np.float32)

            return torch.from_numpy(feat), torch.tensor(self.labels[idx], dtype=torch.long)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

class FeatureLSTM(nn.Module):
    def __init__(self, input_size=768, hidden=512, num_layers=2,
                 num_classes=300, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[LSTM] Parameters: {total:,}")

    def forward(self, x):
        # x: [B, 8, 768]
        out, _ = self.lstm(x)       # [B, 8, hidden*2]
        feat = out.mean(dim=1)      # [B, hidden*2]
        return self.classifier(feat)