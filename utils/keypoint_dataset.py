import cv2
import numpy as np
import random
import os
import json
import torch
from torch.utils.data import Dataset
import mediapipe as mp
from collections import Counter

class NpyKeypointDataset(Dataset):
    def __init__(self, json_file, npy_dir, split='train', label_map=None):
        with open(json_file, 'r') as f:
            wlasl_data = json.load(f)

        self.samples = []
        # 如果没有传入 label_map（比如训练集），则自己构建一个
        self.action_to_idx = label_map if label_map is not None else {}
        
        for vid_id, info in wlasl_data.items():
            # 匹配对应的数据集划分 (train / val / test)
            if info.get('subset') == split:
                npy_path = os.path.join(npy_dir, f"{vid_id}.npy")
                
                # 只加载那些成功提取了 MediaPipe 特征的视频
                if os.path.exists(npy_path):
                    # 获取标签
                    label_raw = info.get('action', info.get('label', ''))
                    
                    # 💥 核心修复：如果是列表（比如 [15, "book"]），转成元组，让它可以作为 dict 的 key
                    if isinstance(label_raw, list):
                        label_str = label_raw[0] 
                    else:
                        label_str = label_raw
                    
                    if label_map is None and label_str not in self.action_to_idx:
                        self.action_to_idx[label_str] = len(self.action_to_idx)
                        
                    self.samples.append((npy_path, label_str))
                    
        print(f"[{split.upper()}] 成功加载 {len(self.samples)} 个 .npy 特征文件")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
            npy_path, label_str = self.samples[idx]
            
            # 加载 npy 数据
            data = np.load(npy_path, allow_pickle=True)
            
            # 💥 核心兼容逻辑：
            # 如果维度是 0 且里面是字典，说明是我们刚用新脚本提取的
            if data.ndim == 0 and isinstance(data.item(), dict):
                features = data.item()['feature']
            # 否则，说明是你以前用其他代码存下来的纯数组老文件
            else:
                features = data
                
            x = torch.FloatTensor(features)
            y = self.action_to_idx[label_str]
            
            return x, y

class WLASLKeypointDataset(Dataset):
    def __init__(self, json_file, video_root, split='train',
                 num_frames=32, label_map=None):
        with open(json_file, 'r') as f:
            full_data = json.load(f)

        self.video_root = video_root
        self.num_frames = num_frames
        self.split      = split
        self.video_ids  = []
        self.labels     = []

        # ── label map（和原来一样）──
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
            video_path = os.path.join(video_root, f"{vid_id}.mp4")
            if not os.path.exists(video_path):
                continue
            raw   = info.get('action', info.get('label'))
            a_str = str(raw[0] if isinstance(raw, list) else raw)
            if a_str not in self.action_to_idx:
                continue
            self.labels.append(self.action_to_idx[a_str])
            self.video_ids.append(vid_id)

        print(f"[{split.upper()}] {len(self.video_ids)} videos | "
              f"{len(self.action_to_idx)} classes")

        # MediaPipe Hands，每个 worker 独立初始化
        self._hands = None

    def _get_hands(self):
        # DataLoader 多进程时不能共享 MediaPipe 对象，懒加载
        if self._hands is None:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,       # 逐帧处理，不做时序追踪
                max_num_hands=2,
                min_detection_confidence=0.3  # 低一点，减少漏检
            )
        return self._hands

    def _extract_keypoints(self, video_path):
        """
        返回 [num_frames, 126] 的 numpy 数组
        126 = 2只手 × 21个关键点 × 3(x, y, z)
        检测不到手的帧填 0
        """
        cap   = cv2.VideoCapture(video_path)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = set(np.linspace(0, total - 1, self.num_frames, dtype=int))

        hands    = self._get_hands()
        keypoints = []
        last_good = np.zeros(126, dtype=np.float32)

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i not in indices:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)

            kp = np.zeros(126, dtype=np.float32)
            if results.multi_hand_landmarks:
                for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks[:2]):
                    offset = hand_idx * 63  # 每只手 21*3=63
                    for j, lm in enumerate(hand_lm.landmark):
                        kp[offset + j*3]     = lm.x
                        kp[offset + j*3 + 1] = lm.y
                        kp[offset + j*3 + 2] = lm.z
                last_good = kp.copy()
            else:
                # 检测不到手：用上一帧填充
                kp = last_good.copy()

            keypoints.append(kp)
            if len(keypoints) == self.num_frames:
                break

        cap.release()

        if not keypoints:
            return np.zeros((self.num_frames, 126), dtype=np.float32)

        # 帧数不够时用最后一帧补齐
        while len(keypoints) < self.num_frames:
            keypoints.append(keypoints[-1])

        return np.stack(keypoints)  # [T, 126]

    def _augment(self, kp):
        """
        kp: [T, 126]，只在训练时调用
        """
        # 1. TemporalFlip
        if random.random() < 0.3:
            kp = kp[::-1].copy()

        # 2. 轻微抖动（模拟检测噪声）
        kp = kp + np.random.normal(0, 0.01, kp.shape).astype(np.float32)

        # 3. 随机缩放（手的大小变化）
        scale = random.uniform(0.9, 1.1)
        kp    = kp * scale

        return kp

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        try:
            video_path = os.path.join(self.video_root, f"{self.video_ids[idx]}.mp4")
            kp         = self._extract_keypoints(video_path)  # [T, 126]

            if self.split == 'train':
                kp = self._augment(kp)

            kp_tensor = torch.from_numpy(kp.astype(np.float32))  # [T, 126]
            label     = torch.tensor(self.labels[idx], dtype=torch.long)
            return kp_tensor, label

        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))