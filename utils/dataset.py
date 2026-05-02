import cv2
import numpy as np
import random
import os
import json
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class WLASLDataset(Dataset):
    def __init__(self, json_file, video_root, split='train',
                 num_frames=32, label_map=None):
        with open(json_file, 'r') as f:
            full_data = json.load(f)

        self.video_root = video_root
        self.num_frames = num_frames
        self.split      = split
        self.video_ids  = []
        self.labels     = []

        # ── label map ──
        if label_map is None:
            all_actions = set()
            for info in full_data.values():
                raw   = info.get('action', info.get('label'))
                a_str = str(raw[0] if isinstance(raw, list) else raw)
                all_actions.add(a_str)
            self.action_to_idx = {a: i for i, a in enumerate(sorted(all_actions))}
        else:
            self.action_to_idx = label_map

        # ── 过滤当前 split ──
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

    def __len__(self):
        return len(self.video_ids)

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        
        # 用 set 存目标帧号，顺序读取，O(1) 查找
        indices = set(np.linspace(0, total - 1, self.num_frames, dtype=int))
        
        frames = []
        last_good = None
        
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)
                last_good = frame
            if len(frames) == self.num_frames:
                break
        
        cap.release()
        
        if not frames:
            raise ValueError("0 frames read")
        
        while len(frames) < self.num_frames:
            frames.append(last_good)
        
        return frames

    # ── 数据增强（训练 vs 测试不同策略）──
    # def _transform(self, frames):
    #     H, W = frames[0].shape[:2]

    #     if self.split == 'train':
    #         # 1. RandomResizedCrop：先 resize 到 256，再随机裁 224
    #         scale  = random.uniform(0.6, 1.0) # 0.7-> 0.6
    #         new_h  = int(H * scale)
    #         new_w  = int(W * scale)
    #         top    = random.randint(0, H - new_h)
    #         left   = random.randint(0, W - new_w)
    #         frames = [f[top:top+new_h, left:left+new_w] for f in frames]
    #         frames = [cv2.resize(f, (224, 224)) for f in frames]

    #         # # 2. RandomHorizontalFlip
    #         # if random.random() > 0.5:
    #         #     frames = [cv2.flip(f, 1) for f in frames]

    #     else:
    #         # 测试：resize 到 256 → CenterCrop 224
    #         frames = [cv2.resize(f, (256, 256)) for f in frames]
    #         top, left = 16, 16          # (256-224)//2
    #         frames = [f[top:top+224, left:left+224] for f in frames]

    #     # stack → [T, H, W, C] → [C, T, H, W]，归一化
    #     video = np.stack(frames).astype(np.float32) / 255.0  # [T,H,W,C]
    #     mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
    #     std   = np.array(IMAGENET_STD, dtype=np.float32)
    #     video = (video - mean) / std                          # broadcast
    #     video = video.transpose(3, 0, 1, 2)                  # [C,T,H,W]
    #     return torch.from_numpy(video)
    def _transform(self, frames):
        
        H, W = frames[0].shape[:2]
    
        if self.split == 'train':
            # 1. RandomResizedCrop
            scale = random.uniform(0.6, 1.0)  # 比原来更激进
            new_h = int(H * scale)
            new_w = int(W * scale)
            top   = random.randint(0, H - new_h)
            left  = random.randint(0, W - new_w)
            frames = [f[top:top+new_h, left:left+new_w] for f in frames]
            frames = [cv2.resize(f, (224, 224)) for f in frames]
    
            # 2. ColorJitter（对所有帧用同一组随机参数，保证时序一致）
            brightness = random.uniform(0.7, 1.3)
            contrast   = random.uniform(0.7, 1.3)
            saturation = random.uniform(0.7, 1.3)
            frames = [
                cv2.convertScaleAbs(f, alpha=contrast, beta=(brightness - 1) * 128)
                for f in frames
            ]
            frames = [
                cv2.cvtColor(
                    cv2.cvtColor(
                        np.clip(f.astype(np.float32) * saturation, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2HSV
                    ), cv2.COLOR_HSV2RGB
                ) for f in frames
            ]
    
            # 3. RandomGrayscale（10% 概率，帮助模型关注形状而非颜色）
            if random.random() < 0.1:
                frames = [
                    cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                    for f in frames
                ]
    
            # 4. TemporalFlip（时序翻转，手语里合理）
            if random.random() < 0.3:
                frames = frames[::-1]
    
            # 5. RandomErasing（遮挡一小块区域，防止过拟合背景）
            if random.random() < 0.3:
                er_h = random.randint(20, 60)
                er_w = random.randint(20, 60)
                er_t = random.randint(0, 224 - er_h)
                er_l = random.randint(0, 224 - er_w)
                frames = [f.copy() for f in frames]
                for f in frames:
                    f[er_t:er_t+er_h, er_l:er_l+er_w] = 128  # 灰色填充
    
        else:
            frames = [cv2.resize(f, (256, 256)) for f in frames]
            top, left = 16, 16
            frames = [f[top:top+224, left:left+224] for f in frames]
    
        video = np.stack(frames).astype(np.float32) / 255.0
        mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
        std   = np.array(IMAGENET_STD,  dtype=np.float32)
        video = (video - mean) / std
        video = video.transpose(3, 0, 1, 2)
        return torch.from_numpy(video)

    def __getitem__(self, idx):
        try:
            video_path = os.path.join(self.video_root, f"{self.video_ids[idx]}.mp4")
            frames     = self._load_frames(video_path)
            video      = self._transform(frames)
            label      = torch.tensor(self.labels[idx], dtype=torch.long)
            return video, label
        except Exception as e:
            # fallback：换一个随机样本
            return self.__getitem__(random.randint(0, len(self) - 1))

# class WLASLDataset(Dataset):
#     def __init__(self, json_file, video_root, split='train',
#                  num_frames=32, label_map=None):
#         with open(json_file, 'r') as f:
#             full_data = json.load(f)

#         self.video_root = video_root
#         self.num_frames = num_frames
#         self.split      = split
#         self.video_ids  = []
#         self.labels     = []

#         # ── label map ──
#         if label_map is None:
#             all_actions = set()
#             for info in full_data.values():
#                 raw   = info.get('action', info.get('label'))
#                 a_str = str(raw[0] if isinstance(raw, list) else raw)
#                 all_actions.add(a_str)
#             self.action_to_idx = {a: i for i, a in enumerate(sorted(all_actions))}
#         else:
#             self.action_to_idx = label_map

#         # ── 过滤当前 split ──
#         for vid_id, info in full_data.items():
#             if info.get('subset') != split:
#                 continue
#             video_path = os.path.join(video_root, f"{vid_id}.mp4")
#             if not os.path.exists(video_path):
#                 continue
#             raw   = info.get('action', info.get('label'))
#             a_str = str(raw[0] if isinstance(raw, list) else raw)
#             if a_str not in self.action_to_idx:
#                 continue
#             self.labels.append(self.action_to_idx[a_str])
#             self.video_ids.append(vid_id)

#         print(f"[{split.upper()}] {len(self.video_ids)} videos | "
#               f"{len(self.action_to_idx)} classes")

#     def __len__(self):
#         return len(self.video_ids)

#     # def _load_frames(self, video_path):
#     #     cap = cv2.VideoCapture(video_path)
#     #     total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        
#     #     # 用 set 存目标帧号，顺序读取，O(1) 查找
#     #     indices = set(np.linspace(0, total - 1, self.num_frames, dtype=int))
        
#     #     frames = []
#     #     last_good = None
        
#     #     for i in range(total):
#     #         ret, frame = cap.read()
#     #         if not ret:
#     #             break
#     #         if i in indices:
#     #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     #             frame = cv2.resize(frame, (256, 256))
#     #             frames.append(frame)
#     #             last_good = frame
#     #         if len(frames) == self.num_frames:
#     #             break
        
#     #     cap.release()
        
#     #     if not frames:
#     #         raise ValueError("0 frames read")
        
#     #     while len(frames) < self.num_frames:
#     #         frames.append(last_good)
        
#     #     return frames

#     def _load_frames(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("OpenCV 打不开这个文件，可能是头文件损坏")
            
#         total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
#         # indices 最好排序并转成 list
#         indices = sorted(list(set(np.linspace(0, total - 1, self.num_frames, dtype=int))))
        
#         frames = []
#         for idx in indices:
#             # 强行跳转到目标帧（虽然在损坏的H264视频上可能不准，但比逐帧解快100倍）
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (256, 256))
#                 frames.append(frame)

#         cap.release()
        
#         if not frames:
#             raise ValueError(f"帧数是0, indices={indices}")
            
#         # 补齐帧数
#         while len(frames) < self.num_frames:
#             frames.append(frames[-1] if frames else np.zeros((256,256,3), dtype=np.uint8))
            
#         return frames

#     # ── 数据增强（训练 vs 测试不同策略）──
#     def _transform(self, frames):
#         H, W = frames[0].shape[:2]

#         if self.split == 'train':
#             # 1. RandomResizedCrop：先 resize 到 256，再随机裁 224
#             scale  = random.uniform(0.7, 1.0)
#             new_h  = int(H * scale)
#             new_w  = int(W * scale)
#             top    = random.randint(0, H - new_h)
#             left   = random.randint(0, W - new_w)
#             frames = [f[top:top+new_h, left:left+new_w] for f in frames]
#             frames = [cv2.resize(f, (224, 224)) for f in frames]

#             # # 2. RandomHorizontalFlip
#             # if random.random() > 0.5:
#             #     frames = [cv2.flip(f, 1) for f in frames]

#         else:
#             # 测试：resize 到 256 → CenterCrop 224
#             frames = [cv2.resize(f, (256, 256)) for f in frames]
#             top, left = 16, 16          # (256-224)//2
#             frames = [f[top:top+224, left:left+224] for f in frames]

#         # stack → [T, H, W, C] → [C, T, H, W]，归一化
#         video = np.stack(frames).astype(np.float32) / 255.0  # [T,H,W,C]
#         mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
#         std   = np.array(IMAGENET_STD, dtype=np.float32)
#         video = (video - mean) / std                          # broadcast
#         video = video.transpose(3, 0, 1, 2)                  # [C,T,H,W]
#         return torch.from_numpy(video)

#     # def __getitem__(self, idx, _retry=0):
#     #     if _retry > 10:
#     #         dummy_video = torch.zeros(3, self.num_frames, 224, 224)
#     #         dummy_label = torch.tensor(0, dtype=torch.long)
#     #         return dummy_video, dummy_label
#     #     try:
#     #         video_path = os.path.join(self.video_root, f"{self.video_ids[idx]}.mp4")
#     #         frames     = self._load_frames(video_path)
#     #         video      = self._transform(frames)
#     #         label      = torch.tensor(self.labels[idx], dtype=torch.long)
#     #         return video, label
#     #     except Exception as e:
#     #         # fallback：换一个随机样本
#     #         return self.__getitem__(random.randint(0, len(self) - 1), _retry + 1)

#     def __getitem__(self, idx, _retry=0):
#         try:
#             video_path = os.path.join(self.video_root, f"{self.video_ids[idx]}.mp4")
            
#             # 暴力测试：确认文件到底存不存在！
#             if not os.path.exists(video_path):
#                 raise FileNotFoundError(f"文件不存在: {video_path}")
                
#             frames = self._load_frames(video_path)
#             video = self._transform(frames)
#             label = torch.tensor(self.labels[idx], dtype=torch.long)
#             return video, label
            
#         except Exception as e:
#             # 打印出真实的死因！
#             print(f"\n[错误] 视频 {self.video_ids[idx]} 读取失败，原因: {repr(e)}")
            
#             # 为了防止死循环，最多允许重试 3 次，超过直接抛出异常让程序崩溃！
#             if _retry > 3:
#                 raise RuntimeError("连续错误超过3次。")
                
#             return self.__getitem__(random.randint(0, len(self) - 1), _retry + 1)