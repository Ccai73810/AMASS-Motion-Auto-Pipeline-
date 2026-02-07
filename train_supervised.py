import os
import shutil # 引入 shutil 用于文件移动
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置
# ==========================================
CLASS_NAMES = [
    'tracking_dance', 'tracking_fallandgetup', 'tracking_fight', 
    'tracking_gesture', 'tracking_ground', 'tracking_jump', 
    'tracking_obstacle', 'tracking_pose', 'tracking_push', 
    'tracking_run', 'tracking_sit', 'tracking_sprint', 
    'tracking_throw', 'tracking_treadmill', 'tracking_turn', 
    'tracking_vehicle', 'tracking_walk'
]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- 核心修改区 ---
MAX_ROUNDS = 30          # 改成 30 轮！不搬空不罢休
EPOCHS_PER_ROUND = 10    # 每轮训练次数稍微减少一点，加快迭代速度
BATCH_SIZE = 64
SEQ_LEN = 60

# 动态阈值策略：从 0.95 慢慢降到 0.80，保证最后能收底
START_CONF = 0.95
END_CONF = 0.80

# ==========================================
# 2. 数据集类
# ==========================================
class MemoryDataset(Dataset):
    def __init__(self, data_list, seq_len=60):
        self.seq_len = seq_len
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item['path']
        label = item['label']
        
        try:
            with np.load(file_path, allow_pickle=True) as data:
                best_arr = None
                max_len = 0
                for k in data.files:
                    arr = data[k]
                    if hasattr(arr, 'ndim') and arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                        if arr.shape[0] > max_len and arr.shape[1] > 10:
                            max_len = arr.shape[0]
                            best_arr = arr
                
                if best_arr is None:
                    return torch.zeros((self.seq_len, 72), dtype=torch.float32), label
                
                if best_arr.shape[1] >= 72: clip = best_arr[:, :72]
                else: clip = np.hstack([best_arr, np.zeros((best_arr.shape[0], 72 - best_arr.shape[1]))])
                
                if clip.shape[0] >= self.seq_len: clip = clip[:self.seq_len, :]
                else: clip = np.vstack([clip, np.zeros((self.seq_len - clip.shape[0], 72))])
                
                return torch.from_numpy(clip.astype(np.float32)), label
        except:
            return torch.zeros((self.seq_len, 72), dtype=torch.float32), label

# ==========================================
# 3. 扫描目录
# ==========================================
def scan_directory(root_dir):
    labeled_list = []
    unlabeled_list = []
    
    print(f"🔍 正在深度扫描根目录: {root_dir} ...")
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path): continue
        
        is_unlabeled = (folder_name == 'others' or folder_name == 'tracking_general')
        label = -1
        
        if not is_unlabeled:
            if folder_name in LABEL_MAP:
                label = LABEL_MAP[folder_name]
            else:
                continue 
        
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.npz'):
                    item = {'path': os.path.join(root, f), 'label': label}
                    if is_unlabeled:
                        unlabeled_list.append(item)
                    else:
                        labeled_list.append(item)
        
    print(f"✅ 扫描完成!")
    print(f"   - 已分类样本 (训练集): {len(labeled_list)}")
    print(f"   - 待处理样本 (others): {len(unlabeled_list)}")
    return labeled_list, unlabeled_list

# ==========================================
# 4. 模型定义
# ==========================================
class MotionClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=72, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # --- 请修改这里的路径 ---
    ROOT_DIR = r"D:\amassdata\motions_by_type_1\motions_by_type_1" 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_pool, candidate_pool = scan_directory(ROOT_DIR)
    
    if len(train_pool) == 0:
        print("❌ 错误：未找到训练数据！")
        exit()

    history = {'pool_size': [len(train_pool)]}
    
    model = MotionClassifier(len(CLASS_NAMES)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🚀 开始暴力搬运迭代 (共 {MAX_ROUNDS} 轮)")
    print(f"🎯 目标：把 others 里的 {len(candidate_pool)} 个文件搬空！")

    # ================= 循环迭代 =================
    for round_idx in range(MAX_ROUNDS):
        # 计算当前轮次的阈值 (线性递减)
        current_threshold = START_CONF - (START_CONF - END_CONF) * (round_idx / MAX_ROUNDS)
        
        print(f"\n{'='*30}")
        print(f"🔄 Round {round_idx+1} / {MAX_ROUNDS}")
        print(f"⚙️ 当前置信度门槛: {current_threshold:.2f} (越低搬得越猛)")
        print(f"📚 当前训练集规模: {len(train_pool)}")
        print(f"📦 others 剩余文件: {len(candidate_pool)}")
        
        if len(candidate_pool) == 0:
            print("🎉 恭喜！others 文件夹已经被搬空了！")
            break

        # --- A. 训练阶段 ---
        train_ds = MemoryDataset(train_pool, seq_len=SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model.train()
        for epoch in range(EPOCHS_PER_ROUND):
            pbar = tqdm(train_loader, desc=f"   [训练] Ep {epoch+1}/{EPOCHS_PER_ROUND}", leave=False)
            for x, y in pbar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- B. 挖掘与搬运阶段 ---
        print(f"   [挖掘] 正在扫描 others...")
        
        candidate_ds = MemoryDataset(candidate_pool, seq_len=SEQ_LEN)
        candidate_loader = DataLoader(candidate_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model.eval()
        new_labeled_samples = []
        remaining_candidates = []
        moved_count_this_round = 0
        
        with torch.no_grad():
            global_idx = 0
            for x, _ in tqdm(candidate_loader, desc="   [搬运中]"):
                x = x.to(DEVICE)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                for i in range(x.size(0)):
                    current_item = candidate_pool[global_idx]
                    prob = max_probs[i].item()
                    pred_label_idx = preds[i].item()
                    
                    if prob > current_threshold:
                        # === 物理搬运 ===
                        target_folder_name = CLASS_NAMES[pred_label_idx]
                        target_dir = os.path.join(ROOT_DIR, target_folder_name)
                        
                        original_filename = os.path.basename(current_item['path'])
                        # 加上 auto_ 前缀方便识别
                        new_filename = f"auto_r{round_idx+1}_{original_filename}"
                        target_path = os.path.join(target_dir, new_filename)
                        
                        try:
                            # 真实的物理移动！
                            shutil.move(current_item['path'], target_path)
                            
                            # 更新内存状态
                            current_item['label'] = pred_label_idx
                            current_item['path'] = target_path 
                            new_labeled_samples.append(current_item)
                            moved_count_this_round += 1
                            
                        except Exception as e:
                            # 如果搬运失败（比如文件占用），就跳过
                            remaining_candidates.append(current_item) 
                    else:
                        remaining_candidates.append(current_item)
                    
                    global_idx += 1
        
        # --- C. 汇报战果 ---
        print(f"🚚 本轮成功搬运: {moved_count_this_round} 个文件")
        print(f"📉 others 剩余文件数: {len(remaining_candidates)}")
        
        if moved_count_this_round == 0:
            print("⚠️ 警告：本轮没有搬走任何文件，可能是阈值太高或模型学不动了。")
            if current_threshold <= END_CONF:
                 print("🛑 已达到最低阈值且无新样本，强行结束。")
                 break
            
        train_pool.extend(new_labeled_samples)
        candidate_pool = remaining_candidates
        history['pool_size'].append(len(train_pool))

    print("\n✅ 所有迭代完成！")
    print("快去 others 文件夹看看是不是空了！")