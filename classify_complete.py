#!/usr/bin/env python3
"""
完整分类：确保每个文件都被分配到合理的类别
基于运动特征的主成分分析
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

SOURCE_DIR = "/root/gpufree-data/motions"
TARGET_DIR = "/root/gpufree-data/motions_by_type"

def analyze(npz_path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'motion' not in data:
            return None
        motion = data['motion']
        if len(motion.shape) != 2 or motion.shape[0] < 5:
            return None
        
        root_pos = motion[:, :3]
        dof = motion[:, 7:]
        
        h_vel = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1)
        v_pos = root_pos[:, 2]
        v_vel = np.abs(np.diff(v_pos))
        j_vel = np.abs(np.diff(dof, axis=0))
        
        mid = dof.shape[1] // 2
        
        return {
            'h': np.mean(h_vel),           # 水平速度
            'hm': np.max(h_vel),           # 最大水平速度
            'd': np.sum(h_vel),            # 总距离
            'v': np.mean(v_pos),           # 平均高度
            'vr': np.ptp(v_pos),           # 高度范围
            'vs': np.max(v_vel),           # 最大垂直速度
            'j': np.mean(j_vel),           # 关节活动
            'jm': np.max(j_vel),           # 最大关节速度
            'u': np.mean(j_vel[:, mid:]),  # 上肢
            'l': np.mean(j_vel[:, :mid]),  # 下肢
        }
    except:
        return None

def classify(f, fn):
    fn = fn.lower()
    
    # 关键词优先
    kw = {
        'walk': 'tracking_walk', 'jog': 'tracking_run', 'run': 'tracking_run',
        'jump': 'tracking_jump', 'dance': 'tracking_dance', 'kick': 'tracking_fight',
        'punch': 'tracking_fight', 'box': 'tracking_fight', 'throw': 'tracking_throw',
        'catch': 'tracking_throw', 'sit': 'tracking_sit', 'crawl': 'tracking_ground',
        'lie': 'tracking_ground', 'fall': 'tracking_fallandgetup', 'push': 'tracking_push',
        'pull': 'tracking_push', 'lift': 'tracking_push', 'turn': 'tracking_turn',
        'gesture': 'tracking_gesture', 'wave': 'tracking_gesture', 'rom': 'tracking_pose',
        'pose': 'tracking_pose', 'treadmill': 'tracking_treadmill', 'sprint': 'tracking_sprint',
        'obstacle': 'tracking_obstacle', 'motorcycle': 'tracking_vehicle',
    }
    
    for k, c in kw.items():
        if k in fn:
            return c
    
    if f is None:
        return 'tracking_general'
    
    # 基于特征的决策树分类
    h, hm, d, v, vr, vs, j, jm, u, l = [f[k] for k in ['h','hm','d','v','vr','vs','j','jm','u','l']]
    
    # 1. 明显跳跃
    if vr > 0.15 and vs > 0.03:
        return 'tracking_jump'
    
    # 2. 地面动作
    if v < 0.45:
        return 'tracking_ground'
    
    # 3. 快速移动
    if h > 0.03:
        return 'tracking_run'
    
    # 4. 中速移动
    if h > 0.01 and d > 0.2:
        return 'tracking_walk'
    
    # 5. 坐/起
    if 0.1 < vr < 0.35 and h < 0.01:
        return 'tracking_sit'
    
    # 6. 上肢为主
    if u > l * 1.2:
        return 'tracking_throw' if jm > 0.3 else 'tracking_gesture'
    
    # 7. 高关节活动
    if j > 0.1:
        return 'tracking_fight' if h > 0.005 else 'tracking_push'
    
    # 8. 静止
    if h < 0.004 and j < 0.05:
        return 'tracking_pose'
    
    # 9. 根据主要特征分配
    # 有移动倾向
    if d > 0.1:
        if h > 0.008:
            return 'tracking_walk'
        else:
            return 'tracking_locomotion'
    
    # 有活动倾向
    if j > 0.06:
        if u > l:
            return 'tracking_gesture'
        else:
            return 'tracking_inplace'
    
    # 有高度变化
    if vr > 0.08:
        return 'tracking_sit'
    
    # 默认根据最显著特征
    features = {'h': h*10, 'j': j*10, 'vr': vr*5, 'u': u*8}
    dominant = max(features, key=features.get)
    
    mapping = {
        'h': 'tracking_walk',
        'j': 'tracking_inplace',
        'vr': 'tracking_sit',
        'u': 'tracking_gesture',
    }
    
    return mapping.get(dominant, 'tracking_pose')

def proc(args):
    npz, src, tgt = args
    try:
        f = analyze(npz)
        c = classify(f, npz.stem)
        
        d = tgt / c
        d.mkdir(exist_ok=True)
        
        name = "_".join(npz.relative_to(src).parts).replace("/", "_")
        t = d / name
        
        if not t.exists():
            os.link(npz, t)
            return (c, 1)
        return (c, 0)
    except:
        return (None, 0)

def main():
    src = Path(SOURCE_DIR)
    tgt = Path(TARGET_DIR)
    
    if tgt.exists():
        import shutil
        shutil.rmtree(tgt)
    tgt.mkdir()
    
    print("扫描...")
    files = list(src.rglob("*.npz"))
    print(f"文件: {len(files)}\n分类...")
    
    stats = defaultdict(int)
    
    with mp.Pool(8) as pool:
        for i, (c, n) in enumerate(pool.imap_unordered(proc, [(f, src, tgt) for f in files]), 1):
            if c:
                stats[c] += n
            if i % 1000 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")
    
    print("\n" + "="*60)
    total = sum(stats.values())
    for c in sorted(stats.keys()):
        n = stats[c]
        print(f"{c:<30} {n:>6} ({n/total*100:>5.1f}%)")
    print("-"*60)
    print(f"{'总计':<30} {total:>6} (100.0%)")
    print("="*60)
    print(f"\n✅ 完成: {tgt}")

if __name__ == "__main__":
    main()
