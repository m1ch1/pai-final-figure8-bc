#!/usr/bin/env python3
"""
行動模倣(画像 → 前進/左/右 の3クラス分類)の学習【完成版】

入力 : ~/dataset/ep_N.npz(bag_to_dataset.py の出力)
出力 : policy.pt   … 正規化まで含めた TorchScript モデル
                      入力は uint8 の (N, H, W, 3) RGB 画像、出力は3クラスのスコア
       history.csv … エポックごとの損失・正解率(レポートのグラフ用)

使い方:
  python3 train_full.py --data ~/dataset --epochs 15
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ['forward', 'left', 'right']


# ---------------- データ ----------------
def load_episodes(data_dir):
    files = glob.glob(os.path.join(data_dir, 'ep_*.npz'))
    files.sort(key=lambda f: int(re.search(r'ep_(\d+)', f).group(1)))
    eps = []
    for f in files:
        d = np.load(f)
        eps.append((os.path.basename(f)[:-4], d['images'], d['labels']))
    return eps


def stack(eps, stride):
    x = np.concatenate([im[::stride] for _, im, _ in eps])
    y = np.concatenate([lb[::stride] for _, _, lb in eps])
    return x, y


# ---------------- モデル ----------------
class Normalize(nn.Module):
    """uint8 (N,H,W,3) → 正規化した float (N,3,H,W)。推論ノードとの前処理のずれを防ぐため、モデルに含める"""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std


class PolicyCNN(nn.Module):
    def __init__(self, h, w, n_classes=3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        # MaxPool を3回 → 縦横それぞれ 1/8。全体平均プーリングは使わない(物の「位置」が判断材料なので)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * (h // 8) * (w // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------- 評価 ----------------
def evaluate(model, loader, crit):
    model.eval()
    cm = np.zeros((3, 3), dtype=int)   # 行: 正解, 列: 予測
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            loss_sum += crit(out, yb).item() * len(yb)
            n += len(yb)
            np.add.at(cm, (yb.numpy(), out.argmax(1).numpy()), 1)
    acc = np.trace(cm) / max(cm.sum(), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return loss_sum / max(n, 1), acc, recall.mean(), cm


def print_cm(cm):
    print('            予測→ ' + ' '.join(f'{c:>8s}' for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f'  正解 {c:>8s}   ' + ' '.join(f'{v:8d}' for v in cm[i]))


# ---------------- メイン ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/dataset'))
    ap.add_argument('--out', default='policy.pt')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--stride', type=int, default=2, help='何枚に1枚使うか(連続フレームはほぼ同じ画像)')
    ap.add_argument('--val-every', type=int, default=5, help='何エピソードに1つを検証用にするか')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)

    eps = load_episodes(args.data)
    # エピソード単位で分ける(連続フレームが学習と検証の両方に入ると、精度が高く出すぎる)
    val_eps = [e for k, e in enumerate(eps) if k % args.val_every == args.val_every - 1]
    tr_eps = [e for k, e in enumerate(eps) if k % args.val_every != args.val_every - 1]
    print('学習:', [e[0] for e in tr_eps])
    print('検証:', [e[0] for e in val_eps])

    xtr, ytr = stack(tr_eps, args.stride)
    xva, yva = stack(val_eps, args.stride)
    h, w = xtr.shape[1:3]
    print(f'学習 {len(ytr)} 枚 / 検証 {len(yva)} 枚  画像 {w}x{h}')

    # 正規化の統計は学習データだけから計算する
    mean = (xtr.reshape(-1, 3).mean(0) / 255.0).tolist()
    std = (xtr.reshape(-1, 3).std(0) / 255.0 + 1e-6).tolist()

    # クラスの重み = 全体の枚数 / (クラス数 × そのクラスの枚数)
    counts = np.bincount(ytr, minlength=3)
    weights = counts.sum() / (3 * np.maximum(counts, 1))
    print('クラスの枚数:', dict(zip(CLASSES, counts.tolist())),
          ' 重み:', dict(zip(CLASSES, np.round(weights, 2).tolist())))

    model = nn.Sequential(Normalize(mean, std), PolicyCNN(h, w))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                           batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(xva), torch.from_numpy(yva)),
                           batch_size=512)

    best_bacc, best_state = -1.0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            n += len(yb)
        tr_loss = loss_sum / n
        va_loss, va_acc, va_bacc, _ = evaluate(model, va_loader, crit)
        history.append([epoch, tr_loss, va_loss, va_acc, va_bacc])
        mark = ''
        if va_bacc > best_bacc:  # クラスごとの正解率の平均(偏りに強い)で一番よいものを残す
            best_bacc = va_bacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            mark = '  *best'
        print(f'epoch {epoch:2d}  train_loss {tr_loss:.3f}  val_loss {va_loss:.3f}  '
              f'val_acc {va_acc:.3f}  val_bal_acc {va_bacc:.3f}{mark}')

    model.load_state_dict(best_state)
    _, acc, bacc, cm = evaluate(model, va_loader, crit)
    print(f'\n最良モデル: val_acc {acc:.3f}  val_bal_acc {bacc:.3f}')
    print_cm(cm)

    model.eval()
    torch.jit.script(model).save(args.out)
    with open('history.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_bal_acc'])
        wr.writerows(history)
    print(f'\n保存: {os.path.abspath(args.out)} , history.csv')


if __name__ == '__main__':
    main()
