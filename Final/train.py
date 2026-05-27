"""
Mini AlphaGo 訓練程式 (8-Channel 升級版)
環境：Google Colab + GPU
輸出：.mem 權重檔（INT8，給 FPGA 用）與 scales.json
"""

import os
import re
import json
import copy
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# 自動偵測是否在 Colab 環境
try:
    from google.colab import files, drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================================
# 0. 設定區
# ==========================================
CONFIG = {
    # 🌟 記得修改成你的 SGF 壓縮檔路徑
    'sgf_zip'      : '/content/drive/MyDrive/ckpt/sgf.zip',
    'ckpt_dir'     : '/content/drive/MyDrive/ckpt_8ch_resnet',
    'max_files'    : 34572,
    'epochs'       : 30,
    'batch_size'   : 1024,
    'lr'           : 1e-3,
    'filters'      : 64,
    'num_res'      : 2,  # Tower 1 代表有 1 個 ResBlock (2層卷積)
    'board_size'   : 9,
}

# ==========================================
# 1. 圍棋邏輯引擎與 8-Channel 特徵提取
# ==========================================
class GoBoard:
    def __init__(self, size=9):
        self.size  = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.ko_point = None

    def clear(self):
        self.board.fill(0)
        self.ko_point = None

    def get_group_and_liberties(self, x, y):
        color = self.board[y][x]
        if color == 0:
            return set(), set()
        group     = set()
        liberties = set()
        frontier  = [(x, y)]
        while frontier:
            cx, cy = frontier.pop()
            if (cx, cy) in group:
                continue
            group.add((cx, cy))
            for nx, ny in self._neighbors(cx, cy):
                nc = self.board[ny][nx]
                if nc == color and (nx, ny) not in group:
                    frontier.append((nx, ny))
                elif nc == 0:
                    liberties.add((nx, ny))
        return group, liberties

    def play_move(self, x, y, color):
        if self.board[y][x] != 0:
            return False
        if self.ko_point == (x, y):
            return False

        self.board[y][x] = color
        opponent = 2 if color == 1 else 1

        captured = []
        for nx, ny in self._neighbors(x, y):
            if self.board[ny][nx] == opponent:
                grp, libs = self.get_group_and_liberties(nx, ny)
                if len(libs) == 0:
                    for dx, dy in grp:
                        self.board[dy][dx] = 0
                        captured.append((dx, dy))

        _, my_libs = self.get_group_and_liberties(x, y)
        if len(my_libs) == 0:
            self.board[y][x] = 0
            return False

        self.ko_point = captured[0] if len(captured) == 1 else None
        return True

    def _neighbors(self, x, y):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                yield nx, ny

    def to_features_8ch(self, current_color, last_move, prev_move):
        my_color = current_color
        op_color = 2 if current_color == 1 else 1

        # Ch0, Ch1: 雙方盤面 (我方優先)
        ch0 = (self.board == my_color).astype(np.float32)
        ch1 = (self.board == op_color).astype(np.float32)

        # 快速計算全盤的氣數分佈
        lib_map = np.zeros_like(self.board, dtype=int)
        visited = set()
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] != 0 and (x, y) not in visited:
                    grp, libs = self.get_group_and_liberties(x, y)
                    num_libs = len(libs)
                    for gx, gy in grp:
                        lib_map[gy][gx] = num_libs
                        visited.add((gx, gy))

        # Ch2, Ch3: 我方的氣 (打吃與雙叫吃)
        ch2 = ((self.board == my_color) & (lib_map == 1)).astype(np.float32)
        ch3 = ((self.board == my_color) & (lib_map == 2)).astype(np.float32)

        # Ch4, Ch5: 對方的氣
        ch4 = ((self.board == op_color) & (lib_map == 1)).astype(np.float32)
        ch5 = ((self.board == op_color) & (lib_map == 2)).astype(np.float32)

        # Ch6, Ch7: 歷史落子
        ch6 = np.zeros((self.size, self.size), dtype=np.float32)
        if last_move is not None:
            ch6[last_move[1], last_move[0]] = 1.0

        ch7 = np.zeros((self.size, self.size), dtype=np.float32)
        if prev_move is not None:
            ch7[prev_move[1], prev_move[0]] = 1.0

        return np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7], axis=0)

# ==========================================
# 2. SGF 資料集 (無縫讀取 ZIP)
# ==========================================
class SGFDataset(Dataset):
    def __init__(self, sgf_zip, max_files=34572, augment=True):
        self.augment = augment
        self.size    = CONFIG['board_size']
        self.samples = []
        self._load(sgf_zip, max_files)

    def _load(self, sgf_zip, max_files):
        with zipfile.ZipFile(sgf_zip, 'r') as zf:
            sgf_names = sorted(
                name for name in zf.namelist()
                if name.lower().endswith('.sgf')
            )[:max_files]

            print(f"zip 內找到 {len(sgf_names)} 個 SGF 檔案，開始解析並建立 8-Channel 樣本...")

            board   = GoBoard(self.size)
            skipped = 0

            for idx, name in enumerate(sgf_names):
                try:
                    raw  = zf.read(name)
                    text = raw.decode('utf-8', errors='ignore')
                except Exception:
                    skipped += 1
                    continue

                # 解析勝負
                re_m = re.search(r'RE\[([^\]]+)\]', text)
                if not re_m:
                    skipped += 1
                    continue
                result = re_m.group(1)
                if   result.startswith('B'): black_win =  1.0
                elif result.startswith('W'): black_win = -1.0
                else:
                    skipped += 1
                    continue

                # 解析手順
                moves = []
                for i in range(len(text) - 4):
                    if (text[i] in ['B','W'] and text[i+1] == '[' and text[i+4] == ']'):
                        color = 1 if text[i] == 'B' else 2
                        x = ord(text[i+2]) - ord('a')
                        y = ord(text[i+3]) - ord('a')
                        if 0 <= x < self.size and 0 <= y < self.size:
                            moves.append((color, x, y))

                if len(moves) < 5:
                    skipped += 1
                    continue

                # 產生帶有歷史記憶的樣本
                board.clear()
                prev_move = None
                last_move = None

                for color, x, y in moves:
                    feat       = board.to_features_8ch(color, last_move, prev_move)
                    policy_idx = y * self.size + x
                    value      = black_win if color == 1 else -black_win

                    self.samples.append((
                        feat.copy(),
                        policy_idx,
                        np.float32(value)
                    ))

                    board.play_move(x, y, color)
                    prev_move = last_move
                    last_move = (x, y)

                if (idx + 1) % 2000 == 0:
                    print(f"  已解析 {idx+1}/{len(sgf_names)} 局，"
                          f"樣本數：{len(self.samples):,}")

        print(f"\n解析完成！")
        print(f"  有效 8-Channel 樣本：{len(self.samples):,}")
        print(f"  跳過棋局：{skipped}")

    def _augment8(self, feat, idx):
        s = self.size
        k = np.random.randint(8)
        if idx >= s * s:
            return feat, idx
        r, c = divmod(idx, s)
        f = feat.copy()
        if k >= 4:
            f  = np.flip(f, axis=2).copy()
            c  = s - 1 - c
            k -= 4
        f = np.rot90(f, k, axes=(1, 2)).copy()
        for _ in range(k):
            r, c = c, s - 1 - r
        return f, r * s + c

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, policy_idx, value = self.samples[idx]
        if self.augment:
            feat, policy_idx = self._augment8(feat, policy_idx)
        return (
            torch.FloatTensor(feat),
            torch.tensor(policy_idx, dtype=torch.long),
            torch.tensor(value,      dtype=torch.float32)
        )

# ==========================================
# 3. 神經網路架構 (8-Channel 輸入)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)

class MiniAlphaGo(nn.Module):
    def __init__(self, filters=64, num_res=2):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(8, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.tower = nn.Sequential(
            *[ResBlock(filters) for _ in range(num_res)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 81)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(81, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.tower(x)
        return self.policy_head(x), self.value_head(x)

# ==========================================
# 4. Trainer
# ==========================================
class Trainer:
    def __init__(self, config):
        self.cfg    = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用裝置：{self.device}")
        os.makedirs(config['ckpt_dir'], exist_ok=True)
        self.ckpt_path = os.path.join(config['ckpt_dir'], 'checkpoint.pth')
        self.best_path = os.path.join(config['ckpt_dir'], 'best_model.pth')

    def build_dataloaders(self):
        print("\n載入資料集...")
        dataset = SGFDataset(
            self.cfg['sgf_zip'],
            max_files=self.cfg['max_files'],
            augment=True
        )
        n_val   = max(1000, int(len(dataset) * 0.05))
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train_set, batch_size=self.cfg['batch_size'],
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_set, batch_size=self.cfg['batch_size'],
            shuffle=False, num_workers=2
        )
        print(f"訓練樣本：{n_train:,}，驗證樣本：{n_val:,}")
        return train_loader, val_loader

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_val_loss, log):
        torch.save({
            'epoch'         : epoch,
            'model_state'   : model.state_dict(),
            'optim_state'   : optimizer.state_dict(),
            'sched_state'   : scheduler.state_dict(),
            'best_val_loss' : best_val_loss,
            'log'           : log,
        }, self.ckpt_path)
        print(f"  Checkpoint 儲存至 {self.ckpt_path}")

    def load_checkpoint(self, model, optimizer, scheduler):
        if not os.path.exists(self.ckpt_path):
            return 0, float('inf'), []
        print(f"發現 Checkpoint，繼續上次訓練...")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        print(f"  從 Epoch {ckpt['epoch']+1} 繼續")
        return ckpt['epoch'] + 1, ckpt['best_val_loss'], ckpt['log']

    def train(self):
        train_loader, val_loader = self.build_dataloaders()
        model     = MiniAlphaGo(self.cfg['filters'], self.cfg['num_res']).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=self.cfg['lr'], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg['epochs'])
        start_epoch, best_val_loss, log = self.load_checkpoint(
            model, optimizer, scheduler
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型參數量：{total_params:,}")
        print(f"開始訓練 Epoch {start_epoch+1} ~ {self.cfg['epochs']}\n")

        for epoch in range(start_epoch, self.cfg['epochs']):
            model.train()
            t_loss = t_ploss = t_vloss = t_acc = 0
            n_batch = 0

            for states, policies, values in train_loader:
                states   = states.to(self.device)
                policies = policies.to(self.device)
                values   = values.to(self.device)
                pred_p, pred_v = model(states)
                loss_p = F.cross_entropy(pred_p, policies)
                loss_v = F.mse_loss(pred_v.squeeze(1), values)
                loss   = loss_p + loss_v
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss  += loss.item()
                t_ploss += loss_p.item()
                t_vloss += loss_v.item()
                t_acc   += (pred_p.argmax(1) == policies).float().mean().item()
                n_batch += 1

            scheduler.step()

            model.eval()
            v_loss = v_acc = v_n = 0
            with torch.no_grad():
                for states, policies, values in val_loader:
                    states   = states.to(self.device)
                    policies = policies.to(self.device)
                    values   = values.to(self.device)
                    pred_p, pred_v = model(states)
                    loss_p = F.cross_entropy(pred_p, policies)
                    loss_v = F.mse_loss(pred_v.squeeze(1), values)
                    v_loss += (loss_p + loss_v).item()
                    v_acc  += (pred_p.argmax(1) == policies).float().mean().item()
                    v_n    += 1

            avg_t_loss = t_loss  / n_batch
            avg_t_acc  = t_acc   / n_batch
            avg_v_loss = v_loss  / v_n
            avg_v_acc  = v_acc   / v_n

            entry = {
                'epoch'  : epoch + 1,
                't_loss' : round(avg_t_loss, 4),
                't_acc'  : round(avg_t_acc,  4),
                'v_loss' : round(avg_v_loss, 4),
                'v_acc'  : round(avg_v_acc,  4),
            }
            log.append(entry)
            print(f"Epoch {epoch+1:3d}/{self.cfg['epochs']} | "
                  f"Loss {avg_t_loss:.4f} "
                  f"(P:{t_ploss/n_batch:.4f} V:{t_vloss/n_batch:.4f}) | "
                  f"Acc {avg_t_acc:.3f} | "
                  f"Val Loss {avg_v_loss:.4f} | "
                  f"Val Acc {avg_v_acc:.3f}")

            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                torch.save(model.state_dict(), self.best_path)
                print(f"  最佳模型更新 (Val Loss: {best_val_loss:.4f})")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, best_val_loss, log
                )

        print("\n訓練完成！載入最佳模型進行導出...")
        model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        return model, log

# ==========================================
# 5. INT8 權重量化與導出
# ==========================================
class WeightExporter:
    def __init__(self, model, out_dir='/content/drive/MyDrive/ckpt'):
        self.model   = model.cpu().eval()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _quantize_int8(self, weight_np):
        max_abs = np.max(np.abs(weight_np))
        if max_abs == 0:
            return np.zeros_like(weight_np, dtype=np.int8), 1.0
        scale = max_abs / 127.0
        q     = np.clip(np.round(weight_np / scale), -128, 127).astype(np.int8)
        return q, scale

    def _to_mem_file(self, int8_arr, filepath):
        flat = int8_arr.flatten()
        with open(filepath, 'w') as f:
            for v in flat:
                bits = format(int(v) & 0xFF, '08b')
                f.write(bits + '\n')

    def _save_scale(self, scales_dict, filepath):
        with open(filepath, 'w') as f:
            json.dump(scales_dict, f, indent=2)

    def export(self):
        print("\n開始導出 INT8 權重...")
        scales      = {}
        model_fused = self._fuse_bn(self.model)

        for name, param in model_fused.named_parameters():
            arr      = param.detach().numpy()
            q, scale = self._quantize_int8(arr)
            safe     = name.replace('.', '_')
            fpath    = os.path.join(self.out_dir, f'{safe}.mem')
            self._to_mem_file(q, fpath)
            scales[safe] = {
                'scale' : float(scale),
                'shape' : list(arr.shape),
                'size'  : int(np.prod(arr.shape))
            }
            print(f"  {name:50s} shape={arr.shape}  scale={scale:.6f}")

        self._save_scale(scales, os.path.join(self.out_dir, 'scales.json'))
        self._verify_quantization(model_fused, scales)
        print(f"\n導出完成！檔案儲存於：{self.out_dir}")
        print(f"共導出 {len(scales)} 個權重檔案")
        return scales

    def _fuse_bn(self, model):
        m = copy.deepcopy(model)

        def fuse_conv_bn(conv, bn):
            w       = conv.weight.data
            gamma   = bn.weight.data
            beta    = bn.bias.data
            mean    = bn.running_mean
            var     = bn.running_var
            eps     = bn.eps
            std     = torch.sqrt(var + eps)
            w_fused = w * (gamma / std).view(-1, 1, 1, 1)
            b_fused = beta - gamma * mean / std
            fused   = nn.Conv2d(
                conv.in_channels, conv.out_channels,
                conv.kernel_size, conv.stride, conv.padding, bias=True
            )
            fused.weight.data = w_fused
            fused.bias.data   = b_fused
            return fused

        m.entry[0] = fuse_conv_bn(m.entry[0], m.entry[1])
        m.entry[1] = nn.Identity()
        for block in m.tower:
            block.net[0] = fuse_conv_bn(block.net[0], block.net[1])
            block.net[1] = nn.Identity()
            block.net[3] = fuse_conv_bn(block.net[3], block.net[4])
            block.net[4] = nn.Identity()
        return m

    def _verify_quantization(self, model, scales):
        print("\n量化誤差驗證...")
        total_err = 0
        n = 0
        for name, param in model.named_parameters():
            arr   = param.detach().numpy()
            safe  = name.replace('.', '_')
            scale = scales[safe]['scale']
            q     = np.clip(np.round(arr / scale), -128, 127)
            err   = np.mean(np.abs(arr - q * scale))
            total_err += err
            n += 1
        print(f"  平均量化誤差：{total_err/n:.6f}")
        print(f"  （越小越好，< 0.01 為佳）")

# ==========================================
# 6. 主程式
# ==========================================
def main():
    if IN_COLAB:
        print("掛載 Google Drive...")
        drive.mount('/content/drive')

    trainer    = Trainer(CONFIG)
    model, log = trainer.train()

    exporter = WeightExporter(model, out_dir=CONFIG['ckpt_dir'])
    scales   = exporter.export()

    log_path = os.path.join(CONFIG['ckpt_dir'], 'train_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n訓練 log 儲存至：{log_path}")

    if IN_COLAB:
        print("\n下載權重檔案...")
        for fname in os.listdir(CONFIG['ckpt_dir']):
            if fname.endswith('.mem') or fname in ['scales.json', 'train_log.json']:
                files.download(os.path.join(CONFIG['ckpt_dir'], fname))

    print("\n全部完成！")

if __name__ == '__main__':
    main()
