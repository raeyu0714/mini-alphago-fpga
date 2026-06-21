# Mini AlphaGo Zero — FPGA + PC 混合系統

## 目錄
1. [專案概述](#1-專案概述)
2. [系統架構](#2-系統架構)
3. [UART 封包協議](#3-uart-封包協議)
4. [遊戲流程](#4-遊戲流程)
5. [AI 回合流程](#5-ai-回合流程)
6. [CNN 網路架構](#6-cnn-網路架構)
7. [CNN 推理流程](#7-cnn-推理流程)
8. [MCTS 演算法](#8-mcts-演算法)
9. [訓練流程](#9-訓練流程)
10. [FPGA 模組說明](#10-fpga-模組說明)
11. [Python 檔案說明](#11-python-檔案說明)
12. [權重與 Scale 檔案](#12-權重與-scale-檔案)

---

## 1. 專案概述

本專案實作一個 **Mini AlphaGo Zero** 系統，在 9×9 圍棋盤上運行，採用**軟硬體混合**設計：

- **FPGA（Artix-7）**：負責 CNN 硬體推理、圍棋規則檢查、VGA 畫面輸出、使用者輸入
- **PC（Python）**：負責 MCTS 樹搜索演算法
- 兩者透過 **UART 115200 baud** 通訊

AI 扮演**白棋（玩家 2）**，人類扮演**黑棋（玩家 1）**。

---

## 2. 系統架構

```
┌─────────────────────────────────────┐     UART 115200     ┌──────────────────────────────┐
│             FPGA (Artix-7)          │◄───────────────────►│          PC (Python)         │
│                                     │                     │                              │
│  ┌──────────┐    ┌───────────────┐  │  0xAA: 玩家落子通知  │  MCTS 樹搜索                 │
│  │ 遊戲 FSM  │───►│  go_ai_core  │  │  0xBB: CNN 推理結果  │  pcaitest.py                 │
│  │          │    │  (HIL 引擎)   │  │  0xCC: 評估請求      │                              │
│  └──────────┘    └───────┬───────┘  │  0xDD: AI 最終落子   │  N_SIMS=200 次模擬            │
│       │                  │          │                     │  C_PUCT=1.5                  │
│  ┌────▼─────┐    ┌───────▼───────┐  │                     └──────────────────────────────┘
│  │ 規則引擎  │    │  CNN 推理引擎  │  │
│  │          │    │  conv_unit    │  │
│  └──────────┘    │  fcunit       │  │
│  ┌──────────┐    └───────────────┘  │
│  │ VGA 輸出  │                       │
│  │ 640×480  │                       │
│  └──────────┘                       │
└─────────────────────────────────────┘
```

### 硬體規格
| 項目 | 規格 |
|------|------|
| FPGA 晶片 | Artix-7 |
| 系統時脈 | 100 MHz |
| VGA 時脈 | 25 MHz（4 分頻） |
| UART 鮑率 | 115200 |
| 棋盤大小 | 9×9 |

---

## 3. UART 封包協議

### 0xAA — FPGA → PC：玩家落子通知
FPGA 偵測到人類落子後，傳送棋盤狀態給 PC 啟動 MCTS。

| 位元組 | 內容 |
|--------|------|
| [0] | 0xAA（起始標頭） |
| [1] | 0x01（固定） |
| [2]~[12] | 黑棋位置（81-bit 整數，低位元組先，11 bytes） |
| [13]~[23] | 白棋位置（81-bit 整數，低位元組先，11 bytes） |
| [24] | 0x55（結束標記） |

### 0xBB — FPGA → PC：CNN 推理結果
FPGA 完成一次 CNN 推理後，傳送 Policy + Value 給 PC。

| 位元組 | 內容 |
|--------|------|
| [0] | 0xBB（起始標頭） |
| [1]~[81] | Policy logits（81 個 INT8，對應 9×9 棋盤） |
| [82]~[83] | Value（INT16，大端序） |
| [84] | 0x55（結束標記） |

### 0xCC — PC → FPGA：CNN 評估請求
PC 每次 MCTS 模擬都傳送一個棋盤局面要求 FPGA 做 CNN 推理。

| 位元組 | 內容 |
|--------|------|
| [0] | 0xCC（起始標頭） |
| [1]~[11] | 當前玩家棋子（81-bit，11 bytes） |
| [12]~[22] | 對手棋子（81-bit，11 bytes） |
| [23] | last_move（0~80，255=無） |
| [24] | prev_move（0~80，255=無） |
| [25] | 0x55（結束標記） |

### 0xDD — PC → FPGA：AI 最終落子
PC 完成 MCTS 搜索後，傳送最佳落子給 FPGA 更新棋盤。

| 位元組 | 內容 |
|--------|------|
| [0] | 0xDD（起始標頭） |
| [1] | x 座標（0~8） |
| [2] | y 座標（0~8） |
| [3] | 0x55（結束標記） |

---

## 4. 遊戲流程

### FSM 狀態圖（game_fsm.sv）

```
上電/重置
    │
    ▼
┌─────────┐  btn_left/right 選模式
│ S_MENU  │──────────────────────────► game_mode
│ 選擇模式 │
└────┬────┘
     │ place_btn
     ▼
┌───────────────┐
│ S_WAIT_INPUT  │◄──────────────────────────────────────┐
│ 等待輸入       │                                       │
└───────┬───────┘                                       │
        │ (AI 模式且輪到白棋)                              │
        │ start_ai=1                                     │
        ▼                                               │
┌───────────────┐                                       │
│  S_AI_TURN   │                                       │
│  等待 AI 完成  │                                       │
└───────┬───────┘                                       │
        │ ai_done                                       │
        ▼ start_rule_check=1                            │
┌───────────────┐   不合法（人類）                        │
│ S_CHECK_RULES │───────────────────────────────────────┘
│  規則檢查      │
└───────┬───────┘
        │ is_legal_move
        ▼ update_board=1
┌───────────────┐
│ S_COMMIT_MOVE │
│  更新棋盤      │
└───────┬───────┘
        ▼
┌───────────────┐
│ S_SWITCH_TURN │
│  換邊          │
└───────┬───────┘
        │
        └───────────────────────────────────────────────┘

end_game_sw → S_SCORING → count_done → S_GAME_OVER
```

### game_mode 說明
| game_mode | 模式 |
|-----------|------|
| 2'b00 | 雙人對戰（黑棋與白棋均由人類控制） |
| 2'b01 | 人機對戰（黑棋=人類，白棋=AI） |

---

## 5. AI 回合流程

```
S_WAIT_INPUT（輪到白棋，AI 模式）
    │ start_ai=1
    ▼
go_ai_core.sv 啟動
    │
    ├── S_SEND_REQUEST：把目前棋盤用 0xAA 封包傳給 PC
    │
    ▼
PC 收到 0xAA
    │
    └── 開始 MCTS 搜索（200 次模擬）
            │
            └── 每次模擬：
                    1. 選擇（PUCT）
                    2. 傳送 0xCC 給 FPGA（要求 CNN 推理）
                    3. FPGA 收到 0xCC → S_LOAD_FEATURE → S_CNN_INFERENCE
                    4. FPGA 傳回 0xBB（Policy + Value）
                    5. 展開節點
                    6. Negamax backup

MCTS 完成 → PC 傳送 0xDD（最佳落子）
    │
    ▼
go_ai_core.sv 收到 0xDD → ai_done=1
    │
    ▼
game_fsm.sv：S_AI_TURN → S_CHECK_RULES → S_COMMIT_MOVE → S_SWITCH_TURN
```

---

## 6. CNN 網路架構

### 整體結構

```
輸入：[8, 9, 9] INT8
    │
    ▼
Entry Conv  (8→64, 3×3, BN, ReLU)
    │
    ▼
ResBlock 0  (64→64, 3×3, BN, ReLU) × 2 + skip
    │
    ▼
ResBlock 1  (64→64, 3×3, BN, ReLU) × 2 + skip
    │
    ├──────────────────────────────┐
    ▼ Policy Head                 ▼ Value Head
Conv 1×1 (64→2, BN, ReLU)        Conv 1×1 (64→1, BN, ReLU)
Flatten (→162)                   Flatten (→81)
FC (162→81)                      FC (81→64, ReLU)
Policy logits [81]                FC (64→1)
                                  Tanh
                                  Value scalar
```

### 各層參數（對應 FPGA layer_controller.sv）

| 層 | 輸入通道 | 輸出通道 | 核大小 | Shift | 說明 |
|----|---------|---------|--------|-------|------|
| Entry Conv | 8 | 64 | 3×3 | 6 | 入口卷積 |
| Tower0 Conv1 | 64 | 64 | 3×3 | 7 | ResBlock0 第1層 |
| Tower0 Conv2 | 64 | 64 | 3×3 | 7 | ResBlock0 第2層（+ skip） |
| Tower1 Conv1 | 64 | 64 | 3×3 | 7 | ResBlock1 第1層 |
| Tower1 Conv2 | 64 | 64 | 3×3 | 7 | ResBlock1 第2層（+ skip） |
| Policy Conv | 64 | 2 | 1×1 | 13 | Policy head 卷積 |
| Policy FC | 162 | 81 | — | 13 | Policy head 全連接 |
| Value Conv | 64 | 1 | 1×1 | 13 | Value head 卷積 |
| Value FC1 | 81 | 64 | — | 13 | Value head 全連接1 |
| Value FC2 | 64 | 1 | — | 13 | Value head 全連接2 |

### 8 通道輸入特徵（to_features_8ch）

| 通道 | 內容 |
|------|------|
| Ch0 | 當前玩家的棋子位置（127=有，0=無） |
| Ch1 | 對手的棋子位置 |
| Ch2 | 當前玩家氣數=1（即將被吃）的棋子 |
| Ch3 | 當前玩家氣數=2 的棋子 |
| Ch4 | 對手氣數=1 的棋子（可吃子） |
| Ch5 | 對手氣數=2 的棋子 |
| Ch6 | 上一手落子位置（127 填滿） |
| Ch7 | 上上手落子位置（127 填滿） |

---

## 7. CNN 推理流程

### conv_unit.sv 狀態機

每個輸出點（output channel × 空間位置）都按以下順序執行：

```
S_IDLE
  │ start
  ▼
S_INIT_BIAS ──► S_WAIT_BIAS ──► S_INIT_ACC
（設定 bias_addr）  （等 BRAM）    acc = sign_extend(bias_data)
                                    │
                     ┌──────────────┘
                     │  對每個 (ic, kr, kc) tap 重複：
                     ▼
              S_LOAD_ADDR（設定 fmap_in_addr、weight_addr）
                     │
                     ▼
              S_WAIT_MEM（等 BRAM，2 cycles）
                     │
                     ▼
              S_CAPTURE_DATA（鎖存 fmap_in_data、weight_data）
                     │
                     ▼
              S_MAC（acc += captured_weight × captured_fmap）
                     │
                     ▼
              S_NEXT_TAP──► 還有 tap → 回 S_LOAD_ADDR
                     │
                     │ 所有 tap 完成
                     ▼
              S_FINISH_POINT
              （result = (acc >> shift) + skip；ReLU；飽和到 INT8；寫入 fmap_out）
                     │
                     ▼
              S_NEXT_POSITION──► 還有位置 → 回 S_INIT_BIAS（載入下一個 oc 的 bias）
                     │
                     │ 所有輸出點完成
                     ▼
              S_DONE
```

**Bias 載入機制：**
- `bias_addr = cnt_oc`（每個輸出通道對應一個 bias 值）
- `S_INIT_BIAS → S_WAIT_BIAS`：等 BRAM 讀取延遲（2 cycles）
- `S_INIT_ACC`：`acc = {{24{bias_data[7]}}, bias_data}`（符號擴展到 32-bit）
- 後續所有 MAC 結果都累加在已含 bias 的 acc 上
- **Bias ROM 對應表：**

| 層 | Bias ROM | 備註 |
|----|----------|------|
| Entry（L0） | `rom_entry_bias` | 64 個 INT8 bias |
| Tower0 Conv1（L1） | `rom_tower0_conv1_bias` | 64 個 |
| Tower0 Conv2（L2） | `rom_tower0_conv2_bias` | 64 個 |
| Tower1 Conv1（L3） | `rom_tower1_conv1_bias` | 64 個 |
| Tower1 Conv2（L4） | `rom_tower1_conv2_bias` | 64 個 |
| Policy Conv（L5） | 無 ROM，固定 0 | PyTorch `bias=False`（BN 後接），正確 |
| Value Conv（L7） | 無 ROM，固定 0 | PyTorch `bias=False`（BN 後接），正確 |

**關鍵位址計算：**
- 特徵圖讀取：`fmap_in_addr = {1'b0, cnt_ic[5:0], position_in}`（14-bit）
- 3×3 權重：`weight_addr = cnt_oc × in_ch × 9 + cnt_ic × 9 + cnt_kr × 3 + cnt_kc`
- 1×1 權重：`weight_addr = cnt_oc × in_ch + cnt_ic`

**Skip Connection：**
- `safe_skip_val = {24'd0, skip_rdata}`（零擴展）
- 因為 skip 來源是 ReLU 輸出（≥0，bit[7]=0），零擴展等同符號擴展，結果正確

### fcunit.sv 流程

每個輸出神經元按以下順序執行：

```
S_IDLE
  │ start
  ▼
S_INIT_BIAS ──► S_WAIT_BIAS ──► S_INIT_ACC
（設定 bias_addr）  （等 BRAM）    acc = sign_extend(bias_data)
                                    │
                     ┌──────────────┘
                     │  對每個輸入 cnt_i 重複：
                     ▼
              S_LOAD_WEIGHT（設定 weight_addr、fmap_in_addr）
                     │
                     ▼
              S_WAIT_DATA（等 BRAM，2 cycles）
                     │
                     ▼
              S_MAC（acc += weight_data × fmap_in_data）
                     │
                     ▼
              S_NEXT_INPUT──► 還有輸入 → 回 S_LOAD_WEIGHT
                     │
                     │ 所有輸入完成
                     ▼
              S_FINISH_OUTPUT
              （result = acc >> shift；ReLU；飽和到 INT8；寫入 fmap_out）
                     │
                     ▼
              S_NEXT_OUTPUT──► 還有輸出神經元 → 回 S_INIT_BIAS
                     │
                     │ 所有輸出完成
                     ▼
              S_DONE
```

**FC Bias ROM 對應表：**

| 層 | Bias ROM | 備註 |
|----|----------|------|
| Policy FC（L6） | `rom_policy_fc_bias` | 81 個 INT8 bias |
| Value FC1（L8） | `rom_value_fc1_bias` | 64 個 INT8 bias |
| Value FC2（L9） | 無 ROM，固定 0 | 訓練時 bias 接近 0，影響極小 |

**Policy FC 位址映射（162 個輸入 = 2 通道 × 81）：**
- cnt_i < 81：讀 `{ch=0, pos=cnt_i}`
- cnt_i ≥ 81：讀 `{ch=1, pos=cnt_i-81}`

---

### Shift 的用途與計算方式

#### 為什麼需要 Shift？

CNN 的每一層推理都是 INT8 × INT8 的乘法累加（MAC）。兩個 INT8（範圍 -128 ～ 127）相乘後結果為 16-bit，加上幾千次累加後 acc 會擴展到 INT32。但下一層的輸入必須仍是 INT8（-128 ～ 127），所以需要把 INT32 的 acc **縮回 INT8 的範圍**，這個縮放動作就是 **算術右移（arithmetic right shift）**：

```
result_int8 = acc >>> scale_shift
```

右移 N 位等同於除以 2^N，並保留符號（負數仍是負數）。

#### Shift 值如何計算？

每一層的 shift 值由訓練後的 **量化 scale** 決定，計算公式（見 `simulate.py`）：

```python
def calc_shift(w_scale, in_scale, out_scale=1/127):
    combined = w_scale * in_scale      # 乘法後實際的數值範圍
    ratio    = combined / out_scale    # 相對於 INT8 輸出範圍的倍數
    return round(-log2(ratio))         # 用 2 的冪次近似，取整數位移量
```

- `w_scale`：該層權重的最大絕對值 / 127（量化時的縮放比例）
- `in_scale`：輸入特徵圖的 scale（通常等於上一層的 out_scale，即 1/127）
- `out_scale`：輸出 INT8 所對應的實際數值範圍，固定為 1/127

**直觀理解：**
- 若兩個 scale 都是 1/127，乘法後數值範圍變成 1/127²
- 要把結果映射回 1/127，需要乘以 127 ≈ 2^7，等價於右移 7 位
- 所以 shift ≈ 7 是 Tower 層的典型值

#### 各層 Shift 值一覽

| 層 | Shift | 說明 |
|----|-------|------|
| Entry Conv | 6 | 輸入是 0/127 的二值特徵，scale 較大，shift 稍小 |
| Tower0/1 Conv1/Conv2 | 7 | 標準 64 通道卷積，scale 接近 1/127 |
| Policy Conv（1×1） | 7 | 同上 |
| Value Conv（1×1） | 7 | 同上 |
| Policy FC | 6 | 輸入通道數多（162），累加量大，shift 稍小 |
| Value FC1 | 7 | 標準全連接 |
| Value FC2 | 8 | 輸出為單一 scalar，scale 較小，shift 稍大 |

#### Shift 在硬體中的位置

```
acc（INT32）= bias + Σ(weight × input)
                    │
                    ▼  acc >>> scale_shift
             shifted（INT32，數值已縮回 INT8 範圍）
                    │
                    ▼  + skip（若為 ResBlock 第2層）
             pre_relu_val
                    │
                    ▼  ReLU + 飽和截斷
             result_int8（INT8，-128 ～ 127）
```

Shift 的時機是**在加 skip 之前**，這樣 skip 的數值範圍（已是 INT8）才能與 shifted 直接相加而不溢位。

---

### 為什麼 Shift 不影響模型準確度？

#### 1. Shift 是「有依據的近似」，不是隨機誤差

Shift 值由 `calc_shift` 從訓練好的權重反推，使用 `round(-log2(ratio))` 找最接近真實縮放比例的 2 的冪次。例如真實比例需要除以 100，選 2^7 = 128，誤差只有 28%。這個誤差最大不超過 √2 倍（log2 空間的 ±0.5 bit），是**有界的**。

#### 2. Shift 是「一致性偏差」，不是「隨機雜訊」

| 類型 | 特性 | 對模型的影響 |
|------|------|-------------|
| 隨機雜訊 | 每次結果不同 | 破壞輸出的可重現性，排序可能改變 |
| Shift（系統性偏差） | 每次結果完全相同 | 整體縮放，**相對大小完全保留** |

同一個局面每次跑 FPGA 結果完全一致，神經網路學到的排序關係完全保留。

#### 3. 模型準確度依賴「相對大小」，不依賴「絕對數值」

**Policy head（落子機率）：**
```
logits = [10, 5, 8, 3, ...]  →  softmax  →  最大值對應的棋步
```
即使 shift 讓所有 logits 縮小 2 倍變成 `[5, 2.5, 4, 1.5, ...]`，softmax 後排序完全不變，AI 選的棋步一樣。

**Value head（勝負評估）：**
```
value ∈ [-1, 1]  →  MCTS 只需要判斷正負方向與相對大小
```
只要符號與排序正確，MCTS backup 的方向就正確，搜索結果不受影響。

#### 4. INT8 的 256 個等級對 9×9 圍棋足夠

Policy head 只需要比較 81 個 logit 的大小，INT8 的 256 個等級提供的精度遠超過這個需求，量化誤差造成排序改變的機率極低。

#### 真正會影響準確度的情況

| 原因 | 說明 |
|------|------|
| 權重有極端離群值 | 超出 scale 範圍的權重被截斷到 ±127，資訊遺失 |
| Shift 值選太大 | 有效數值變成 0，例如原本是 1 右移 8 位後消失 |
| Shift 值選太小 | 結果超出 INT8 範圍，飽和截斷到 ±127，多個不同輸入映射到同一輸出 |

`calc_shift` 使用 `round`（四捨五入）確保選到最接近的 shift，把上述風險壓到最低。這也是為什麼量化前需要先確認 `scales.json` 的數值合理，而不是直接猜一個 shift 硬填。

---

## 8. MCTS 演算法

### MCTSNode 資料結構

```python
class MCTSNode:
    board        # 當前棋盤狀態
    color        # 上一個落子的顏色（1=黑，2=白）
    parent       # 父節點
    move         # 到達此節點的落子
    prior        # CNN 給出的先驗機率 P
    visit_count  # 訪問次數 N
    value_sum    # 累計價值 W
    children     # 子節點字典 {(x,y): MCTSNode}
    is_expanded  # 是否已展開
```

### PUCT 公式

```
UCB = Q(s,a) + C_PUCT × P(s,a) × sqrt(N(s)) / (1 + N(s,a))

Q(s,a) = value_sum / visit_count   （若 visit_count=0 則 Q=0）
C_PUCT = 1.5
```

### 一次 MCTS 模擬流程

```
1. 選擇（Selection）
   從 root 出發，每次選 UCB 最高的子節點，直到未展開的節點

2. 評估（Evaluation）
   把當前節點的棋盤用 0xCC 傳給 FPGA
   收到 0xBB：Policy probs 陣列 + Value 純量

3. 展開（Expansion）
   對所有合法落子建立子節點，用 CNN Policy 作為 prior

4. 反傳（Backup）— Negamax 零和反轉
   node.value_sum += v
   v = -v（每往上一層就取反，體現零和對弈）
```

### CNN 輸出解碼（decode_fpga_output）

```python
# Policy
logits = fpga_int8 / 127.0 × 4.0   # × 4.0 是溫度銳化
logits -= max(logits)
probs = softmax(logits)

# Value
val_float = fpga_int16 / 127.0
value = tanh(val_float × 2.0)       # 映射到 [-1, 1]
```

### 最終落子選擇

MCTS 完成後選**訪問次數最多**的子節點（貪心，無溫度參數）：
```python
best_move = max(root.children, key=lambda n: n.visit_count)
```

---

## 9. 訓練流程

### 資料來源
- SGF 棋譜壓縮檔（最多 34,572 局）
- 資料集類別：`GoDataset`（繼承 `torch.utils.data.Dataset`）

### 8 通道特徵提取（to_features_8ch）
1. 從 SGF 解析每一手棋
2. 對每個局面，以「當前玩家視角」建立 8 通道特徵
3. 標籤：policy = 實際落子位置（one-hot），value = 最終勝負（±1）

### 訓練設定
| 參數 | 值 |
|------|----|
| Epochs | 30 |
| Batch size | 1024 |
| 學習率 | 1e-3（CosineAnnealingLR） |
| 損失函數 | CE（Policy） + MSE（Value） |
| 優化器 | Adam |

### INT8 量化與匯出流程

```
訓練完成的 FP32 模型
    │
    ▼
BN 融合（_fuse_bn）
  entry[0] + entry[1] → fused Conv2d
  tower ResBlock 各層 BN 全部融合
    │
    ▼
逐層計算 scale = max(|w|) / 127
    │
    ▼
INT8 量化：q = round(w / scale)，clip 到 [-128, 127]
    │
    ▼
轉為 8-bit 二進制字串，寫成 .mem 檔（給 Verilog $readmemb 讀取）
    │
    ▼
scales.json（記錄每層 scale 值，FPGA shift 計算用）
```

### Shift 計算公式（simulate.py）
```python
shift = round(-log2(w_scale × in_scale / out_scale))
out_scale = 1/127
```

---

## 10. FPGA 模組說明

### 頂層與控制

| 模組 | 檔案 | 功能 |
|------|------|------|
| top | top.sv | 頂層連接：時脈、按鍵去彈、所有子模組 |
| game_fsm | game_fsm.sv | 主遊戲 FSM（8 個狀態，控制遊戲流程） |
| go_ai_core | go_ai_core.sv | HIL 引擎：FPGA↔PC UART 通訊，驅動 CNN 推理 |

### CNN 推理引擎

| 模組 | 檔案 | 功能 |
|------|------|------|
| cnn_engine | cnn_engine.sv | CNN 頂層排程（呼叫 layer_controller） |
| layer_controller | layer_controller.sv | 10 層 CNN 依序執行，管理 skip 連接 |
| conv_unit | conv_unit.sv | INT8 卷積運算單元（3×3 與 1×1） |
| fcunit | fcunit.sv | INT8 全連接層（Policy FC、Value FC1/FC2） |

### 遊戲邏輯

| 模組 | 檔案 | 功能 |
|------|------|------|
| board_manager | board_manager.sv | 棋盤狀態儲存、落子、提子 |
| rule_engine | rule_engine.sv | 合法性判斷（佔位、自殺、眼位） |
| group_liberty_scanner | group_liberty_scanner.sv | BFS 洪水填充，計算棋群氣數 |
| territory_counter | territory_counter.sv | 終局數子（中國規則面積計分） |
| board_liberty_builder | board_liberty_builder.sv | 建立 8 通道 CNN 輸入（氣數特徵） |

### 通訊介面

| 模組 | 檔案 | 功能 |
|------|------|------|
| uart_rx | uart_rx.sv | UART 接收（CLKS_PER_BIT=868，100MHz/115200） |
| uart_tx | uart_tx.sv | UART 發送 |
| packet_parser | packet_parser.sv | 解析 0xCC（26 bytes）與 0xDD（4 bytes）封包 |
| packet_builder | packet_builder.sv | 組裝 0xAA（25 bytes）與 0xBB（85 bytes）封包 |

### 顯示與輸入

| 模組 | 檔案 | 功能 |
|------|------|------|
| vga_controller | vga_controller.sv | VGA 640×480 @25MHz：選單、棋盤、棋子、游標 |
| button_debouncer | button_debouncer.sv | 按鍵去彈（20-bit 計數，約 10ms） |
| clk_divider | clk_divider.sv | 100MHz → 25MHz（2-bit 計數器） |

---

## 11. Python 檔案說明

| 檔案 | 功能 |
|------|------|
| pcaitest.py | **主程式**：MCTS + FPGA HIL 對局，完整 MCTSNode、PUCT 搜索、UART 通訊 |
| train.py | **訓練程式**：8 通道特徵、2 個 ResBlock、INT8 量化匯出 .mem 與 scales.json |
| simulate.py | **推理模擬**：在 PC 上模擬 FPGA 的 INT8 CNN 推理，用於驗證權重正確性 |

---

## 12. 權重與 Scale 檔案

### scales.json 各層 shape

| 鍵名 | Shape | 說明 |
|------|-------|------|
| entry_0_weight | [64, 8, 3, 3] | 入口卷積權重（8 通道輸入） |
| entry_0_bias | [64] | 入口卷積 bias（BN 融合後） |
| tower_0_net_0_weight | [64, 64, 3, 3] | ResBlock0 第1層 |
| tower_0_net_3_weight | [64, 64, 3, 3] | ResBlock0 第2層 |
| tower_1_net_0_weight | [64, 64, 3, 3] | ResBlock1 第1層 |
| tower_1_net_3_weight | [64, 64, 3, 3] | ResBlock1 第2層 |
| policy_head_0_weight | [2, 64, 1, 1] | Policy head 卷積 |
| policy_head_4_weight | [81, 162] | Policy FC |
| policy_head_4_bias | [81] | Policy FC bias |
| value_head_0_weight | [1, 64, 1, 1] | Value head 卷積 |
| value_head_4_weight | [64, 81] | Value FC1 |
| value_head_4_bias | [64] | Value FC1 bias |
| value_head_6_weight | [1, 64] | Value FC2 |

### .mem 檔格式
- 每行一個 INT8 數值，以**8-bit 二進制字串**表示
- 由 Verilog `$readmemb` 讀取進 BRAM ROM
- 路徑：`Final/weights/`

