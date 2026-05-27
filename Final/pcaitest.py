import serial, time, math, numpy as np

PORT       = 'COM6'
BAUD       = 115_200
TIMEOUT    = 30
N_SIMS     = 200
C_PUCT     = 1.5
BOARD_SIZE = 9

class GoBoard:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.ko_point = None

    def copy(self):
        b = GoBoard(self.size)
        b.board = self.board.copy()
        b.ko_point = self.ko_point
        return b

    def get_group_and_liberties(self, x, y):
        color = self.board[y][x]
        if color == 0: return set(), set()
        group, liberties, frontier = set(), set(), [(x, y)]
        while frontier:
            cx, cy = frontier.pop()
            if (cx, cy) in group: continue
            group.add((cx, cy))
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    nc = self.board[ny][nx]
                    if nc == color and (nx, ny) not in group: frontier.append((nx, ny))
                    elif nc == 0: liberties.add((nx, ny))
        return group, liberties

    def is_legal(self, x, y, color):
        if not (0 <= x < self.size and 0 <= y < self.size): return False
        if self.board[y][x] != 0: return False
        if self.ko_point == (x, y): return False
        test = self.copy()
        test.board[y][x] = color
        opp = 2 if color == 1 else 1
        captured = []
        for nx, ny in test._neighbors(x, y):
            if test.board[ny][nx] == opp:
                grp, libs = test.get_group_and_liberties(nx, ny)
                if not libs:
                    for dx, dy in grp: test.board[dy][dx] = 0
        _, my_libs = test.get_group_and_liberties(x, y)
        if not my_libs: return False
        return True

    def play_move(self, x, y, color):
        if not self.is_legal(x, y, color): return False
        self.board[y][x] = color
        opp = 2 if color == 1 else 1
        captured = []
        for nx, ny in self._neighbors(x, y):
            if self.board[ny][nx] == opp:
                grp, libs = self.get_group_and_liberties(nx, ny)
                if not libs:
                    for dx, dy in grp: 
                        self.board[dy][dx] = 0
                        captured.append((dx, dy))
        self.ko_point = captured[0] if len(captured) == 1 else None
        return True

    def get_legal_moves(self, color):
        return [(c, r) for r in range(self.size) for c in range(self.size) if self.is_legal(c, r, color)]

    def _neighbors(self, x, y):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            if 0 <= x+dx < self.size and 0 <= y+dy < self.size: yield x+dx, y+dy

class MCTSNode:
    def __init__(self, board, color, parent=None, move=None, prior=0.0):
        self.board = board
        self.color = color 
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False

    @property
    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def ucb_score(self, c_puct):
        if self.parent is None: return 0.0
        return self.q_value + (c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count))

    def select_child(self, c_puct):
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))

    def expand(self, probabilities):
        next_color = 2 if self.color == 1 else 1
        legal_moves = self.board.get_legal_moves(next_color)
        if not legal_moves:
            self.is_expanded = True
            return
        
        priors = np.array([probabilities[r * 9 + c] for c, r in legal_moves])
        if priors.sum() > 0: priors /= priors.sum()
        else: priors = np.ones(len(legal_moves)) / len(legal_moves) # 防呆

        for i, (x, y) in enumerate(legal_moves):
            nb = self.board.copy()
            nb.play_move(x, y, next_color)
            self.children[(x, y)] = MCTSNode(board=nb, color=next_color, parent=self, move=(x, y), prior=priors[i])
        self.is_expanded = True

    def backup(self, value):
        # 🌟 標準 Negamax 零和反轉：子節點的勝率 = 父節點的敗率
        node = self
        v = -value
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            v = -v
            node = node.parent

def decode_fpga_output(fpga_policy_int8, fpga_value_int8):
    logits = fpga_policy_int8.astype(np.float64) / 127.0
    val_float = float(fpga_value_int8) / 127.0
    logits = logits * 4.0 
    logits -= np.max(logits)
    exp_p = np.exp(logits)
    return exp_p / np.sum(exp_p), np.tanh(val_float * 2.0)

# 🌟 解決身分認同 Bug：把當前玩家的石頭固定塞進 Ch0 (bb)
def pack_cc_8ch(board, current_color, last_move=255, prev_move=255):
    my_color = current_color
    op_color = 2 if current_color == 1 else 1

    my_flat = (board.board == my_color).flatten().astype(np.int32)
    op_flat = (board.board == op_color).flatten().astype(np.int32)

    bb = sum(int(my_flat[i]) << i for i in range(81))
    wb = sum(int(op_flat[i]) << i for i in range(81))

    buf = bytearray(26)
    buf[0] = 0xCC
    for i in range(11):
        buf[1+i]  = (bb >> (i*8)) & 0xFF
        buf[12+i] = (wb >> (i*8)) & 0xFF
    buf[23], buf[24], buf[25] = int(last_move) & 0xFF, int(prev_move) & 0xFF, 0x55
    return bytes(buf)

def evaluate_fpga(ser, board, current_color, lm, pm):
    # 🌟 UART 防錯機制：清空殘留的 0xDD 垃圾
    ser.reset_input_buffer()
    ser.write(pack_cc_8ch(board, current_color, lm, pm))
    
    # 耐心等待 0xBB (防止 FPGA 沒改好還在噴 0xDD)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        h = ser.read(1)
        if not h: continue
        if h[0] == 0xBB:
            rest = ser.read(84)
            if len(rest) == 84 and rest[-1] == 0x55:
                pol_int8 = np.array([rest[i] if rest[i]<128 else rest[i]-256 for i in range(81)], dtype=np.int8)
                val_int16 = (rest[81] << 8) | rest[82]
                if val_int16 >= 32768: val_int16 -= 65536
                return decode_fpga_output(pol_int8, val_int16)
    return None, None

def mcts_search(ser, root_board, root_lm, root_pm, n_sims=N_SIMS):
    root = MCTSNode(board=root_board, color=1)
    t_start = time.time()
    print(f"\n  [MCTS] 開始搜索（{n_sims}次）...")

    success_sims = 0
    for sim in range(n_sims):
        node = root
        while node.is_expanded and node.children: node = node.select_child(C_PUCT)

        if node.parent is None: node_lm, node_pm = root_lm, root_pm
        else:
            node_lm = node.move[1] * 9 + node.move[0]
            node_pm = root_lm if node.parent.parent is None else node.parent.move[1] * 9 + node.parent.move[0]

        next_color = 2 if node.color == 1 else 1
        probs, value = evaluate_fpga(ser, node.board, next_color, node_lm, node_pm)
        
        if probs is None: 
            print("⚠️ FPGA 讀取超時或失敗！", end='\r')
            continue
            
        success_sims += 1
        if not node.is_expanded: node.expand(probs)
        node.backup(value)

        if (sim + 1) % 50 == 0: print(f"  [MCTS] {sim+1:3d}/{n_sims} 耗時 {time.time()-t_start:.1f}s")

    print(f"  [MCTS] 完成 {success_sims} 次有效評估，總耗時 {time.time()-t_start:.1f}s")
    if not root.children: return None
    
    best_move, _ = max(root.children.items(), key=lambda i: i[1].visit_count)
    print(f"\n  [MCTS] Top 5 推薦：")
    for (x, y), n in sorted(root.children.items(), key=lambda i: i[1].visit_count, reverse=True)[:5]:
        print(f"    {'✅' if root_board.is_legal(x,y,2) else '❌'} ({x},{y}) 訪問={n.visit_count:3d} Q={n.q_value:+.3f} P={n.prior:.3f}")
    return best_move

def get_diff_move(old_b, new_b):
    y, x = np.where((new_b.board - old_b.board != 0) & (new_b.board != 0) & (old_b.board == 0))
    return x[0] + y[0] * 9 if len(y) > 0 else 255

def print_board(b):
    print("    " + " ".join(str(c) for c in range(9)) + "\n   " + "──"*9)
    for r in range(9):
        print(f"{r}: " + " ".join("●" if b.board[r][c]==1 else "○" if b.board[r][c]==2 else "." for c in range(9)))

def parse_aa(data):
    if len(data) != 25 or data[0] != 0xAA or data[24] != 0x55: return None
    bb, wb = sum(data[2+i]<<(i*8) for i in range(11)), sum(data[13+i]<<(i*8) for i in range(11))
    board = GoBoard()
    for i in range(81):
        if (bb >> i) & 1: board.board[i//9][i%9] = 1
        elif (wb >> i) & 1: board.board[i//9][i%9] = 2
    return board

def run():
    print("="*60 + f"\n  Mini AlphaGo 實戰模式！ (MCTS×{N_SIMS} + FPGA CNN)\n" + "="*60)
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        time.sleep(0.5); ser.reset_input_buffer()
        print(f"✅ 連接 {PORT} 成功\n")
    except Exception as e: return print(f"❌ {e}")

    move_count, lm, pm, prev_b = 0, 255, 255, GoBoard()

    while True:
        try:
            print("[等待FPGA 0xAA 玩家落子...]", end='\r')
            h = ser.read(1)
            if not h or h[0] != 0xAA: continue
            rest = ser.read(24)
            if len(rest) != 24: continue

            board = parse_aa(bytes([0xAA]) + rest)
            if board is None: continue

            ts = (board.board != 0).sum()
            if ts <= 1:
                lm, pm = (np.where(board.board!=0)[1][0] + np.where(board.board!=0)[0][0]*9, 255) if ts == 1 else (255, 255)
            else:
                diff = get_diff_move(prev_b, board)
                if diff != 255: pm, lm = lm, diff

            move_count += 1
            print(f"\n{'='*60}\n[第 {move_count} 回合] 黑:{(board.board==1).sum()} 白:{(board.board==2).sum()} | 軌跡 LM:{lm} PM:{pm}")
            print_board(board)

            best = mcts_search(ser, board, lm, pm, N_SIMS)
            if best is None:
                print("  ⚠️ 無合法落子，Pass")
                best = (4, 4) # 避免死機，雖然理論上不會發生

            x, y = best
            print(f"\n  ★ AI 落子：({x}, {y})")
            ser.write(bytes([0xDD, x&0xFF, y&0xFF, 0x55]))
            
            pm, lm = lm, x + y * 9
            prev_b = board.copy()
            prev_b.play_move(x, y, 2)

        except KeyboardInterrupt: break
    ser.close()

if __name__ == '__main__': run()