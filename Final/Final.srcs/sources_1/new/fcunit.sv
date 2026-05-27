`timescale 1ns / 1ps
// ============================================================
// fc_unit.sv (修正版)
// ============================================================
// 全連接層（Fully Connected Layer）
//
// 支援三種層：
//   Policy FC  : 2ch×9×9=162 → 81  (input_is_spatial=1, in_features=162)
//   Value  FC1 : 1ch×9×9=81  → 64  (input_is_spatial=1, in_features=81)
//   Value  FC2 : 64ch×1×1=64 → 1   (input_is_spatial=0, in_features=64)
//
// 地址對應：
//   input_is_spatial=1, in_features=162 (Policy FC)：
//     cnt_i=0~80   → fmap addr = {ch=0, pos=cnt_i}
//     cnt_i=81~161 → fmap addr = {ch=1, pos=cnt_i-81}
//
//   input_is_spatial=1, in_features=81 (Value FC1)：
//     cnt_i=0~80   → fmap addr = {ch=0, pos=cnt_i}
//
//   input_is_spatial=0, in_features=64 (Value FC2)：
//     cnt_i=0~63   → fmap addr = {ch=cnt_i, pos=0}
//
// 輸出地址：
//   output[oc] → fmap addr = {ch=oc, pos=0}
// ============================================================

module fc_unit (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    output logic         done,

    // 配置
    input  logic [7:0]   in_features,      // 輸入維度（162/81/64）
    input  logic [6:0]   out_features,     // 輸出維度（81/64/1）
    input  logic         use_relu,
    input  logic [4:0]   scale_shift,
    input  logic         input_is_spatial, // 1=Policy/ValueFC1, 0=ValueFC2

    // 讀取輸入 fmap
    output logic [13:0]  fmap_in_addr,
    input  logic [7:0]   fmap_in_data,

    // 寫入輸出 fmap
    output logic         fmap_out_we,
    output logic [13:0]  fmap_out_addr,
    output logic [7:0]   fmap_out_data,

    // 讀取權重：weight[oc][i] = mem[oc * in_features + i]
    output logic [15:0]  weight_addr,
    input  logic [7:0]   weight_data,

    // 讀取bias
    output logic [6:0]   bias_addr,
    input  logic [7:0]   bias_data,

    // Debug
    output logic [3:0]   debug_state
);

    // ─────────────────────────────────────
    // 狀態機
    // ─────────────────────────────────────
    typedef enum logic [3:0] {
        S_IDLE,
        S_INIT_BIAS,
        S_WAIT_BIAS,
        S_INIT_ACC,
        S_LOAD_WEIGHT,
        S_WAIT_DATA,
        S_MAC,
        S_NEXT_INPUT,
        S_FINISH_OUTPUT,
        S_NEXT_OUTPUT,
        S_DONE
    } state_t;

    state_t state;
    assign debug_state = state;

    // ─────────────────────────────────────
    // 計數器
    // ─────────────────────────────────────
    logic [6:0] cnt_oc;   // 輸出維度計數 (0~out_features-1)
    logic [7:0] cnt_i;    // 輸入維度計數 (0~in_features-1)

    // 累加器
    logic signed [31:0] acc;

    // 反量化 + ReLU
    logic signed [31:0] shifted;
    logic signed [7:0]  result_int8;

    always_comb begin
        shifted = acc >>> scale_shift;
        if (use_relu && shifted < 0)
            result_int8 = 8'sd0;
        else if (shifted > 32'sd127)
            result_int8 = 8'sd127;
        else if (shifted < -32'sd128)
            result_int8 = -8'sd128;
        else
            result_int8 = shifted[7:0];
    end

    // ─────────────────────────────────────
    // 地址計算（組合邏輯）
    // ─────────────────────────────────────
    logic [6:0] fc_ch, fc_pos;

    always_comb begin
        // 輸入地址
        if (input_is_spatial) begin
            if (in_features == 8'd162) begin
                // Policy FC：2ch×9×9
                // cnt_i 0~80  → ch=0, pos=cnt_i
                // cnt_i 81~161 → ch=1, pos=cnt_i-81
                if (cnt_i < 8'd81) begin
                    fc_ch  = 7'd0;
                    fc_pos = cnt_i[6:0];
                end else begin
                    fc_ch  = 7'd1;
                    fc_pos = cnt_i[6:0] - 7'd81;
                end
            end else begin
                // Value FC1：1ch×9×9
                // cnt_i 0~80 → ch=0, pos=cnt_i
                fc_ch  = 7'd0;
                fc_pos = cnt_i[6:0];
            end
            fmap_in_addr = {fc_ch, fc_pos};
        end else begin
            // Value FC2：64ch×1×1
            // cnt_i 0~63 → ch=cnt_i, pos=0
            fmap_in_addr = {cnt_i[6:0], 7'd0};
        end

        // 輸出地址：ch=cnt_oc, pos=0
        fmap_out_addr = {cnt_oc[6:0], 7'd0};

        // weight地址：oc * in_features + i
        weight_addr = {9'd0, cnt_oc} * {8'd0, in_features} + {8'd0, cnt_i};

        // bias地址
        bias_addr = cnt_oc;
    end

    // ─────────────────────────────────────
    // 主狀態機
    // ─────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            cnt_oc        <= '0;
            cnt_i         <= '0;
            acc           <= '0;
            fmap_out_we   <= 1'b0;
            fmap_out_data <= '0;
        end else begin
            done        <= 1'b0;
            fmap_out_we <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        cnt_oc <= '0;
                        cnt_i  <= '0;
                        state  <= S_INIT_BIAS;
                    end
                end

                // 等bias讀回（2 clk）
                S_INIT_BIAS: state <= S_WAIT_BIAS;
                S_WAIT_BIAS: state <= S_INIT_ACC;

                S_INIT_ACC: begin
                    acc   <= $signed({{24{bias_data[7]}}, bias_data});
                    cnt_i <= '0;
                    state <= S_LOAD_WEIGHT;
                end

                // 等weight和input讀回（2 clk）
                S_LOAD_WEIGHT: state <= S_WAIT_DATA;
                S_WAIT_DATA:   state <= S_MAC;

                S_MAC: begin
                    acc   <= acc + ($signed(weight_data) * $signed(fmap_in_data));
                    state <= S_NEXT_INPUT;
                end

                S_NEXT_INPUT: begin
                    if (cnt_i < in_features - 1) begin
                        cnt_i <= cnt_i + 1'b1;
                        state <= S_LOAD_WEIGHT;
                    end else begin
                        state <= S_FINISH_OUTPUT;
                    end
                end

                S_FINISH_OUTPUT: begin
                    fmap_out_we   <= 1'b1;
                    fmap_out_data <= result_int8;
                    state         <= S_NEXT_OUTPUT;
                end

                S_NEXT_OUTPUT: begin
                    if (cnt_oc < out_features - 1) begin
                        cnt_oc <= cnt_oc + 1'b1;
                        cnt_i  <= '0;
                        state  <= S_INIT_BIAS;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule