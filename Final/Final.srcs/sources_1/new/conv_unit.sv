`timescale 1ns / 1ps
// ============================================================
// Module      : conv_unit
// Description : 穩定 Pipeline 版 (修正 BRAM 延遲與乘法溢位)
// ============================================================

module conv_unit (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    output logic         done,
    
    input  logic [6:0]   in_ch,
    input  logic [6:0]   out_ch,
    input  logic [1:0]   kernel_size,
    input  logic         use_relu,
    input  logic [4:0]   scale_shift,
    
    output logic [13:0]  fmap_in_addr,
    input  logic [7:0]   fmap_in_data,
    
    output logic         fmap_out_we,
    output logic [13:0]  fmap_out_addr,
    output logic [7:0]   fmap_out_data,
    
    output logic [15:0]  weight_addr,
    input  logic [7:0]   weight_data,
    output logic [6:0]   bias_addr,       
    input  logic [7:0]   bias_data,

    input  logic         do_skip_add,
    output logic [13:0]  skip_raddr,
    input  logic [7:0]   skip_rdata,

    output logic [3:0]   debug_state
);

    // =========================================================
    // 穩定 Pipeline 狀態機 (拆解讀/等/抓/算/寫)
    // =========================================================
    typedef enum logic [3:0] {
        S_IDLE,
        S_INIT_BIAS,
        S_WAIT_BIAS,
        S_INIT_ACC,
        S_LOAD_ADDR,    // 送出位址
        S_WAIT_MEM,     // 等待 BRAM 延遲
        S_CAPTURE_DATA, // 鎖存乾淨資料
        S_MAC,          // 安全乘加
        S_NEXT_TAP,
        S_FINISH_POINT, 
        S_NEXT_POSITION,
        S_DONE
    } state_t;

    state_t state;
    assign debug_state = state;

    logic [6:0] cnt_oc;
    logic [3:0] cnt_r, cnt_c;
    logic [6:0] cnt_ic;
    logic [1:0] cnt_kr, cnt_kc;

    // 資料鎖存器 (隔離毛刺與時序違規)
    logic signed [7:0]  captured_fmap;
    logic signed [7:0]  captured_weight;
    logic signed [15:0] mult_res;         

    logic signed [31:0] acc;
    logic signed [31:0] shifted;
    logic signed [31:0] pre_relu_val;
    logic signed [7:0]  result_int8;

    logic signed [4:0]  tap_r, tap_c;
    logic               in_bounds;
    logic [6:0]         position, position_in;

    // =========================================================
    // 位址轉換邏輯
    // =========================================================
    always_comb begin
        if (kernel_size == 2'd3) begin
            tap_r = $signed({1'b0, cnt_r}) + $signed({3'b0, cnt_kr}) - 5'sd1;
            tap_c = $signed({1'b0, cnt_c}) + $signed({3'b0, cnt_kc}) - 5'sd1;
        end else begin
            tap_r = $signed({1'b0, cnt_r});
            tap_c = $signed({1'b0, cnt_c});
        end
        
        in_bounds = (tap_r >= 0) && (tap_r < 9) && (tap_c >= 0) && (tap_c < 9);

        case (cnt_r)
            4'd0: position = 7'd0  + cnt_c;
            4'd1: position = 7'd9  + cnt_c;
            4'd2: position = 7'd18 + cnt_c;
            4'd3: position = 7'd27 + cnt_c;
            4'd4: position = 7'd36 + cnt_c;
            4'd5: position = 7'd45 + cnt_c;
            4'd6: position = 7'd54 + cnt_c;
            4'd7: position = 7'd63 + cnt_c;
            4'd8: position = 7'd72 + cnt_c;
            default: position = 7'd0;
        endcase

        case (tap_r[3:0])
            4'd0: position_in = 7'd0  + tap_c[3:0];
            4'd1: position_in = 7'd9  + tap_c[3:0];
            4'd2: position_in = 7'd18 + tap_c[3:0];
            4'd3: position_in = 7'd27 + tap_c[3:0];
            4'd4: position_in = 7'd36 + tap_c[3:0];
            4'd5: position_in = 7'd45 + tap_c[3:0];
            4'd6: position_in = 7'd54 + tap_c[3:0];
            4'd7: position_in = 7'd63 + tap_c[3:0];
            4'd8: position_in = 7'd72 + tap_c[3:0];
            default: position_in = 7'd0;
        endcase

        fmap_in_addr  = {1'b0, cnt_ic[5:0], position_in};
        fmap_out_addr = {cnt_oc[6:0], position};
        bias_addr     = cnt_oc[6:0];
        skip_raddr    = {cnt_oc[6:0], position};
    end

    // 權重位址
    logic [15:0] w_idx_3x3, w_idx_1x1;
    always_comb begin
        w_idx_3x3   = 16'(cnt_oc) * 16'(in_ch) * 16'd9 + 16'(cnt_ic) * 16'd9 + 16'(cnt_kr) * 16'd3 + 16'(cnt_kc);
        w_idx_1x1   = 16'(cnt_oc) * 16'(in_ch) + 16'(cnt_ic);
        weight_addr = (kernel_size == 2'd3) ? w_idx_3x3 : w_idx_1x1;
    end

    // =========================================================
    // 安全數學運算
    // =========================================================
    assign mult_res = captured_weight * captured_fmap;

    logic signed [31:0] safe_skip_val;
    assign safe_skip_val = {24'd0, skip_rdata}; 
    
    always_comb begin
        shifted = acc >>> scale_shift;
        pre_relu_val = do_skip_add ? (shifted + safe_skip_val) : shifted;

        if (use_relu && pre_relu_val < 0)
            result_int8 = 8'sd0;
        else if (pre_relu_val > 32'sd127)
            result_int8 = 8'sd127;
        else if (pre_relu_val < -32'sd128)
            result_int8 = -8'sd128;
        else
            result_int8 = pre_relu_val[7:0];
    end

    // =========================================================
    // 主狀態機
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            cnt_oc <= '0; cnt_r <= '0; cnt_c <= '0;
            cnt_ic <= '0; cnt_kr <= '0; cnt_kc <= '0;
            acc             <= '0;
            fmap_out_we     <= 1'b0;
            fmap_out_data   <= '0;
            captured_fmap   <= 8'd0;
            captured_weight <= 8'd0;
        end else begin
            done        <= 1'b0;
            fmap_out_we <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        cnt_oc <= '0; cnt_r <= '0; cnt_c <= '0;
                        cnt_ic <= '0; cnt_kr <= '0; cnt_kc <= '0;
                        state  <= S_INIT_BIAS;
                    end
                end

                S_INIT_BIAS: state <= S_WAIT_BIAS;
                S_WAIT_BIAS: state <= S_INIT_ACC;

                S_INIT_ACC: begin
                    acc    <= $signed({{24{bias_data[7]}}, bias_data});
                    cnt_ic <= '0; cnt_kr <= '0; cnt_kc <= '0;
                    state  <= S_LOAD_ADDR;
                end

                S_LOAD_ADDR: begin
                    state <= S_WAIT_MEM;
                end

                S_WAIT_MEM: begin
                    state <= S_CAPTURE_DATA;
                end

                S_CAPTURE_DATA: begin
                    if (in_bounds) begin
                        captured_fmap   <= $signed(fmap_in_data);
                        captured_weight <= $signed(weight_data);
                    end else begin
                        captured_fmap   <= 8'd0;
                        captured_weight <= 8'd0;
                    end
                    state <= S_MAC;
                end

                S_MAC: begin
                    acc   <= acc + $signed({{16{mult_res[15]}}, mult_res});
                    state <= S_NEXT_TAP;                                         
                end

                S_NEXT_TAP: begin
                    if (kernel_size == 2'd3) begin
                        if (cnt_kc < 2'd2) begin
                            cnt_kc <= cnt_kc + 1'b1;
                            state  <= S_LOAD_ADDR;
                        end else begin
                            cnt_kc <= '0;
                            if (cnt_kr < 2'd2) begin
                                cnt_kr <= cnt_kr + 1'b1;
                                state  <= S_LOAD_ADDR;
                            end else begin
                                cnt_kr <= '0;
                                if (cnt_ic < in_ch - 1) begin
                                    cnt_ic <= cnt_ic + 1'b1;
                                    state  <= S_LOAD_ADDR;
                                end else begin
                                    state <= S_FINISH_POINT;
                                end
                            end
                        end
                    end else begin
                        if (cnt_ic < in_ch - 1) begin
                            cnt_ic <= cnt_ic + 1'b1;
                            state  <= S_LOAD_ADDR;
                        end else begin
                            state <= S_FINISH_POINT;
                        end
                    end
                end

                S_FINISH_POINT: begin
                    fmap_out_we   <= 1'b1;
                    fmap_out_data <= result_int8;  
                    state         <= S_NEXT_POSITION;
                end

                S_NEXT_POSITION: begin
                    cnt_ic <= '0; cnt_kr <= '0; cnt_kc <= '0;
                    if (cnt_c < 4'd8) begin
                        cnt_c <= cnt_c + 1'b1;
                        state <= S_INIT_BIAS;
                    end else begin
                        cnt_c <= '0;
                        if (cnt_r < 4'd8) begin
                            cnt_r <= cnt_r + 1'b1;
                            state <= S_INIT_BIAS;
                        end else begin
                            cnt_r <= '0;
                            if (cnt_oc < out_ch - 1) begin
                                cnt_oc <= cnt_oc + 1'b1;
                                state  <= S_INIT_BIAS;
                            end else begin
                                state <= S_DONE;
                            end
                        end
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