`timescale 1ns / 1ps
// ============================================================
// Module      : cnn_engine
// Description : Go AI 卷積神經網路總引擎 (8-Channel 算氣特徵版)
// ============================================================

module cnn_engine (
    // 系統時脈與重置
    input  logic         clk,
    input  logic         rst_n,
    
    // 控制與外部資料介面 (與 UART 介接)
    input  logic         start,             // 保留相容性，但現在改由 eval_board_valid 觸發
    input  logic [161:0] eval_board_out,    // 來自 packet_parser 的盤面
    input  logic         eval_board_valid,  // UART 接收完成觸發
    input  logic [7:0]   rx_last_move,      // 上一手 (0~80, 255=無)
    input  logic [7:0]   rx_prev_move,      // 上上一手
    
    // 舊版 read_data 介面 (已棄用，但保留 Port 避免上層報錯)
    output logic [7:0]   feat_read_addr,
    input  logic [7:0]   feat_read_data,
    
    // 最終預測輸出
    output logic [7:0]   policy_out [0:80],
    output logic [15:0]  value_out,
    output logic         done,

    // ==========================================
    // ? HIL Debug 介面
    // ==========================================
    output logic [7:0]   debug_layer_done_cnt,
    output logic [3:0]   debug_conv_state,
    output logic [3:0]   debug_layer_idx,
    output logic [2:0]   debug_lc_state,
    output logic [2:0]   debug_eng_state,
    
    output logic         debug_dump_req,
    input  logic         debug_dump_ack,
    output logic [6:0]   debug_ch_num,
    input  logic [13:0]  debug_read_addr,
    output logic [7:0]   debug_read_data
);

    // =========================================================
    // ? 1. 硬體算氣與 8-Channel 產生器
    // =========================================================
    logic        lib_builder_start;
    logic        lib_builder_done;
    logic [80:0] black_lib1, black_lib2;
    logic [80:0] white_lib1, white_lib2;

    board_liberty_builder u_lib_builder (
        .clk(clk),
        .rst_n(rst_n),
        .start(lib_builder_start),
        .board_state(eval_board_out),
        .black_lib1(black_lib1),
        .black_lib2(black_lib2),
        .white_lib1(white_lib1),
        .white_lib2(white_lib2),
        .done(lib_builder_done)
    );

    // ? 8-Channel 暫存區 (LUTRAM)
    logic [7:0] input_fmap [0:7][0:80];

    // =========================================================
    // 內部訊號宣告：主狀態機
    // =========================================================
    typedef enum logic [2:0] {
        E_IDLE        = 3'd0,
        E_WAIT_LIB    = 3'd1, // 等待算氣完成
        E_CNN_RUN     = 3'd2, // 神經網路運算
        E_READ_POLICY = 3'd3, // 遺跡狀態
        E_READ_VALUE  = 3'd4, // 讀出 Value
        E_DONE        = 3'd5  // 完成
    } engine_state_t;

    engine_state_t eng_state;
    assign debug_eng_state = eng_state;

    // =========================================================
    // 2. Layer Controller
    // =========================================================
    logic        lc_start, lc_done;
    logic        conv_start, conv_done;
    logic [6:0]  conv_in_ch, conv_out_ch;
    logic [1:0]  conv_kernel_size;
    logic        conv_use_relu;
    logic [4:0]  conv_scale_shift;
    
    logic        fc_start, fc_done;
    logic [7:0]  fc_in_features;
    logic [6:0]  fc_out_features;
    logic        fc_use_relu;
    logic [4:0]  fc_scale_shift;
    logic        fc_input_is_spatial;
    
    logic [3:0]  current_layer;
    logic        fmap_swap;
    logic        use_input_buffer;
    logic        do_skip_save, do_skip_add;
    
    logic [80*8-1:0] policy_flat_unused;
    logic [15:0]     value_lc_unused;
    
    layer_controller u_lc (
        .clk(clk), .rst_n(rst_n),
        .start(lc_start), .done(lc_done),
        .conv_start(conv_start), .conv_done(conv_done),
        .conv_in_ch(conv_in_ch), .conv_out_ch(conv_out_ch),
        .conv_kernel_size(conv_kernel_size),
        .conv_use_relu(conv_use_relu),
        .conv_scale_shift(conv_scale_shift),
        .fc_start(fc_start), .fc_done(fc_done),
        .fc_in_features(fc_in_features),
        .fc_out_features(fc_out_features),
        .fc_use_relu(fc_use_relu),
        .fc_scale_shift(fc_scale_shift),
        .fc_input_is_spatial(fc_input_is_spatial),
        .current_layer(current_layer),
        .fmap_swap(fmap_swap),
        .use_input_buffer(use_input_buffer),
        .do_skip_save(do_skip_save),
        .do_skip_add(do_skip_add),
        
        .policy_flat(policy_flat_unused),
        .value_out(value_lc_unused),
        .debug_layer_idx(debug_layer_idx),
        .debug_lc_state(debug_lc_state),
        .debug_dump_req(debug_dump_req),
        .debug_dump_ack(debug_dump_ack),
        .debug_ch_num(debug_ch_num)
    );

    // =========================================================
    // 3. 特徵圖緩衝區
    // =========================================================
    logic        fa_we, fb_we;
    logic [13:0] fa_waddr, fb_waddr;
    logic [7:0]  fa_wdata, fb_wdata;
    logic [13:0] fa_raddr, fb_raddr;
    logic [7:0]  fa_rdata, fb_rdata;
    logic [13:0] conv_fmap_in_addr;
    logic [7:0]  conv_fmap_in_data;
    logic        conv_fmap_out_we;
    logic [13:0] conv_fmap_out_addr;
    logic [7:0]  conv_fmap_out_data;
    logic [15:0] conv_weight_addr;
    logic [7:0]  conv_weight_data;
    logic [6:0]  conv_bias_addr;
    logic [7:0]  conv_bias_data;
    
    logic [13:0] skip_rdata;
    logic [13:0] conv_skip_raddr;  
    logic [13:0] actual_skip_raddr; 

    fmap_buffer_a fmap_a (.clka(clk), .clkb(clk), .wea(fa_we), .addra(fa_waddr), .dina(fa_wdata), .addrb(fa_raddr), .doutb(fa_rdata));
    fmap_buffer_b fmap_b (.clka(clk), .clkb(clk), .wea(fb_we), .addra(fb_waddr), .dina(fb_wdata), .addrb(fb_raddr), .doutb(fb_rdata));

    assign actual_skip_raddr = (debug_dump_req && debug_layer_idx == 4'd1) ? debug_read_addr : conv_skip_raddr;

    // =========================================================
    // 4. Conv Unit 與 FC Unit 實例化
    // =========================================================
    conv_unit u_conv (
        .clk(clk), .rst_n(rst_n),
        .start(conv_start), .done(conv_done),
        .in_ch(conv_in_ch), .out_ch(conv_out_ch),
        .kernel_size(conv_kernel_size),
        .use_relu(conv_use_relu),
        .scale_shift(conv_scale_shift),
        
        .fmap_in_addr(conv_fmap_in_addr),
        .fmap_in_data(conv_fmap_in_data),
        .fmap_out_we(conv_fmap_out_we),
        .fmap_out_addr(conv_fmap_out_addr),
        .fmap_out_data(conv_fmap_out_data),
        
        .weight_addr(conv_weight_addr),
        .weight_data(conv_weight_data),
        .bias_addr(conv_bias_addr),
        .bias_data(conv_bias_data),
        
        .do_skip_add(do_skip_add),
        .skip_raddr(conv_skip_raddr), 
        .skip_rdata(skip_rdata),
        .debug_state(debug_conv_state)
    );

    logic [13:0] fc_fmap_in_addr;
    logic [7:0]  fc_fmap_in_data;
    logic        fc_fmap_out_we;
    logic [13:0] fc_fmap_out_addr;
    logic [7:0]  fc_fmap_out_data;
    logic [15:0] fc_weight_addr;
    logic [7:0]  fc_weight_data;
    logic [6:0]  fc_bias_addr;
    logic [7:0]  fc_bias_data;

    fc_unit u_fc (
        .clk(clk), .rst_n(rst_n),
        .start(fc_start), .done(fc_done),
        .in_features(fc_in_features),
        .out_features(fc_out_features),
        .use_relu(fc_use_relu),
        .scale_shift(fc_scale_shift),
        .input_is_spatial(fc_input_is_spatial),
        
        .fmap_in_addr(fc_fmap_in_addr),
        .fmap_in_data(fc_fmap_in_data),
        .fmap_out_we(fc_fmap_out_we),
        .fmap_out_addr(fc_fmap_out_addr),
        .fmap_out_data(fc_fmap_out_data),
        
        .weight_addr(fc_weight_addr),
        .weight_data(fc_weight_data),
        .bias_addr(fc_bias_addr),
        .bias_data(fc_bias_data),
        .debug_state()
    );

    // =========================================================
    // 5. Memory 多工器與讀取選擇
    // =========================================================
    logic fmap_role, fmap_role_snapshot;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)         fmap_role <= 1'b0;
        else if (lc_start)  fmap_role <= 1'b0;
        else if (fmap_swap) fmap_role <= ~fmap_role;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)       fmap_role_snapshot <= 1'b0;
        else if (lc_done) fmap_role_snapshot <= fmap_role;
    end

    logic [13:0] out_read_addr;
    logic        reading_out;
    logic        is_fc_running;
    
    assign reading_out   = (eng_state == E_READ_POLICY) || (eng_state == E_READ_VALUE);
    assign is_fc_running = (current_layer == 4'd6) || (current_layer == 4'd8) || (current_layer == 4'd9);

    always_comb begin
        if (reading_out) begin
            fa_raddr = out_read_addr; fb_raddr = out_read_addr;
        end else if (debug_dump_req) begin
            fa_raddr = debug_read_addr; fb_raddr = debug_read_addr;
        end else if (is_fc_running) begin
            fa_raddr = fc_fmap_in_addr; fb_raddr = fc_fmap_in_addr;
        end else begin
            if (fmap_role == 1'b0) begin
                fa_raddr = conv_fmap_in_addr; fb_raddr = do_skip_add ? conv_skip_raddr : 14'd0;
            end else begin
                fb_raddr = conv_fmap_in_addr; fa_raddr = do_skip_add ? conv_skip_raddr : 14'd0;
            end
        end
    end

    always_comb begin
        if (debug_layer_idx == 4'd6) begin
            debug_read_data = policy_out[debug_read_addr[13:7]]; 
        end else if (fmap_role == 1'b1) begin
            debug_read_data = fa_rdata;
        end else begin
            debug_read_data = fb_rdata;
        end
    end

    assign skip_rdata = (fmap_role == 1'b1) ? fa_rdata : fb_rdata;
    assign feat_read_addr = conv_fmap_in_addr[6:0]; // 遺跡

    // ? 全新 8-Channel 讀取路徑
    always_comb begin
        if (use_input_buffer) begin
            conv_fmap_in_data = input_fmap[conv_fmap_in_addr[13:7]][conv_fmap_in_addr[6:0]];
        end 
        else if (fmap_role == 1'b0) begin
            conv_fmap_in_data = fa_rdata;
        end 
        else begin
            conv_fmap_in_data = fb_rdata;
        end
    end    

    always_comb begin
        if (fmap_role == 1'b0) fc_fmap_in_data = fa_rdata;
        else                   fc_fmap_in_data = fb_rdata;
    end

    always_comb begin
        fa_we = 1'b0; fa_waddr = 14'd0; fa_wdata = 8'd0;
        fb_we = 1'b0; fb_waddr = 14'd0; fb_wdata = 8'd0;
        
        if (current_layer != 4'd6) begin
            if (fmap_role == 1'b1) begin
                fa_we    = conv_fmap_out_we | fc_fmap_out_we;
                fa_waddr = fc_fmap_out_we ? fc_fmap_out_addr : conv_fmap_out_addr;
                fa_wdata = fc_fmap_out_we ? fc_fmap_out_data : conv_fmap_out_data;
            end else begin
                fb_we    = conv_fmap_out_we | fc_fmap_out_we;
                fb_waddr = fc_fmap_out_we ? fc_fmap_out_addr : conv_fmap_out_addr;
                fb_wdata = fc_fmap_out_we ? fc_fmap_out_data : conv_fmap_out_data;
            end
        end
    end

    // =========================================================
    // 6. Weight/Bias ROM
    // =========================================================
    logic [7:0] w_entry, w_t0c1, w_t0c2, w_t1c1, w_t1c2, w_pc, w_pfc, w_vc, w_vfc1, w_vfc2;
    logic [7:0] b_entry, b_t0c1, b_t0c2, b_t1c1, b_t1c2, b_pfc, b_vfc1;

    rom_entry_weight        u_w_entry (.clka(clk),  .addra(conv_weight_addr[12:0]),  .douta(w_entry)); // ? Depth 變 4608, addr 需要 13 bits
    rom_entry_bias          u_b_entry (.clk(clk),   .addr(conv_bias_addr[5:0]),      .data(b_entry));
    rom_tower0_conv1_weight u_w_t0c1  (.clka(clk),  .addra(conv_weight_addr),        .douta(w_t0c1));
    rom_tower0_conv1_bias   u_b_t0c1  (.clk(clk),   .addr(conv_bias_addr[5:0]),      .data(b_t0c1));
    rom_tower0_conv2_weight u_w_t0c2  (.clka(clk),  .addra(conv_weight_addr),        .douta(w_t0c2));
    rom_tower0_conv2_bias   u_b_t0c2  (.clk(clk),   .addr(conv_bias_addr[5:0]),      .data(b_t0c2));
    rom_tower1_conv1_weight u_w_t1c1  (.clka(clk),  .addra(conv_weight_addr),        .douta(w_t1c1));
    rom_tower1_conv1_bias   u_b_t1c1  (.clk(clk),   .addr(conv_bias_addr[5:0]),      .data(b_t1c1));
    rom_tower1_conv2_weight u_w_t1c2  (.clka(clk),  .addra(conv_weight_addr),        .douta(w_t1c2));
    rom_tower1_conv2_bias   u_b_t1c2  (.clk(clk),   .addr(conv_bias_addr[5:0]),      .data(b_t1c2));
    rom_policy_conv_weight  u_w_pc    (.clk(clk),   .addr(conv_weight_addr[6:0]),    .data(w_pc));
    rom_policy_fc_weight    u_w_pfc   (.clka(clk),  .addra(fc_weight_addr[13:0]),    .douta(w_pfc));
    rom_policy_fc_bias      u_b_pfc   (.clk(clk),   .addr(fc_bias_addr[6:0]),        .data(b_pfc));
    rom_value_conv_weight   u_w_vc    (.clk(clk),   .addr(conv_weight_addr[5:0]),    .data(w_vc));
    rom_value_fc1_weight    u_w_vfc1  (.clka(clk),  .addra(fc_weight_addr[12:0]),    .douta(w_vfc1));
    rom_value_fc1_bias      u_b_vfc1  (.clk(clk),   .addr(fc_bias_addr[5:0]),        .data(b_vfc1));
    rom_value_fc2_weight    u_w_vfc2  (.clk(clk),   .addr(fc_weight_addr[5:0]),      .data(w_vfc2));

    always_comb begin
        case (current_layer)
            4'd0: conv_weight_data = w_entry;
            4'd1: conv_weight_data = w_t0c1;
            4'd2: conv_weight_data = w_t0c2;
            4'd3: conv_weight_data = w_t1c1; 
            4'd4: conv_weight_data = w_t1c2; 
            4'd5: conv_weight_data = w_pc;
            4'd7: conv_weight_data = w_vc;
            default: conv_weight_data = 8'd0;
        endcase
    end

    always_comb begin
        case (current_layer)
            4'd0: conv_bias_data = b_entry;
            4'd1: conv_bias_data = b_t0c1;
            4'd2: conv_bias_data = b_t0c2;
            4'd3: conv_bias_data = b_t1c1; 
            4'd4: conv_bias_data = b_t1c2; 
            default: conv_bias_data = 8'd0;
        endcase
    end

    always_comb begin
        case (current_layer)
            4'd6: fc_weight_data = w_pfc;
            4'd8: fc_weight_data = w_vfc1;
            4'd9: fc_weight_data = w_vfc2;
            default: fc_weight_data = 8'd0;
        endcase
    end

    always_comb begin
        case (current_layer)
            4'd6: fc_bias_data = b_pfc;
            4'd8: fc_bias_data = b_vfc1;
            default: fc_bias_data = 8'd0;
        endcase
    end

    // =========================================================
    // 7. Debug：Layer 完成計數
    // =========================================================
    logic [7:0] layer_cnt;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                    layer_cnt <= 8'd0;
        else if (lc_start)             layer_cnt <= 8'd0;
        else if (conv_done | fc_done)  layer_cnt <= layer_cnt + 1'b1;
    end
    assign debug_layer_done_cnt = layer_cnt;

    // =========================================================
    // 8. 總調度狀態機 (包含算氣攔截)
    // =========================================================
    logic [6:0]  read_cnt, read_cnt_r1, read_cnt_r2;
    logic        read_valid_r1, read_valid_r2;
    logic        reading_policy_sig, reading_value_sig;

    always_comb begin
        if (eng_state == E_READ_VALUE) out_read_addr = 14'd0;
        else out_read_addr = {read_cnt[6:0], 7'd0};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            eng_state          <= E_IDLE;
            lib_builder_start  <= 1'b0;
            lc_start           <= 1'b0;
            done               <= 1'b0;
            read_cnt           <= 7'd0;
            read_cnt_r1        <= 7'd0;
            read_cnt_r2        <= 7'd0;
            read_valid_r1      <= 1'b0;
            read_valid_r2      <= 1'b0;
            reading_policy_sig <= 1'b0;
            reading_value_sig  <= 1'b0;
            value_out          <= 16'd0;
            for (int i = 0; i < 81; i++) policy_out[i] <= 8'd0;
        end else begin
            lib_builder_start <= 1'b0;
            lc_start <= 1'b0;
            done     <= 1'b0;

            if (current_layer == 4'd6 && fc_fmap_out_we) begin
                policy_out[fc_fmap_out_addr[13:7]] <= fc_fmap_out_data;
            end

            case (eng_state)
                E_IDLE: begin
                    // ? 攔截：如果收到了有效的盤面，先啟動算氣
                    if (start || eval_board_valid) begin
                        lib_builder_start <= 1'b1;
                        eng_state         <= E_WAIT_LIB;
                    end
                end

                E_WAIT_LIB: begin
                    // 等待算氣模組完成
                    if (lib_builder_done) begin
                        // 瞬間將算好的 8 個 Channel 炸進 input_fmap LUTRAM 裡
                        for (int i = 0; i < 81; i++) begin
                            // Ch0, Ch1: 盤面
                            input_fmap[0][i] <= (eval_board_out[i*2+:2] == 2'b01) ? 8'd127 : 8'd0;
                            input_fmap[1][i] <= (eval_board_out[i*2+:2] == 2'b10) ? 8'd127 : 8'd0;

                            // Ch2, Ch3: 黑棋氣
                            input_fmap[2][i] <= black_lib1[i] ? 8'd127 : 8'd0;
                            input_fmap[3][i] <= black_lib2[i] ? 8'd127 : 8'd0;

                            // Ch4, Ch5: 白棋氣
                            input_fmap[4][i] <= white_lib1[i] ? 8'd127 : 8'd0;
                            input_fmap[5][i] <= white_lib2[i] ? 8'd127 : 8'd0;

                            // Ch6, Ch7: 歷史落子
                            input_fmap[6][i] <= (rx_last_move == i) ? 8'd127 : 8'd0;
                            input_fmap[7][i] <= (rx_prev_move == i) ? 8'd127 : 8'd0;
                        end
                        // 觸發 CNN 引擎
                        lc_start  <= 1'b1;
                        eng_state <= E_CNN_RUN;
                    end
                end

                E_CNN_RUN: begin
                    if (lc_done) begin
                        read_cnt           <= 7'd0;
                        read_valid_r1      <= 1'b0;
                        read_valid_r2      <= 1'b0;
                        reading_value_sig  <= 1'b1;
                        eng_state          <= E_READ_VALUE;
                    end
                end

                E_READ_VALUE: begin
                    read_valid_r1 <= reading_value_sig;
                    read_valid_r2 <= read_valid_r1;

                    if (reading_value_sig)
                        reading_value_sig <= 1'b0;

                    if (read_valid_r2) begin
                        if (fmap_role_snapshot == 1'b0) begin
                            value_out <= fa_rdata[7] ? {8'hFF, fa_rdata} : {8'h00, fa_rdata};
                        end else begin
                            value_out <= fb_rdata[7] ? {8'hFF, fb_rdata} : {8'h00, fb_rdata};
                        end
                    end

                    if (!reading_value_sig && !read_valid_r1 && !read_valid_r2)
                        eng_state <= E_DONE;
                end

                E_DONE: begin
                    done      <= 1'b1;
                    eng_state <= E_IDLE;
                end

                default: eng_state <= E_IDLE;
            endcase
        end
    end
endmodule