`timescale 1ns / 1ps
// ============================================================
// Module      : layer_controller
// Description : 神經網路層級調度中樞 (支援 ResNet Skip 組合邏輯優化)
// ============================================================

module layer_controller (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    output logic         done,

    output logic         conv_start,
    input  logic         conv_done,
    output logic [6:0]   conv_in_ch,
    output logic [6:0]   conv_out_ch,
    output logic [1:0]   conv_kernel_size,
    output logic         conv_use_relu,
    output logic [4:0]   conv_scale_shift,

    output logic         fc_start,
    input  logic         fc_done,
    output logic [7:0]   fc_in_features,
    output logic [6:0]   fc_out_features,
    output logic         fc_use_relu,
    output logic [4:0]   fc_scale_shift,
    output logic         fc_input_is_spatial,

    output logic [3:0]   current_layer,
    output logic         fmap_swap,
    output logic         use_input_buffer,
    output logic         do_skip_save,
    output logic         do_skip_add,
    output logic [80*8-1:0] policy_flat,
    output logic [15:0]  value_out,

    output logic [3:0]   debug_layer_idx,
    output logic [2:0]   debug_lc_state,
    output logic         debug_dump_req, 
    input  logic         debug_dump_ack, 
    output logic [6:0]   debug_ch_num    
);

    localparam L_ENTRY                = 4'd0;
    localparam L_T0_CONV1             = 4'd1;
    localparam L_T0_CONV2             = 4'd2;
    localparam L_T1_CONV1             = 4'd3;
    localparam L_T1_CONV2             = 4'd4;
    localparam L_POLICY_C             = 4'd5;
    localparam L_POLICY_FC            = 4'd6;
    localparam L_VALUE_C              = 4'd7;
    localparam L_VALUE_FC1            = 4'd8;
    localparam L_VALUE_FC2            = 4'd9;
    // ★ layer_idx=10：PolicyFC 輸入 dump（PC端用這個 idx 識別）
    localparam L_POLICYFC_INPUT_DUMP  = 4'd10;

    // ★ enum 從 3bit 改成 4bit（多一個狀態）
    typedef enum logic [3:0] {
        S_IDLE          = 4'd0,
        S_PREP_LAYER    = 4'd1,
        S_RUN_LAYER     = 4'd2,
        S_WAIT_LAYER    = 4'd3,
        S_POST_LAYER    = 4'd4,
        S_NEXT_LAYER    = 4'd5,
        S_DONE          = 4'd6,
        S_DEBUG_DUMP    = 4'd7,
        S_FC_INPUT_DUMP = 4'd8   // ★ 新增
    } state_t;

    state_t state;
    logic [3:0] layer_idx;

    assign debug_layer_idx = layer_idx;
    assign debug_lc_state  = state[2:0];
    assign current_layer   = layer_idx;

    logic is_fc_layer;
    assign is_fc_layer = (layer_idx == L_POLICY_FC) || 
                         (layer_idx == L_VALUE_FC1) || 
                         (layer_idx == L_VALUE_FC2);

    assign do_skip_save = (layer_idx == L_ENTRY) || (layer_idx == L_T0_CONV2);;
    assign do_skip_add  = (layer_idx == L_T0_CONV2) || (layer_idx == L_T1_CONV2);

    // =========================================================
    // Conv 參數表
    // =========================================================
    logic [6:0] p_in_ch, p_out_ch;
    logic [1:0] p_kernel;
    logic       p_relu;
    logic [4:0] p_shift;

    // ★ debug_ch_num：FC 輸入 dump 時固定送 2（2ch × 81 bytes）
    assign debug_ch_num = (layer_idx == L_POLICYFC_INPUT_DUMP) ? 7'd2 : 
                          (is_fc_layer) ? fc_out_features : p_out_ch;

    assign conv_in_ch       = p_in_ch;
    assign conv_out_ch      = p_out_ch;
    assign conv_kernel_size = p_kernel;
    assign conv_use_relu    = p_relu;
    assign conv_scale_shift = p_shift;

    always_comb begin
        p_in_ch  = 7'd64; p_out_ch = 7'd64;
        p_kernel = 2'd3;  p_relu   = 1'b1; p_shift = 5'd5;

        case (layer_idx)
            L_ENTRY:    begin p_in_ch = 7'd8;  p_out_ch = 7'd64; p_kernel = 2'd3; p_relu = 1'b1; p_shift = 5'd6; end
            L_T0_CONV1: begin p_in_ch = 7'd64; p_out_ch = 7'd64; p_kernel = 2'd3; p_relu = 1'b1; p_shift = 5'd7; end
            L_T0_CONV2: begin p_in_ch = 7'd64; p_out_ch = 7'd64; p_kernel = 2'd3; p_relu = 1'b1; p_shift = 5'd7; end
            L_T1_CONV1: begin p_in_ch = 7'd64; p_out_ch = 7'd64; p_kernel = 2'd3; p_relu = 1'b1; p_shift = 5'd7; end
            L_T1_CONV2: begin p_in_ch = 7'd64; p_out_ch = 7'd64; p_kernel = 2'd3; p_relu = 1'b1; p_shift = 5'd7; end
            L_POLICY_C: begin p_in_ch = 7'd64; p_out_ch = 7'd2;  p_kernel = 2'd1; p_relu = 1'b1; p_shift = 5'd7; end
            L_VALUE_C:  begin p_in_ch = 7'd64; p_out_ch = 7'd1;  p_kernel = 2'd1; p_relu = 1'b1; p_shift = 5'd7; end
            default: ;
        endcase
    end

    // =========================================================
    // FC 參數表
    // =========================================================
    logic [7:0] fc_in;
    logic [6:0] fc_out;
    logic       fc_relu;
    logic [4:0] fc_shift;
    logic       fc_spatial;

    assign fc_in_features      = fc_in;
    assign fc_out_features     = fc_out;
    assign fc_use_relu         = fc_relu;
    assign fc_scale_shift      = fc_shift;
    assign fc_input_is_spatial = fc_spatial;

    always_comb begin
        fc_in = 8'd162; fc_out = 7'd81;
        fc_relu = 1'b0; fc_shift = 5'd5; fc_spatial = 1'b1;

        case (layer_idx)
            L_POLICY_FC: begin fc_in = 8'd162; fc_out = 7'd81; fc_relu = 1'b0; fc_shift = 5'd6; fc_spatial = 1'b1; end
            L_VALUE_FC1: begin fc_in = 8'd81;  fc_out = 7'd64; fc_relu = 1'b1; fc_shift = 5'd7; fc_spatial = 1'b1; end
            L_VALUE_FC2: begin fc_in = 8'd64;  fc_out = 7'd1;  fc_relu = 1'b0; fc_shift = 5'd8; fc_spatial = 1'b0; end
            default: ;
        endcase
    end

    assign policy_flat = '0;
    assign value_out   = '0;

    // =========================================================
    // 主狀態機
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            layer_idx        <= 4'd0;
            done             <= 1'b0;
            conv_start       <= 1'b0;
            fc_start         <= 1'b0;
            fmap_swap        <= 1'b0;
            use_input_buffer <= 1'b0;
            debug_dump_req   <= 1'b0;
        end else begin
            done           <= 1'b0;
            conv_start     <= 1'b0;
            fc_start       <= 1'b0;
            fmap_swap      <= 1'b0;
            debug_dump_req <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        layer_idx        <= L_ENTRY;
                        use_input_buffer <= 1'b1;
                        state            <= S_PREP_LAYER;
                    end
                end

                S_PREP_LAYER: state <= S_RUN_LAYER;

                S_RUN_LAYER: begin
                    if (is_fc_layer) fc_start   <= 1'b1;
                    else             conv_start <= 1'b1;
                    state <= S_WAIT_LAYER;
                end

                S_WAIT_LAYER: begin
                    if (is_fc_layer) begin
                        if (fc_done)   state <= S_POST_LAYER;
                    end else begin
                        if (conv_done) state <= S_POST_LAYER;
                    end
                end

                //S_POST_LAYER: state <= S_DEBUG_DUMP;
                S_POST_LAYER: state <= S_NEXT_LAYER;
                S_DEBUG_DUMP: begin
                    debug_dump_req <= 1'b1;
                    if (debug_dump_ack) begin
                        debug_dump_req <= 1'b0;
                        state <= S_NEXT_LAYER;
                    end
                end

                S_NEXT_LAYER: begin
                    use_input_buffer <= 1'b0;
                    fmap_swap        <= 1'b1;

                    if (layer_idx == L_VALUE_FC2) begin
                        state <= S_DONE;
                    end else if (layer_idx == L_T1_CONV2) begin
                        layer_idx <= L_POLICY_C;
                        state     <= S_PREP_LAYER;
                    end else begin
                        layer_idx <= layer_idx + 1'b1;
                        state     <= S_PREP_LAYER;
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