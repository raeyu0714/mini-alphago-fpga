`timescale 1ns / 1ps

// Hardware-in-the-Loop (HIL) engine for the AI player.
//
// FSM loop for one AI turn:
//   S_IDLE            - wait for start_ai from game_fsm
//   S_SEND_REQUEST    - send 0xAA board state to PC (triggers MCTS)
//   S_WAIT_PC_PACKET  - wait for either:
//                         0xCC eval request  -> run one CNN inference
//                         0xDD final move    -> latch best_move and finish
//   S_LOAD_FEATURE    - one-cycle delay to let CNN engine latch the board
//   S_CNN_INFERENCE   - wait for cnn_engine to finish; then send 0xBB result
//   S_SEND_RESULT     - wait for packet_builder to finish sending 0xBB
//   S_DONE            - assert ai_done, wait for game_fsm to de-assert start_ai
//
// Debug states (CNN mid-inference BRAM dump):
//   S_CNN_DEBUG_TX     - stream debug data over UART
//   S_CNN_DEBUG_RESUME - wait for debug_dump_req to clear before resuming
module go_ai_core #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
)(
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start_ai,
    input  logic [161:0] board_state,
    input  logic         uart_rx,
    output logic         uart_tx,
    output logic         ai_done,
    output logic [3:0]   best_move_x,
    output logic [3:0]   best_move_y,

    // Debug outputs
    output logic [3:0]   debug_state,
    output logic         debug_eval_valid,
    output logic         debug_final_valid,
    output logic         debug_send_done,
    output logic         debug_pkt_error,
    output logic [7:0]   debug_layer_cnt,
    output logic [3:0]   debug_layer_idx,
    output logic [3:0]   debug_conv_state,
    output logic [2:0]   debug_lc_state,
    output logic [2:0]   debug_eng_state
);
    typedef enum logic [3:0] {
        S_IDLE             = 4'd0,
        S_SEND_REQUEST     = 4'd1,
        S_WAIT_PC_PACKET   = 4'd2,
        S_LOAD_FEATURE     = 4'd3,
        S_CNN_INFERENCE    = 4'd4,
        S_SEND_RESULT      = 4'd5,
        S_DONE             = 4'd6,
        S_CNN_DEBUG_TX     = 4'd7,
        S_CNN_DEBUG_RESUME = 4'd8
    } state_t;

    state_t state;
    assign debug_state = state;

    // =========================================================
    // UART RX / TX
    // =========================================================
    logic [7:0] rx_data;
    logic       rx_valid;

    uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE)) u_uart_rx (
        .clk  (clk),   .rst_n(rst_n),
        .rx   (uart_rx),
        .data (rx_data), .valid(rx_valid)
    );

    logic [7:0] tx_data;
    logic       tx_start;
    logic       tx_busy;

    uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE)) u_uart_tx (
        .clk  (clk),   .rst_n(rst_n),
        .data (tx_data), .start(tx_start),
        .tx   (uart_tx), .busy (tx_busy)
    );

    // =========================================================
    // Packet Parser
    // =========================================================
    logic [161:0] eval_board_out;
    logic         eval_board_valid;
    logic [3:0]   final_x_from_pc;
    logic [3:0]   final_y_from_pc;
    logic         final_move_valid;
    logic         packet_error;
    logic [7:0]   rx_last_move;
    logic [7:0]   rx_prev_move;

    packet_parser u_parser (
        .clk             (clk),   .rst_n(rst_n),
        .rx_data         (rx_data), .rx_valid(rx_valid),
        .eval_board_out  (eval_board_out),
        .eval_board_valid(eval_board_valid),
        .rx_last_move    (rx_last_move),
        .rx_prev_move    (rx_prev_move),
        .final_x         (final_x_from_pc),
        .final_y         (final_y_from_pc),
        .final_move_valid(final_move_valid),
        .packet_error    (packet_error)
    );

    // Sticky flags for debug visibility
    logic eval_seen, final_seen, error_seen;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            eval_seen  <= 1'b0;
            final_seen <= 1'b0;
            error_seen <= 1'b0;
        end else begin
            if (state == S_IDLE && start_ai) begin
                eval_seen  <= 1'b0;
                final_seen <= 1'b0;
                error_seen <= 1'b0;
            end
            if (eval_board_valid) eval_seen  <= 1'b1;
            if (final_move_valid) final_seen <= 1'b1;
            if (packet_error)     error_seen <= 1'b1;
        end
    end
    assign debug_eval_valid = eval_seen;
    assign debug_final_valid = final_seen;
    assign debug_pkt_error   = error_seen;

    // =========================================================
    // Debug BRAM dump channel (CNN engine <-> packet_builder)
    // =========================================================
    logic        debug_dump_req;
    logic        debug_dump_ack;
    logic [6:0]  debug_ch_num;
    logic [13:0] debug_read_addr;
    logic [7:0]  debug_read_data;
    logic        send_debug_start;

    // =========================================================
    // Packet Builder
    // =========================================================
    logic         send_request_start;
    logic         send_result_start;
    logic         send_done;
    logic         builder_busy;
    logic [161:0] board_to_send;
    logic [7:0]   policy_to_send [0:80];
    logic [15:0]  value_to_send;

    packet_builder u_builder (
        .clk               (clk),   .rst_n(rst_n),
        .send_request_start(send_request_start),
        .send_result_start (send_result_start),
        .board_state_in    (board_to_send),
        .policy_in         (policy_to_send),
        .value_in          (value_to_send),
        .send_debug_start  (send_debug_start),
        .debug_layer_idx   (debug_layer_idx),
        .debug_ch_num      (debug_ch_num),
        .debug_read_addr   (debug_read_addr),
        .debug_read_data   (debug_read_data),
        .tx_data           (tx_data),   .tx_start(tx_start),
        .tx_busy           (tx_busy),
        .send_done         (send_done), .busy(builder_busy)
    );

    logic send_done_seen;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) send_done_seen <= 1'b0;
        else begin
            if (state == S_IDLE && start_ai) send_done_seen <= 1'b0;
            if (send_done)                   send_done_seen <= 1'b1;
        end
    end
    assign debug_send_done = send_done_seen;

    // =========================================================
    // Feature buffer (pass-through; CNN engine builds 8-ch internally)
    // =========================================================
    logic [7:0] feat_read_addr;
    logic [7:0] feat_read_data;
    logic [161:0] feat_board_in;

    always_comb begin
        if (feat_read_addr < 8'd81)
            feat_read_data = {6'd0, feat_board_in[feat_read_addr * 2 +: 2]};
        else
            feat_read_data = 8'd0;
    end

    // =========================================================
    // CNN Engine
    // =========================================================
    logic        cnn_start;
    logic        cnn_done;
    logic [7:0]  cnn_policy [0:80];
    logic [15:0] cnn_value;

    cnn_engine u_cnn (
        .clk              (clk),   .rst_n(rst_n),
        .start            (cnn_start),
        .eval_board_out   (eval_board_out),
        .eval_board_valid (eval_board_valid),
        .rx_last_move     (rx_last_move),
        .rx_prev_move     (rx_prev_move),
        .feat_read_addr   (feat_read_addr),
        .feat_read_data   (feat_read_data),
        .policy_out       (cnn_policy),
        .value_out        (cnn_value),
        .done             (cnn_done),
        .debug_layer_done_cnt(debug_layer_cnt),
        .debug_conv_state (debug_conv_state),
        .debug_layer_idx  (debug_layer_idx),
        .debug_lc_state   (debug_lc_state),
        .debug_eng_state  (debug_eng_state),
        .debug_dump_req   (debug_dump_req),
        .debug_dump_ack   (debug_dump_ack),
        .debug_ch_num     (debug_ch_num),
        .debug_read_addr  (debug_read_addr),
        .debug_read_data  (debug_read_data)
    );

    // Wire CNN policy output to packet builder
    genvar gi;
    generate
        for (gi = 0; gi < 81; gi = gi + 1) begin : POLICY_CONNECT
            assign policy_to_send[gi] = cnn_policy[gi];
        end
    endgenerate

    // =========================================================
    // Main FSM
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            ai_done            <= 1'b0;
            best_move_x        <= 4'd0;
            best_move_y        <= 4'd0;
            send_request_start <= 1'b0;
            send_result_start  <= 1'b0;
            send_debug_start   <= 1'b0;
            debug_dump_ack     <= 1'b0;
            cnn_start          <= 1'b0;
            feat_board_in      <= 162'd0;
            board_to_send      <= 162'd0;
            value_to_send      <= 16'd0;
        end else begin
            send_request_start <= 1'b0;
            send_result_start  <= 1'b0;
            send_debug_start   <= 1'b0;
            cnn_start          <= 1'b0;

            case (state)
                S_IDLE: begin
                    ai_done <= 1'b0;
                    if (start_ai) begin
                        board_to_send      <= board_state;
                        send_request_start <= 1'b1;
                        state              <= S_SEND_REQUEST;
                    end
                end

                S_SEND_REQUEST: begin
                    if (send_done) state <= S_WAIT_PC_PACKET;
                end

                S_WAIT_PC_PACKET: begin
                    if (eval_board_valid) begin
                        feat_board_in <= eval_board_out;
                        state         <= S_LOAD_FEATURE;
                    end else if (final_move_valid) begin
                        best_move_x <= final_x_from_pc;
                        best_move_y <= final_y_from_pc;
                        ai_done     <= 1'b1;
                        state       <= S_DONE;
                    end
                end

                S_LOAD_FEATURE: begin
                    cnn_start <= 1'b1;
                    state     <= S_CNN_INFERENCE;
                end

                S_CNN_INFERENCE: begin
                    if (cnn_done) begin
                        value_to_send     <= cnn_value;
                        send_result_start <= 1'b1;
                        state             <= S_SEND_RESULT;
                    end else if (debug_dump_req && !debug_dump_ack) begin
                        send_debug_start <= 1'b1;
                        state            <= S_CNN_DEBUG_TX;
                    end
                end

                S_CNN_DEBUG_TX: begin
                    if (send_done) begin
                        debug_dump_ack <= 1'b1;
                        state          <= S_CNN_DEBUG_RESUME;
                    end
                end

                S_CNN_DEBUG_RESUME: begin
                    debug_dump_ack <= 1'b0;
                    if (!debug_dump_req) state <= S_CNN_INFERENCE;
                end

                S_SEND_RESULT: begin
                    if (send_done) state <= S_WAIT_PC_PACKET;
                end

                S_DONE: begin
                    ai_done <= 1'b1;
                    if (!start_ai) begin
                        ai_done <= 1'b0;
                        state   <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
