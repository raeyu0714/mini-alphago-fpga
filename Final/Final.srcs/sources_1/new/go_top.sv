`timescale 1ns / 1ps

// Top-level module for the Mini AlphaGo Zero FPGA system.
//
// Instantiates and wires:
//   - Clock divider  (100 MHz -> 25 MHz for VGA)
//   - Button debouncer for each of the five buttons
//   - Cursor logic
//   - Game FSM
//   - Rule engine + board manager
//   - go_ai_core (UART HIL + CNN)
//   - Territory counter
//   - VGA controller
//
// LED debug mapping:
//   [2:0]  - game FSM state (S_MENU=0 .. S_GAME_OVER=7)
//   [6:3]  - go_ai_core main FSM state
//   [7]    - ai_done
//   [11:8] - current CNN layer index
//   [14:12]- conv_unit FSM state
//   [15]   - at least one CNN layer has completed since last reset
module go_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        btnC,
    input  logic        btnU,
    input  logic        btnD,
    input  logic        btnL,
    input  logic        btnR,
    input  logic        end_game_sw,
    input  logic        RsRx,
    output logic        RsTx,
    output logic [3:0]  vgaRed,
    output logic [3:0]  vgaGreen,
    output logic [3:0]  vgaBlue,
    output logic        Hsync,
    output logic        Vsync,
    output logic [15:0] led
);
    // =========================================================
    // Clock and button debouncing
    // =========================================================
    logic clk_25m;
    clk_divider u_clk_div (
        .clk_100m(clk), .rst_n(rst_n), .clk_25m(clk_25m)
    );

    logic pulse_U, pulse_D, pulse_L, pulse_R, pulse_C;
    button_debouncer db_U (.clk(clk), .rst_n(rst_n), .btn_in(btnU), .btn_pulse(pulse_U));
    button_debouncer db_D (.clk(clk), .rst_n(rst_n), .btn_in(btnD), .btn_pulse(pulse_D));
    button_debouncer db_L (.clk(clk), .rst_n(rst_n), .btn_in(btnL), .btn_pulse(pulse_L));
    button_debouncer db_R (.clk(clk), .rst_n(rst_n), .btn_in(btnR), .btn_pulse(pulse_R));
    button_debouncer db_C (.clk(clk), .rst_n(rst_n), .btn_in(btnC), .btn_pulse(pulse_C));

    // =========================================================
    // Cursor
    // =========================================================
    logic [3:0] cursor_x, cursor_y;
    cursor_logic u_cursor (
        .clk      (clk),    .rst_n(rst_n),
        .btn_up   (pulse_U), .btn_down(pulse_D),
        .btn_left (pulse_L), .btn_right(pulse_R),
        .cursor_x (cursor_x), .cursor_y(cursor_y)
    );

    // =========================================================
    // Game logic signals
    // =========================================================
    logic [161:0] board_state;
    logic [1:0]   current_turn;
    logic         start_rule_check, rule_check_done, is_legal_move;
    logic         update_board;
    logic [80:0]  capture_mask;
    logic         start_count, count_done, game_over;
    logic [7:0]   p1_score, p2_score;
    logic [161:0] territory_map;
    logic [1:0]   game_mode;
    logic         start_ai, ai_done;
    logic [3:0]   ai_cursor_x, ai_cursor_y;
    logic         is_in_menu;
    logic [2:0]   fsm_debug_signal;

    // =========================================================
    // AI debug signals
    // =========================================================
    logic [3:0]  ai_debug_state;
    logic        ai_debug_eval;
    logic        ai_debug_final;
    logic        ai_debug_send;
    logic        ai_debug_error;
    logic [7:0]  ai_layer_cnt;
    logic [3:0]  ai_layer_idx;
    logic [3:0]  ai_conv_state;
    logic [2:0]  ai_lc_state;
    logic [2:0]  ai_eng_state;

    // =========================================================
    // AI core
    // =========================================================
    go_ai_core #(
        .CLK_FREQ (100_000_000),
        .BAUD_RATE(115_200)
    ) u_ai_core (
        .clk              (clk),
        .rst_n            (rst_n),
        .start_ai         (start_ai),
        .board_state      (board_state),
        .uart_rx          (RsRx),
        .uart_tx          (RsTx),
        .ai_done          (ai_done),
        .best_move_x      (ai_cursor_x),
        .best_move_y      (ai_cursor_y),
        .debug_state      (ai_debug_state),
        .debug_eval_valid (ai_debug_eval),
        .debug_final_valid(ai_debug_final),
        .debug_send_done  (ai_debug_send),
        .debug_pkt_error  (ai_debug_error),
        .debug_layer_cnt  (ai_layer_cnt),
        .debug_layer_idx  (ai_layer_idx),
        .debug_conv_state (ai_conv_state),
        .debug_lc_state   (ai_lc_state),
        .debug_eng_state  (ai_eng_state)
    );

    // Route cursor: in PvAI mode during AI's turn, use AI-computed position
    logic [3:0] active_cursor_x, active_cursor_y;
    assign active_cursor_x = (game_mode == 2'b01 && current_turn == 2'b10) ? ai_cursor_x : cursor_x;
    assign active_cursor_y = (game_mode == 2'b01 && current_turn == 2'b10) ? ai_cursor_y : cursor_y;

    // =========================================================
    // Territory counter
    // =========================================================
    territory_counter u_counter (
        .clk           (clk),   .rst_n(rst_n),
        .start_counting(start_count),
        .board_state   (board_state),
        .counting_done (count_done),
        .black_score   (p1_score),
        .white_score   (p2_score),
        .territory_map (territory_map)
    );

    // =========================================================
    // Game FSM
    // =========================================================
    game_fsm u_fsm (
        .clk             (clk),   .rst_n(rst_n),
        .place_btn       (pulse_C),
        .btn_left        (pulse_L), .btn_right(pulse_R),
        .end_game_sw     (end_game_sw),
        .count_done      (count_done),
        .rule_check_done (rule_check_done),
        .is_legal_move   (is_legal_move),
        .start_rule_check(start_rule_check),
        .update_board    (update_board),
        .current_turn    (current_turn),
        .start_count     (start_count),
        .game_over       (game_over),
        .ai_done         (ai_done),
        .game_mode       (game_mode),
        .start_ai        (start_ai),
        .is_in_menu      (is_in_menu),
        .fsm_state_debug (fsm_debug_signal)
    );

    // =========================================================
    // Rule engine
    // =========================================================
    rule_engine u_rule (
        .clk          (clk),   .rst_n(rst_n),
        .start_check  (start_rule_check),
        .board_state  (board_state),
        .current_turn (current_turn),
        .check_done   (rule_check_done),
        .is_legal     (is_legal_move),
        .capture_mask (capture_mask),
        .cursor_x     (active_cursor_x),
        .cursor_y     (active_cursor_y)
    );

    // =========================================================
    // Board manager
    // =========================================================
    board_manager u_board (
        .clk          (clk),   .rst_n(rst_n),
        .update_board (update_board),
        .current_turn (current_turn),
        .capture_mask (capture_mask),
        .board_state  (board_state),
        .cursor_x     (active_cursor_x),
        .cursor_y     (active_cursor_y)
    );

    // =========================================================
    // VGA controller
    // =========================================================
    vga_controller u_vga (
        .clk_25m      (clk_25m),
        .rst_n        (rst_n),
        .cursor_x     (cursor_x),
        .cursor_y     (cursor_y),
        .game_over    (game_over),
        .is_in_menu   (is_in_menu),
        .mode_sel     (game_mode[0]),
        .p1_score     (p1_score),
        .p2_score     (p2_score),
        .territory_map(territory_map),
        .board_state  (board_state),
        .current_turn (current_turn),
        .vga_hsync    (Hsync),
        .vga_vsync    (Vsync),
        .vga_r        (vgaRed),
        .vga_g        (vgaGreen),
        .vga_b        (vgaBlue)
    );

    // =========================================================
    // LED debug assignments
    // =========================================================
    assign led[2:0]   = fsm_debug_signal;
    assign led[6:3]   = ai_debug_state;
    assign led[7]     = ai_done;
    assign led[11:8]  = ai_layer_idx;
    assign led[14:12] = ai_conv_state[2:0];
    assign led[15]    = (ai_layer_cnt > 8'd0);

endmodule
