`timescale 1ns / 1ps

// Validates a proposed move and builds the capture_mask for board_manager.
//
// Checks (in order):
//   1. Target cell is empty.
//   2. Scan each of the four neighbors; if the neighbor is an enemy group with
//      zero liberties after the move, add it to capture_mask.
//   3. If the placed stone's own group has zero liberties AND nothing was
//      captured, the move is suicide -> illegal.
//
// Note: Ko rule is not implemented; the human player can re-capture immediately.
module rule_engine (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start_check,
    input  logic [161:0] board_state,
    input  logic [3:0]   cursor_x,
    input  logic [3:0]   cursor_y,
    input  logic [1:0]   current_turn,
    output logic         check_done,
    output logic         is_legal,
    output logic [80:0]  capture_mask
);
    // Flat index of the target cell
    logic [6:0] target_idx;
    assign target_idx = cursor_y * 4'd9 + cursor_x;

    logic [1:0] enemy_color;
    assign enemy_color = (current_turn == 2'b01) ? 2'b10 : 2'b01;

    // Simulate the board with the stone already placed (used for liberty checks)
    logic [161:0] sim_board;
    always_comb begin
        sim_board = board_state;
        sim_board[target_idx*2 +: 2] = current_turn;
    end

    // Neighbor validity and indices
    logic       valid_n, valid_s, valid_w, valid_e;
    logic [6:0] idx_n, idx_s, idx_w, idx_e;
    assign valid_n = (cursor_y > 0);
    assign valid_s = (cursor_y < 8);
    assign valid_w = (cursor_x > 0);
    assign valid_e = (cursor_x < 8);
    assign idx_n   = target_idx - 7'd9;
    assign idx_s   = target_idx + 7'd9;
    assign idx_w   = target_idx - 7'd1;
    assign idx_e   = target_idx + 7'd1;

    // group_liberty_scanner interface
    logic        scan_start;
    logic        scan_done;
    logic [6:0]  scan_target_idx;
    logic [1:0]  scan_target_color;
    logic [6:0]  scan_liberty;
    logic [80:0] scan_group_mask;

    group_liberty_scanner u_scanner (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (scan_start),
        .start_idx    (scan_target_idx),
        .board_state  (sim_board),
        .target_color (scan_target_color),
        .liberty_count(scan_liberty),
        .group_mask   (scan_group_mask),
        .done         (scan_done)
    );

    typedef enum logic [3:0] {
        IDLE,
        CHECK_N, WAIT_N,
        CHECK_S, WAIT_S,
        CHECK_W, WAIT_W,
        CHECK_E, WAIT_E,
        CHECK_SELF, WAIT_SELF,
        FINISH
    } state_t;

    state_t state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= IDLE;
            scan_start       <= 1'b0;
            check_done       <= 1'b0;
            is_legal         <= 1'b0;
            capture_mask     <= 81'd0;
        end else begin
            case (state)
                IDLE: begin
                    check_done <= 1'b0;
                    scan_start <= 1'b0;
                    if (start_check) begin
                        if (board_state[target_idx*2 +: 2] != 2'b00) begin
                            // Cell occupied -> immediately illegal
                            is_legal     <= 1'b0;
                            capture_mask <= 81'd0;
                            check_done   <= 1'b1;
                        end else begin
                            capture_mask <= 81'd0;
                            state        <= CHECK_N;
                        end
                    end
                end

                // --- North neighbor ---
                CHECK_N: begin
                    if (valid_n && sim_board[idx_n*2 +: 2] == enemy_color) begin
                        scan_target_idx   <= idx_n;
                        scan_target_color <= enemy_color;
                        scan_start        <= 1'b1;
                        state             <= WAIT_N;
                    end else begin
                        state <= CHECK_S;
                    end
                end
                WAIT_N: begin
                    scan_start <= 1'b0;
                    if (scan_done) begin
                        if (scan_liberty == 7'd0)
                            capture_mask <= capture_mask | scan_group_mask;
                        state <= CHECK_S;
                    end
                end

                // --- South neighbor ---
                CHECK_S: begin
                    if (valid_s && sim_board[idx_s*2 +: 2] == enemy_color) begin
                        scan_target_idx   <= idx_s;
                        scan_target_color <= enemy_color;
                        scan_start        <= 1'b1;
                        state             <= WAIT_S;
                    end else begin
                        state <= CHECK_W;
                    end
                end
                WAIT_S: begin
                    scan_start <= 1'b0;
                    if (scan_done) begin
                        if (scan_liberty == 7'd0)
                            capture_mask <= capture_mask | scan_group_mask;
                        state <= CHECK_W;
                    end
                end

                // --- West neighbor ---
                CHECK_W: begin
                    if (valid_w && sim_board[idx_w*2 +: 2] == enemy_color) begin
                        scan_target_idx   <= idx_w;
                        scan_target_color <= enemy_color;
                        scan_start        <= 1'b1;
                        state             <= WAIT_W;
                    end else begin
                        state <= CHECK_E;
                    end
                end
                WAIT_W: begin
                    scan_start <= 1'b0;
                    if (scan_done) begin
                        if (scan_liberty == 7'd0)
                            capture_mask <= capture_mask | scan_group_mask;
                        state <= CHECK_E;
                    end
                end

                // --- East neighbor ---
                CHECK_E: begin
                    if (valid_e && sim_board[idx_e*2 +: 2] == enemy_color) begin
                        scan_target_idx   <= idx_e;
                        scan_target_color <= enemy_color;
                        scan_start        <= 1'b1;
                        state             <= WAIT_E;
                    end else begin
                        state <= CHECK_SELF;
                    end
                end
                WAIT_E: begin
                    scan_start <= 1'b0;
                    if (scan_done) begin
                        if (scan_liberty == 7'd0)
                            capture_mask <= capture_mask | scan_group_mask;
                        state <= CHECK_SELF;
                    end
                end

                // --- Suicide check: scan own group after all captures ---
                CHECK_SELF: begin
                    scan_target_idx   <= target_idx;
                    scan_target_color <= current_turn;
                    scan_start        <= 1'b1;
                    state             <= WAIT_SELF;
                end
                WAIT_SELF: begin
                    scan_start <= 1'b0;
                    if (scan_done) begin
                        // Suicide: own group has no liberties and nothing was captured
                        if (scan_liberty == 7'd0 && capture_mask == 81'd0)
                            is_legal <= 1'b0;
                        else
                            is_legal <= 1'b1;
                        state <= FINISH;
                    end
                end

                FINISH: begin
                    check_done <= 1'b1;
                    state      <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
