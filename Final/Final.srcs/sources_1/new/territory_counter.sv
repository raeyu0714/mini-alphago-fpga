`timescale 1ns / 1ps

// End-game territory counter (Chinese area rules).
// Scans all 81 intersections:
//   - Black/white stones are counted directly.
//   - Empty regions are flood-filled with group_liberty_scanner (target_color=2'b00).
//     If the region touches only black -> black territory.
//     If the region touches only white -> white territory.
//     Contested regions are not counted.
// Note: komi (white bonus points) is not applied here.
module territory_counter (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start_counting,
    input  logic [161:0] board_state,
    output logic         counting_done,
    output logic [7:0]   black_score,
    output logic [7:0]   white_score,
    output logic [161:0] territory_map
);
    typedef enum logic [2:0] {
        IDLE, SCAN_BOARD, WAIT_SCAN, EVAL_TERRITORY, FINISH
    } state_t;

    state_t      state;
    logic [6:0]  idx;
    logic [80:0] visited_mask;
    logic        scan_start;
    logic        scan_done;
    logic [80:0] scan_group_mask;

    group_liberty_scanner u_empty_scanner (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (scan_start),
        .start_idx    (idx),
        .board_state  (board_state),
        .target_color (2'b00),      // scan empty regions
        .liberty_count(),           // unused for territory counting
        .group_mask   (scan_group_mask),
        .done         (scan_done)
    );

    // Determine which colours border the scanned empty region
    logic touches_black, touches_white;
    logic [6:0] group_size;

    always_comb begin
        touches_black = 1'b0;
        touches_white = 1'b0;
        group_size    = 7'd0;

        for (int i = 0; i < 81; i++) begin
            if (scan_group_mask[i]) begin
                group_size = group_size + 7'd1;

                if (i >= 9) begin
                    if (board_state[(i-9)*2 +: 2] == 2'b01) touches_black = 1'b1;
                    if (board_state[(i-9)*2 +: 2] == 2'b10) touches_white = 1'b1;
                end
                if (i < 72) begin
                    if (board_state[(i+9)*2 +: 2] == 2'b01) touches_black = 1'b1;
                    if (board_state[(i+9)*2 +: 2] == 2'b10) touches_white = 1'b1;
                end
                if (i % 9 != 0) begin
                    if (board_state[(i-1)*2 +: 2] == 2'b01) touches_black = 1'b1;
                    if (board_state[(i-1)*2 +: 2] == 2'b10) touches_white = 1'b1;
                end
                if (i % 9 != 8) begin
                    if (board_state[(i+1)*2 +: 2] == 2'b01) touches_black = 1'b1;
                    if (board_state[(i+1)*2 +: 2] == 2'b10) touches_white = 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            counting_done <= 1'b0;
            black_score   <= 8'd0;
            white_score   <= 8'd0;
            territory_map <= 162'd0;
            idx           <= 7'd0;
            visited_mask  <= 81'd0;
            scan_start    <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    counting_done <= 1'b0;
                    if (start_counting) begin
                        black_score   <= 8'd0;
                        white_score   <= 8'd0;
                        territory_map <= 162'd0;
                        idx           <= 7'd0;
                        visited_mask  <= 81'd0;
                        state         <= SCAN_BOARD;
                    end
                end

                SCAN_BOARD: begin
                    if (idx == 7'd81) begin
                        state <= FINISH;
                    end else if (board_state[idx*2 +: 2] == 2'b01) begin
                        black_score               <= black_score + 8'd1;
                        territory_map[idx*2 +: 2] <= 2'b01;
                        idx                       <= idx + 7'd1;
                    end else if (board_state[idx*2 +: 2] == 2'b10) begin
                        white_score               <= white_score + 8'd1;
                        territory_map[idx*2 +: 2] <= 2'b10;
                        idx                       <= idx + 7'd1;
                    end else begin
                        if (visited_mask[idx]) begin
                            idx <= idx + 7'd1;
                        end else begin
                            scan_start <= 1'b1;
                            state      <= WAIT_SCAN;
                        end
                    end
                end

                WAIT_SCAN: begin
                    scan_start <= 1'b0;
                    if (scan_done) state <= EVAL_TERRITORY;
                end

                EVAL_TERRITORY: begin
                    visited_mask <= visited_mask | scan_group_mask;

                    if (touches_black && !touches_white) begin
                        black_score <= black_score + group_size;
                        for (int i = 0; i < 81; i++) begin
                            if (scan_group_mask[i])
                                territory_map[i*2 +: 2] <= 2'b01;
                        end
                    end else if (!touches_black && touches_white) begin
                        white_score <= white_score + group_size;
                        for (int i = 0; i < 81; i++) begin
                            if (scan_group_mask[i])
                                territory_map[i*2 +: 2] <= 2'b10;
                        end
                    end
                    // Contested territory: no points awarded

                    idx   <= idx + 7'd1;
                    state <= SCAN_BOARD;
                end

                FINISH: begin
                    counting_done <= 1'b1;
                    if (!start_counting) state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
