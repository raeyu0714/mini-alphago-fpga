`timescale 1ns / 1ps

// Holds the 9x9 board state as a packed 162-bit vector.
// Each intersection uses 2 bits: 2'b00=empty, 2'b01=black, 2'b10=white.
//
// On update_board:
//   1. Clear all intersections flagged in capture_mask (opponent stones removed).
//   2. Place current_turn colour at the cursor position.
//   Both happen atomically in the same clock edge.
module board_manager (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         update_board,
    input  logic [3:0]   cursor_x,
    input  logic [3:0]   cursor_y,
    input  logic [1:0]   current_turn,
    input  logic [80:0]  capture_mask,
    output logic [161:0] board_state
);
    // Combinational index; synthesis maps this to a small look-up.
    // cursor_y and cursor_x are both bounded to [0,8] by cursor_logic,
    // so the maximum result is 8*9+8 = 80 which fits in 7 bits.
    logic [6:0] cursor_idx;
    assign cursor_idx = cursor_y * 4'd9 + cursor_x;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            board_state <= 162'd0;
        end else if (update_board) begin
            // Remove captured stones
            for (int i = 0; i < 81; i++) begin
                if (capture_mask[i])
                    board_state[i*2 +: 2] <= 2'b00;
            end
            // Place the new stone
            board_state[cursor_idx*2 +: 2] <= current_turn;
        end
    end
endmodule
