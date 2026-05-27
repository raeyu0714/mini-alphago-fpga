`timescale 1ns / 1ps

// Moves a 9x9 board cursor in response to debounced button pulses.
// Cursor is clamped to [0, 8] on both axes.
module cursor_logic (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       btn_up,
    input  logic       btn_down,
    input  logic       btn_left,
    input  logic       btn_right,
    output logic [3:0] cursor_x,
    output logic [3:0] cursor_y
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cursor_x <= 4'd4;
            cursor_y <= 4'd4;
        end else begin
            if      (btn_up    && cursor_y > 0) cursor_y <= cursor_y - 1;
            else if (btn_down  && cursor_y < 8) cursor_y <= cursor_y + 1;

            if      (btn_left  && cursor_x > 0) cursor_x <= cursor_x - 1;
            else if (btn_right && cursor_x < 8) cursor_x <= cursor_x + 1;
        end
    end
endmodule
