`timescale 1ns / 1ps

// Divide 100 MHz input clock by 4 to produce 25 MHz for VGA.
// A 1-bit counter toggles the output every 2 cycles.
module clk_divider (
    input  logic clk_100m,
    input  logic rst_n,
    output logic clk_25m
);
    logic counter;

    always_ff @(posedge clk_100m or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 1'b0;
            clk_25m <= 1'b0;
        end else begin
            if (counter == 1'b1) begin
                counter <= 1'b0;
                clk_25m <= ~clk_25m;
            end else begin
                counter <= counter + 1'b1;
            end
        end
    end
endmodule
