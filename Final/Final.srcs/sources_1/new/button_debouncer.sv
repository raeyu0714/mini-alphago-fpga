`timescale 1ns / 1ps

// Debounces a mechanical button and produces a single-cycle pulse.
// The input must be stable high for ~10 ms (20-bit counter at 100 MHz)
// before the output pulse is generated.
module button_debouncer (
    input  logic clk,
    input  logic rst_n,
    input  logic btn_in,
    output logic btn_pulse
);
    logic [19:0] delay_cnt;
    logic        dff1, dff2, dff3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            delay_cnt <= '0;
            dff1 <= 1'b0;
            dff2 <= 1'b0;
            dff3 <= 1'b0;
        end else begin
            dff1 <= btn_in;
            if (dff1 == 1'b1) begin
                if (delay_cnt == 20'hFFFFF)
                    dff2 <= 1'b1;
                else
                    delay_cnt <= delay_cnt + 1'b1;
            end else begin
                delay_cnt <= '0;
                dff2      <= 1'b0;
            end
            dff3 <= dff2;
        end
    end

    // Rising-edge detect on dff2 produces a one-cycle pulse
    assign btn_pulse = dff2 & ~dff3;
endmodule
