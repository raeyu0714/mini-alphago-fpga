`timescale 1ns / 1ps

module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
)(
    input  logic       clk,
    input  logic       rst_n,
    input  logic       rx,
    output logic [7:0] data,
    output logic       valid   // pulses high for one clock when a byte is ready
);
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;  // 868 at 100 MHz / 115200

    typedef enum logic [1:0] {
        IDLE, START, DATA, STOP
    } state_t;

    state_t     state;
    logic [9:0] clk_cnt;
    logic [2:0] bit_idx;
    logic [7:0] rx_shift;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            clk_cnt <= '0;
            bit_idx <= '0;
            data    <= '0;
            valid   <= 1'b0;
        end else begin
            valid <= 1'b0;

            case (state)
                IDLE: begin
                    if (rx == 1'b0)  begin  // falling edge = start bit
                        clk_cnt <= '0;
                        state   <= START;
                    end
                end

                START: begin
                    // sample at mid-point of start bit to confirm
                    if (clk_cnt == (CLKS_PER_BIT / 2 - 1)) begin
                        if (rx == 1'b0) begin
                            clk_cnt <= '0;
                            bit_idx <= '0;
                            state   <= DATA;
                        end else begin
                            state <= IDLE;  // glitch, abort
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt           <= '0;
                        rx_shift[bit_idx] <= rx;  // LSB first
                        if (bit_idx == 3'd7) begin
                            state <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        data    <= rx_shift;
                        valid   <= 1'b1;
                        clk_cnt <= '0;
                        state   <= IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
