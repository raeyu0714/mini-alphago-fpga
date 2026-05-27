`timescale 1ns / 1ps

// Parses two UART packet types sent from the PC MCTS engine:
//
//   0xCC (26 bytes): CNN evaluation request
//     [0]     = 0xCC header
//     [1-11]  = current-player bitboard (81 bits, little-endian, 11 bytes)
//     [12-22] = opponent bitboard       (81 bits, little-endian, 11 bytes)
//     [23]    = last_move  (0-80, 255 = none)
//     [24]    = prev_move  (0-80, 255 = none)
//     [25]    = 0x55 footer
//
//   0xDD (4 bytes): final AI move
//     [0]  = 0xDD header
//     [1]  = x coordinate (0-8)
//     [2]  = y coordinate (0-8)
//     [3]  = 0x55 footer
module packet_parser (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [7:0]   rx_data,
    input  logic         rx_valid,
    output logic [161:0] eval_board_out,
    output logic         eval_board_valid,
    output logic [7:0]   rx_last_move,
    output logic [7:0]   rx_prev_move,
    output logic [3:0]   final_x,
    output logic [3:0]   final_y,
    output logic         final_move_valid,
    output logic         packet_error
);
    typedef enum logic [1:0] {
        WAIT_HEADER,
        RECV_EVAL,
        RECV_FINAL,
        CHECK_FOOTER
    } state_t;

    state_t      state;
    logic [7:0]  rx_buf [0:25];
    logic [4:0]  rx_idx;

    // Combinational board assembly from rx_buf (runs continuously; registered on valid footer)
    logic [80:0]  black_plane_comb;
    logic [80:0]  white_plane_comb;
    logic [161:0] board_assembled_comb;

    always_comb begin
        black_plane_comb = 81'd0;
        white_plane_comb = 81'd0;

        // Black bitboard: rx_buf[1..11], 81 bits little-endian
        black_plane_comb[7:0]   = rx_buf[1];
        black_plane_comb[15:8]  = rx_buf[2];
        black_plane_comb[23:16] = rx_buf[3];
        black_plane_comb[31:24] = rx_buf[4];
        black_plane_comb[39:32] = rx_buf[5];
        black_plane_comb[47:40] = rx_buf[6];
        black_plane_comb[55:48] = rx_buf[7];
        black_plane_comb[63:56] = rx_buf[8];
        black_plane_comb[71:64] = rx_buf[9];
        black_plane_comb[79:72] = rx_buf[10];
        black_plane_comb[80]    = rx_buf[11][0];

        // White bitboard: rx_buf[12..22]
        white_plane_comb[7:0]   = rx_buf[12];
        white_plane_comb[15:8]  = rx_buf[13];
        white_plane_comb[23:16] = rx_buf[14];
        white_plane_comb[31:24] = rx_buf[15];
        white_plane_comb[39:32] = rx_buf[16];
        white_plane_comb[47:40] = rx_buf[17];
        white_plane_comb[55:48] = rx_buf[18];
        white_plane_comb[63:56] = rx_buf[19];
        white_plane_comb[71:64] = rx_buf[20];
        white_plane_comb[79:72] = rx_buf[21];
        white_plane_comb[80]    = rx_buf[22][0];

        // Pack into 2-bit-per-cell board_state format
        board_assembled_comb = 162'd0;
        for (int k = 0; k < 81; k++) begin
            if      (black_plane_comb[k]) board_assembled_comb[k*2 +: 2] = 2'b01;
            else if (white_plane_comb[k]) board_assembled_comb[k*2 +: 2] = 2'b10;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= WAIT_HEADER;
            rx_idx           <= '0;
            eval_board_out   <= '0;
            eval_board_valid <= 1'b0;
            rx_last_move     <= 8'd255;
            rx_prev_move     <= 8'd255;
            final_x          <= '0;
            final_y          <= '0;
            final_move_valid <= 1'b0;
            packet_error     <= 1'b0;
            for (int i = 0; i < 26; i++) rx_buf[i] <= 8'd0;
        end else begin
            eval_board_valid <= 1'b0;
            final_move_valid <= 1'b0;
            packet_error     <= 1'b0;

            case (state)
                WAIT_HEADER: begin
                    if (rx_valid) begin
                        if (rx_data == 8'hCC) begin
                            rx_buf[0] <= rx_data;
                            rx_idx    <= 5'd1;
                            state     <= RECV_EVAL;
                        end else if (rx_data == 8'hDD) begin
                            rx_buf[0] <= rx_data;
                            rx_idx    <= 5'd1;
                            state     <= RECV_FINAL;
                        end
                        // Unknown bytes are silently discarded
                    end
                end

                RECV_EVAL: begin
                    if (rx_valid) begin
                        rx_buf[rx_idx] <= rx_data;
                        if (rx_idx == 5'd24) begin
                            rx_idx <= rx_idx + 1'b1;
                            state  <= CHECK_FOOTER;
                        end else begin
                            rx_idx <= rx_idx + 1'b1;
                        end
                    end
                end

                RECV_FINAL: begin
                    if (rx_valid) begin
                        rx_buf[rx_idx] <= rx_data;
                        if (rx_idx == 5'd2) begin
                            rx_idx <= rx_idx + 1'b1;
                            state  <= CHECK_FOOTER;
                        end else begin
                            rx_idx <= rx_idx + 1'b1;
                        end
                    end
                end

                CHECK_FOOTER: begin
                    if (rx_valid) begin
                        if (rx_data == 8'h55) begin
                            if (rx_buf[0] == 8'hCC) begin
                                eval_board_out   <= board_assembled_comb;
                                rx_last_move     <= rx_buf[23];
                                rx_prev_move     <= rx_buf[24];
                                eval_board_valid <= 1'b1;
                            end else if (rx_buf[0] == 8'hDD) begin
                                final_x          <= rx_buf[1][3:0];
                                final_y          <= rx_buf[2][3:0];
                                final_move_valid <= 1'b1;
                            end
                        end else begin
                            packet_error <= 1'b1;
                        end
                        rx_idx <= '0;
                        state  <= WAIT_HEADER;
                    end
                end

                default: state <= WAIT_HEADER;
            endcase
        end
    end
endmodule
