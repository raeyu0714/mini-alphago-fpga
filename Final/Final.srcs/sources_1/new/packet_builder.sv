`timescale 1ns / 1ps

// Serialises three outgoing UART packet types:
//
//   0xAA (25 bytes): board state notification -> sent to PC at start of AI turn
//     [0]     = 0xAA header
//     [1]     = 0x01 (fixed)
//     [2-12]  = black bitboard (81 bits, little-endian, 11 bytes)
//     [13-23] = white bitboard (81 bits, little-endian, 11 bytes)
//     [24]    = 0x55 footer
//
//   0xBB (85 bytes): CNN result -> sent to PC after each MCTS simulation
//     [0]     = 0xBB header
//     [1-81]  = policy logits (81 x INT8)
//     [82-83] = value (INT16, big-endian)
//     [84]    = 0x55 footer
//
//   0xDD (variable, debug only): raw feature-map dump
//     [0]     = 0xDD header
//     [1]     = layer index
//     [2..]   = raw bytes from CNN BRAM
//     [last]  = 0x55 footer
module packet_builder (
    input  logic         clk,
    input  logic         rst_n,

    // Normal operation triggers
    input  logic         send_request_start,  // send 0xAA board notification
    input  logic         send_result_start,   // send 0xBB CNN result
    input  logic [161:0] board_state_in,
    input  logic [7:0]   policy_in [0:80],
    input  logic [15:0]  value_in,

    // Debug dump trigger (0xDD)
    input  logic         send_debug_start,
    input  logic [3:0]   debug_layer_idx,
    input  logic [6:0]   debug_ch_num,
    output logic [13:0]  debug_read_addr,
    input  logic [7:0]   debug_read_data,

    // UART TX interface
    output logic [7:0]   tx_data,
    output logic         tx_start,
    input  logic         tx_busy,
    output logic         send_done,
    output logic         busy
);
    typedef enum logic [2:0] {
        IDLE,
        SEND_AA,
        SEND_BB,
        SEND_DD_HDR,   // debug: send 0xDD header + layer ID
        SEND_DD_DATA,  // debug: stream BRAM bytes
        SEND_DD_FTR,   // debug: send 0x55 footer
        FINISH
    } state_t;

    state_t      state;
    logic [7:0]  tx_buf [0:84];
    logic [13:0] tx_idx;
    logic [13:0] total_len;
    logic [6:0]  dbg_ch;
    logic [6:0]  dbg_pos;

    // Split board_state into separate black/white bitplanes
    logic [80:0] black_plane, white_plane;
    always_comb begin
        for (int k = 0; k < 81; k++) begin
            black_plane[k] = (board_state_in[k*2 +: 2] == 2'b01);
            white_plane[k] = (board_state_in[k*2 +: 2] == 2'b10);
        end
    end

    // Pack 0xAA board notification into tx_buf
    task automatic pack_aa;
        begin
            tx_buf[0]  <= 8'hAA;
            tx_buf[1]  <= 8'h01;
            tx_buf[2]  <= black_plane[7:0];    tx_buf[3]  <= black_plane[15:8];
            tx_buf[4]  <= black_plane[23:16];  tx_buf[5]  <= black_plane[31:24];
            tx_buf[6]  <= black_plane[39:32];  tx_buf[7]  <= black_plane[47:40];
            tx_buf[8]  <= black_plane[55:48];  tx_buf[9]  <= black_plane[63:56];
            tx_buf[10] <= black_plane[71:64];  tx_buf[11] <= black_plane[79:72];
            tx_buf[12] <= {7'b0, black_plane[80]};
            tx_buf[13] <= white_plane[7:0];    tx_buf[14] <= white_plane[15:8];
            tx_buf[15] <= white_plane[23:16];  tx_buf[16] <= white_plane[31:24];
            tx_buf[17] <= white_plane[39:32];  tx_buf[18] <= white_plane[47:40];
            tx_buf[19] <= white_plane[55:48];  tx_buf[20] <= white_plane[63:56];
            tx_buf[21] <= white_plane[71:64];  tx_buf[22] <= white_plane[79:72];
            tx_buf[23] <= {7'b0, white_plane[80]};
            tx_buf[24] <= 8'h55;
        end
    endtask

    // Pack 0xBB CNN result into tx_buf
    task automatic pack_bb;
        integer kk;
        begin
            tx_buf[0] <= 8'hBB;
            for (kk = 0; kk < 81; kk++)
                tx_buf[kk+1] <= policy_in[kk];
            tx_buf[82] <= value_in[15:8];  // big-endian INT16
            tx_buf[83] <= value_in[7:0];
            tx_buf[84] <= 8'h55;
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            tx_data         <= 8'd0;
            tx_start        <= 1'b0;
            tx_idx          <= '0;
            total_len       <= '0;
            send_done       <= 1'b0;
            busy            <= 1'b0;
            debug_read_addr <= 14'd0;
            dbg_ch          <= 7'd0;
            dbg_pos         <= 7'd0;
            for (int i = 0; i < 85; i++) tx_buf[i] <= 8'd0;
        end else begin
            tx_start  <= 1'b0;
            send_done <= 1'b0;

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (send_request_start) begin
                        pack_aa();
                        tx_idx    <= '0;
                        total_len <= 14'd25;
                        busy      <= 1'b1;
                        state     <= SEND_AA;
                    end else if (send_result_start) begin
                        pack_bb();
                        tx_idx    <= '0;
                        total_len <= 14'd85;
                        busy      <= 1'b1;
                        state     <= SEND_BB;
                    end else if (send_debug_start) begin
                        total_len <= {7'd0, debug_ch_num} * 14'd81;
                        tx_idx    <= '0;
                        busy      <= 1'b1;
                        state     <= SEND_DD_HDR;
                    end
                end

                SEND_AA, SEND_BB: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= tx_buf[tx_idx];
                        tx_start <= 1'b1;
                        if (tx_idx == total_len - 1)
                            state  <= FINISH;
                        else
                            tx_idx <= tx_idx + 14'd1;
                    end
                end

                SEND_DD_HDR: begin
                    if (!tx_busy && !tx_start) begin
                        if (tx_idx == 14'd0) begin
                            tx_data  <= 8'hDD;
                            tx_start <= 1'b1;
                            tx_idx   <= 14'd1;
                        end else begin
                            tx_data         <= {4'd0, debug_layer_idx};
                            tx_start        <= 1'b1;
                            tx_idx          <= 14'd0;
                            debug_read_addr <= 14'd0;
                            dbg_ch          <= 7'd0;
                            dbg_pos         <= 7'd0;
                            state           <= SEND_DD_DATA;
                        end
                    end
                end

                SEND_DD_DATA: begin
                    // debug_read_data reflects the address set in the previous cycle (BRAM latency = 1)
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= debug_read_data;
                        tx_start <= 1'b1;

                        if (tx_idx == total_len - 14'd1) begin
                            state <= SEND_DD_FTR;
                        end else begin
                            tx_idx <= tx_idx + 14'd1;
                            if (dbg_pos == 7'd80) begin
                                dbg_pos         <= 7'd0;
                                dbg_ch          <= dbg_ch + 7'd1;
                                debug_read_addr <= {dbg_ch + 7'd1, 7'd0};
                            end else begin
                                dbg_pos         <= dbg_pos + 7'd1;
                                debug_read_addr <= {dbg_ch, dbg_pos + 7'd1};
                            end
                        end
                    end
                end

                SEND_DD_FTR: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= 8'h55;
                        tx_start <= 1'b1;
                        state    <= FINISH;
                    end
                end

                FINISH: begin
                    if (!tx_busy) begin
                        send_done <= 1'b1;
                        busy      <= 1'b0;
                        state     <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
