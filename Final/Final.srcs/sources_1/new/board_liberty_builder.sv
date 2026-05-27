`timescale 1ns / 1ps

// Scans all 81 board intersections and builds four 81-bit liberty bitmaps
// used as CNN input channels 2-5.
//
//   black_lib1[i] = 1  iff  the black group containing stone i has exactly 1 liberty
//   black_lib2[i] = 1  iff  the black group containing stone i has exactly 2 liberties
//   white_lib1[i] / white_lib2[i] = same for white
//
// Each group is scanned once; all stones in the group share the same result
// via group_mask OR-reduction.
module board_liberty_builder (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [161:0] board_state,
    output logic [80:0]  black_lib1,
    output logic [80:0]  black_lib2,
    output logic [80:0]  white_lib1,
    output logic [80:0]  white_lib2,
    output logic         done
);
    typedef enum logic [2:0] {
        S_IDLE,
        S_CHECK_STONE,
        S_WAIT_SCANNER,
        S_UPDATE_MASK,
        S_DONE
    } state_t;

    state_t      state;
    logic [6:0]  idx;
    logic [80:0] visited;

    logic        scan_start;
    logic [6:0]  scan_liberty_count;
    logic [80:0] scan_group_mask;
    logic        scan_done;
    logic [1:0]  target_color;

    assign target_color = board_state[idx*2 +: 2];

    group_liberty_scanner u_scanner (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (scan_start),
        .start_idx    (idx),
        .board_state  (board_state),
        .target_color (target_color),
        .liberty_count(scan_liberty_count),
        .group_mask   (scan_group_mask),
        .done         (scan_done)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            idx        <= 7'd0;
            visited    <= 81'd0;
            black_lib1 <= 81'd0;
            black_lib2 <= 81'd0;
            white_lib1 <= 81'd0;
            white_lib2 <= 81'd0;
            scan_start <= 1'b0;
            done       <= 1'b0;
        end else begin
            scan_start <= 1'b0;
            done       <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        idx        <= 7'd0;
                        visited    <= 81'd0;
                        black_lib1 <= 81'd0;
                        black_lib2 <= 81'd0;
                        white_lib1 <= 81'd0;
                        white_lib2 <= 81'd0;
                        state      <= S_CHECK_STONE;
                    end
                end

                S_CHECK_STONE: begin
                    if (idx > 7'd80) begin
                        state <= S_DONE;
                    end else if (visited[idx] || board_state[idx*2 +: 2] == 2'b00) begin
                        idx <= idx + 7'd1;  // skip empty or already-visited cells
                    end else begin
                        scan_start <= 1'b1;
                        state      <= S_WAIT_SCANNER;
                    end
                end

                S_WAIT_SCANNER: begin
                    if (scan_done) state <= S_UPDATE_MASK;
                end

                S_UPDATE_MASK: begin
                    visited <= visited | scan_group_mask;

                    if (target_color == 2'b01) begin  // black group
                        if      (scan_liberty_count == 7'd1) black_lib1 <= black_lib1 | scan_group_mask;
                        else if (scan_liberty_count == 7'd2) black_lib2 <= black_lib2 | scan_group_mask;
                    end else if (target_color == 2'b10) begin  // white group
                        if      (scan_liberty_count == 7'd1) white_lib1 <= white_lib1 | scan_group_mask;
                        else if (scan_liberty_count == 7'd2) white_lib2 <= white_lib2 | scan_group_mask;
                    end

                    idx   <= idx + 7'd1;
                    state <= S_CHECK_STONE;
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
