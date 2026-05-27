`timescale 1ns / 1ps

// BFS flood-fill to find all stones in a connected group and count its liberties.
//
// Algorithm (one clock per iteration):
//   PROPAGATE: repeatedly OR in same-colour neighbors until group_mask converges.
//   COUNT:     popcount of empty intersections adjacent to any group stone.
//   FINISH:    assert done for one cycle.
//
// board_state encoding: 2 bits per cell, 2'b00=empty, 2'b01=black, 2'b10=white.
module group_liberty_scanner (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [6:0]   start_idx,
    input  logic [161:0] board_state,
    input  logic [1:0]   target_color,
    output logic [6:0]   liberty_count,
    output logic [80:0]  group_mask,
    output logic         done
);
    typedef enum logic [1:0] {
        IDLE, PROPAGATE, COUNT, FINISH
    } state_t;

    state_t state;

    // Pre-compute per-cell colour and empty flags (combinational)
    logic [80:0] color_match, empty_mask;
    always_comb begin
        for (int i = 0; i < 81; i++) begin
            color_match[i] = (board_state[i*2 +: 2] == target_color);
            empty_mask[i]  = (board_state[i*2 +: 2] == 2'b00);
        end
    end

    // Expand group by one hop: add any same-colour cell adjacent to a current member
    logic [80:0] next_group_mask;
    always_comb begin
        for (int i = 0; i < 81; i++) begin
            logic has_neighbor;
            has_neighbor = 1'b0;
            if (i % 9 != 0) has_neighbor |= group_mask[i-1];   // west
            if (i % 9 != 8) has_neighbor |= group_mask[i+1];   // east
            if (i >= 9)     has_neighbor |= group_mask[i-9];   // north
            if (i < 72)     has_neighbor |= group_mask[i+9];   // south
            next_group_mask[i] = group_mask[i] | (color_match[i] & has_neighbor);
        end
    end

    // Count empty cells adjacent to the group (liberties)
    logic [6:0] popcount;
    always_comb begin
        popcount = 7'd0;
        for (int i = 0; i < 81; i++) begin
            logic adj;
            adj = 1'b0;
            if (i % 9 != 0) adj |= group_mask[i-1];
            if (i % 9 != 8) adj |= group_mask[i+1];
            if (i >= 9)     adj |= group_mask[i-9];
            if (i < 72)     adj |= group_mask[i+9];
            if (empty_mask[i] & adj)
                popcount = popcount + 7'd1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            group_mask    <= 81'd0;
            liberty_count <= 7'd0;
            done          <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        group_mask            <= 81'd0;
                        group_mask[start_idx] <= 1'b1;
                        state                 <= PROPAGATE;
                    end
                end

                PROPAGATE: begin
                    if (group_mask == next_group_mask)
                        state <= COUNT;          // converged
                    else
                        group_mask <= next_group_mask;
                end

                COUNT: begin
                    liberty_count <= popcount;
                    state         <= FINISH;
                end

                FINISH: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
