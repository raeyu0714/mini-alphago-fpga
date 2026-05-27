`timescale 1ns / 1ps

// Main game FSM.
//
// States:
//   S_MENU        - mode selection screen (PvP / PvAI)
//   S_WAIT_INPUT  - wait for human button press or trigger AI turn
//   S_AI_TURN     - wait for go_ai_core to finish its MCTS search
//   S_CHECK_RULES - wait for rule_engine to validate the move
//   S_COMMIT_MOVE - assert update_board for one cycle
//   S_SWITCH_TURN - toggle current_turn, return to S_WAIT_INPUT
//   S_SCORING     - trigger territory_counter, wait for count_done
//   S_GAME_OVER   - hold until end_game_sw is de-asserted
//
// game_mode encoding:
//   2'b00 = PvP (both sides human)
//   2'b01 = PvAI (black = human, white = AI)
module game_fsm (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       place_btn,
    input  logic       btn_left,
    input  logic       btn_right,
    input  logic       rule_check_done,
    input  logic       end_game_sw,
    input  logic       count_done,
    input  logic       is_legal_move,
    input  logic       ai_done,
    output logic       start_rule_check,
    output logic       update_board,
    output logic [1:0] current_turn,
    output logic       start_count,
    output logic       game_over,
    output logic [1:0] game_mode,
    output logic       start_ai,
    output logic       is_in_menu,
    output logic [2:0] fsm_state_debug
);
    typedef enum logic [2:0] {
        S_MENU,
        S_WAIT_INPUT,
        S_AI_TURN,
        S_CHECK_RULES,
        S_COMMIT_MOVE,
        S_SWITCH_TURN,
        S_SCORING,
        S_GAME_OVER
    } game_state_t;

    game_state_t state;

    assign is_in_menu      = (state == S_MENU);
    assign fsm_state_debug = state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_MENU;
            game_mode        <= 2'b00;
            current_turn     <= 2'b01;
            start_ai         <= 1'b0;
            start_rule_check <= 1'b0;
            update_board     <= 1'b0;
            start_count      <= 1'b0;
            game_over        <= 1'b0;
        end else begin
            // Global override: end_game_sw triggers scoring from any active state
            if (end_game_sw && state != S_SCORING && state != S_GAME_OVER) begin
                state       <= S_SCORING;
                start_count <= 1'b1;

            end else if (!end_game_sw && state == S_GAME_OVER) begin
                state     <= S_WAIT_INPUT;
                game_over <= 1'b0;

            end else begin
                case (state)
                    S_MENU: begin
                        if      (btn_left)  game_mode <= 2'b00;
                        else if (btn_right) game_mode <= 2'b01;
                        if (place_btn)      state     <= S_WAIT_INPUT;
                    end

                    S_WAIT_INPUT: begin
                        // Automatically start AI when it is white's turn in PvAI mode
                        if (game_mode == 2'b01 && current_turn == 2'b10) begin
                            start_ai <= 1'b1;
                            state    <= S_AI_TURN;
                        end else begin
                            if (place_btn) begin
                                start_rule_check <= 1'b1;
                                state            <= S_CHECK_RULES;
                            end
                        end
                    end

                    S_AI_TURN: begin
                        start_ai <= 1'b0;
                        if (ai_done) begin
                            start_rule_check <= 1'b1;
                            state            <= S_CHECK_RULES;
                        end
                    end

                    S_CHECK_RULES: begin
                        start_rule_check <= 1'b0;
                        if (rule_check_done) begin
                            if (is_legal_move) begin
                                state <= S_COMMIT_MOVE;
                            end else begin
                                // AI illegal move counts as a pass; human must retry
                                if (game_mode == 2'b01 && current_turn == 2'b10)
                                    state <= S_SWITCH_TURN;
                                else
                                    state <= S_WAIT_INPUT;
                            end
                        end
                    end

                    S_COMMIT_MOVE: begin
                        update_board <= 1'b1;
                        state        <= S_SWITCH_TURN;
                    end

                    S_SWITCH_TURN: begin
                        update_board <= 1'b0;
                        current_turn <= (current_turn == 2'b01) ? 2'b10 : 2'b01;
                        state        <= S_WAIT_INPUT;
                    end

                    S_SCORING: begin
                        start_count <= 1'b0;
                        if (count_done) begin
                            game_over <= 1'b1;
                            state     <= S_GAME_OVER;
                        end
                    end

                    S_GAME_OVER: begin
                        // stay here until end_game_sw is released (handled above)
                    end

                    default: state <= S_WAIT_INPUT;
                endcase
            end
        end
    end
endmodule
