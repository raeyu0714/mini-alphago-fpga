`timescale 1ns / 1ps

// VGA controller for a 640x480 @ 60 Hz display (25 MHz pixel clock).
//
// Renders two screens:
//   Menu  (is_in_menu=1): "GO" title + PvP / PvAI mode selector boxes
//   Game  (is_in_menu=0): 9x9 board with stones, cursor, territory overlay
//
// Stone colour: black=12'h111, white=12'hFFF
// Board colour: wood tan 12'hDB7
// Cursor: red hollow square 12'hF00
module vga_controller (
    input  logic         clk_25m,
    input  logic         rst_n,
    input  logic [3:0]   cursor_x,
    input  logic [3:0]   cursor_y,
    input  logic [161:0] board_state,
    input  logic [1:0]   current_turn,
    input  logic         is_in_menu,
    input  logic         mode_sel,
    input  logic         game_over,
    input  logic [7:0]   p1_score,
    input  logic [7:0]   p2_score,
    input  logic [161:0] territory_map,
    output logic         vga_hsync,
    output logic         vga_vsync,
    output logic [3:0]   vga_r,
    output logic [3:0]   vga_g,
    output logic [3:0]   vga_b
);
    // =========================================================
    // Sync counter and video-active region
    // =========================================================
    logic [9:0] h_count, v_count;
    logic       video_on;
    logic [9:0] pixel_x, pixel_y;

    assign pixel_x = h_count;
    assign pixel_y = v_count;

    always_ff @(posedge clk_25m or negedge rst_n) begin
        if (!rst_n) begin
            h_count   <= 10'd0;
            v_count   <= 10'd0;
            vga_hsync <= 1'b1;
            vga_vsync <= 1'b1;
        end else begin
            if (h_count == 10'd799) begin
                h_count <= 10'd0;
                v_count <= (v_count == 10'd524) ? 10'd0 : v_count + 10'd1;
            end else begin
                h_count <= h_count + 10'd1;
            end
            vga_hsync <= ~(h_count >= 10'd656 && h_count < 10'd752);
            vga_vsync <= ~(v_count >= 10'd490 && v_count < 10'd492);
        end
    end

    assign video_on = (h_count < 10'd640 && v_count < 10'd480);

    // =========================================================
    // Menu elements
    // =========================================================
    // "G" letter (pixel-drawn rectangle outline)
    logic is_title_G, is_title_O;
    assign is_title_G =
        (pixel_x >= 235 && pixel_x <= 255 && pixel_y >= 60  && pixel_y <= 150) ||
        (pixel_x >= 235 && pixel_x <= 305 && pixel_y >= 60  && pixel_y <= 80)  ||
        (pixel_x >= 235 && pixel_x <= 305 && pixel_y >= 130 && pixel_y <= 150) ||
        (pixel_x >= 285 && pixel_x <= 305 && pixel_y >= 100 && pixel_y <= 150);

    // "O" letter
    assign is_title_O =
        (pixel_x >= 335 && pixel_x <= 355 && pixel_y >= 60  && pixel_y <= 150) ||
        (pixel_x >= 385 && pixel_x <= 405 && pixel_y >= 60  && pixel_y <= 150) ||
        (pixel_x >= 335 && pixel_x <= 405 && pixel_y >= 60  && pixel_y <= 80)  ||
        (pixel_x >= 335 && pixel_x <= 405 && pixel_y >= 130 && pixel_y <= 150);

    // Mode selector boxes (160x160 each)
    logic is_pvp_box, is_pvp_border, is_pvp_left;
    logic is_pve_box, is_pve_border;

    assign is_pvp_box    = (pixel_x >= 160 && pixel_x <= 280 && pixel_y >= 220 && pixel_y <= 340);
    assign is_pvp_border = is_pvp_box && (pixel_x < 170 || pixel_x > 270 || pixel_y < 230 || pixel_y > 330);
    assign is_pvp_left   = is_pvp_box && (pixel_x < 220);
    assign is_pve_box    = (pixel_x >= 360 && pixel_x <= 480 && pixel_y >= 220 && pixel_y <= 340);
    assign is_pve_border = is_pve_box && (pixel_x < 370 || pixel_x > 470 || pixel_y < 230 || pixel_y > 330);

    // =========================================================
    // In-game board elements
    // =========================================================
    logic is_board_bg, is_grid_line, is_vert_line, is_horz_line;
    logic in_grid_x, in_grid_y;

    assign is_board_bg = (pixel_x >= 140 && pixel_x <= 500) && (pixel_y >= 60 && pixel_y <= 420);
    assign in_grid_x   = (pixel_x >= 160 && pixel_x <= 480);
    assign in_grid_y   = (pixel_y >= 80  && pixel_y <= 400);
    assign is_vert_line = in_grid_x && in_grid_y && ((pixel_x - 10'd160) % 10'd40 == 0);
    assign is_horz_line = in_grid_x && in_grid_y && ((pixel_y - 10'd80)  % 10'd40 == 0);
    assign is_grid_line = is_vert_line || is_horz_line;

    // Cursor (hollow 11x11 square with 7x7 hole)
    logic [9:0] cursor_pixel_x, cursor_pixel_y;
    logic       is_cursor;

    assign cursor_pixel_x = 10'd160 + {6'd0, cursor_x} * 10'd40;
    assign cursor_pixel_y = 10'd80  + {6'd0, cursor_y} * 10'd40;
    assign is_cursor =
        (pixel_x > cursor_pixel_x - 10'd6) && (pixel_x < cursor_pixel_x + 10'd6) &&
        (pixel_y > cursor_pixel_y - 10'd6) && (pixel_y < cursor_pixel_y + 10'd6) &&
       ~((pixel_x > cursor_pixel_x - 10'd4) && (pixel_x < cursor_pixel_x + 10'd4) &&
         (pixel_y > cursor_pixel_y - 10'd4) && (pixel_y < cursor_pixel_y + 10'd4));

    // Map pixel to nearest board intersection
    logic [9:0] px_adj, py_adj;
    logic [3:0] nearest_x, nearest_y;
    logic [6:0] nearest_idx;
    logic [1:0] cell_state, cell_territory;

    assign px_adj = (pixel_x >= 10'd140) ? pixel_x - 10'd140 : 10'd0;
    assign py_adj = (pixel_y >= 10'd60)  ? pixel_y - 10'd60  : 10'd0;

    assign nearest_x = (px_adj < 40) ? 4'd0 : (px_adj < 80)  ? 4'd1 : (px_adj < 120) ? 4'd2 :
                       (px_adj < 160)? 4'd3 : (px_adj < 200) ? 4'd4 : (px_adj < 240) ? 4'd5 :
                       (px_adj < 280)? 4'd6 : (px_adj < 320) ? 4'd7 : 4'd8;
    assign nearest_y = (py_adj < 40) ? 4'd0 : (py_adj < 80)  ? 4'd1 : (py_adj < 120) ? 4'd2 :
                       (py_adj < 160)? 4'd3 : (py_adj < 200) ? 4'd4 : (py_adj < 240) ? 4'd5 :
                       (py_adj < 280)? 4'd6 : (py_adj < 320) ? 4'd7 : 4'd8;

    assign nearest_idx    = {3'd0, nearest_y} * 7'd9 + {3'd0, nearest_x};
    assign cell_state     = board_state   [nearest_idx*2 +: 2];
    assign cell_territory = territory_map [nearest_idx*2 +: 2];

    // Stone detection via squared distance to nearest intersection centre
    logic signed [10:0] dx, dy;
    logic [21:0]        dist_sq;
    logic               is_stone, is_territory_box;

    assign dx             = $signed({1'b0, pixel_x}) - $signed({1'b0, (10'd160 + {6'd0, nearest_x} * 10'd40)});
    assign dy             = $signed({1'b0, pixel_y}) - $signed({1'b0, (10'd80  + {6'd0, nearest_y} * 10'd40)});
    assign dist_sq        = (dx * dx) + (dy * dy);
    assign is_stone        = (dist_sq <= 22'd256) && (cell_state != 2'b00);
    assign is_territory_box = game_over && (cell_state == 2'b00) &&
                              (dx >= -11'sd6 && dx <= 11'sd6) && (dy >= -11'sd6 && dy <= 11'sd6);

    // Turn indicator circle (top-left corner, 60x60 box)
    logic                is_ui_box, is_ui_border, is_ui_stone;
    logic signed [10:0]  ui_dx, ui_dy;

    assign is_ui_box    = (pixel_x >= 30 && pixel_x <= 90) && (pixel_y >= 30 && pixel_y <= 90);
    assign is_ui_border = is_ui_box && (pixel_x < 34 || pixel_x > 86 || pixel_y < 34 || pixel_y > 86);
    assign ui_dx        = $signed({1'b0, pixel_x}) - 11'sd60;
    assign ui_dy        = $signed({1'b0, pixel_y}) - 11'sd60;
    assign is_ui_stone  = ((ui_dx * ui_dx) + (ui_dy * ui_dy) <= 22'd256);

    // =========================================================
    // Pixel colour output
    // =========================================================
    always_comb begin
        {vga_r, vga_g, vga_b} = 12'h000;

        if (video_on) begin
            if (is_in_menu) begin
                // --- Menu screen ---
                if (is_title_G || is_title_O) begin
                    {vga_r, vga_g, vga_b} = 12'hFD0;  // gold title
                end else if (is_pvp_border) begin
                    {vga_r, vga_g, vga_b} = (mode_sel == 1'b0) ? 12'h0F0 : 12'h555;
                end else if (is_pvp_box) begin
                    {vga_r, vga_g, vga_b} = is_pvp_left ? 12'h111 : 12'hFFF;
                end else if (is_pve_border) begin
                    {vga_r, vga_g, vga_b} = (mode_sel == 1'b1) ? 12'h0F0 : 12'h555;
                end else if (is_pve_box) begin
                    {vga_r, vga_g, vga_b} = 12'h111;
                end else begin
                    {vga_r, vga_g, vga_b} = 12'h223;  // dark blue background
                end

            end else begin
                // --- In-game screen ---
                if (is_ui_border) begin
                    {vga_r, vga_g, vga_b} = game_over ? 12'hFD0 : 12'h555;

                end else if (is_ui_box && is_ui_stone) begin
                    // Turn indicator stone colour (or winner indicator after game over)
                    if (game_over)
                        {vga_r, vga_g, vga_b} = (p1_score > p2_score) ? 12'h111 : 12'hFFF;
                    else
                        {vga_r, vga_g, vga_b} = (current_turn == 2'b01) ? 12'h111 : 12'hFFF;

                end else if (is_ui_box) begin
                    {vga_r, vga_g, vga_b} = game_over ? 12'hCA3 : 12'hCCC;

                end else if (is_cursor && !game_over) begin
                    {vga_r, vga_g, vga_b} = 12'hF00;

                end else if (is_stone) begin
                    {vga_r, vga_g, vga_b} = (cell_state == 2'b01) ? 12'h111 : 12'hFFF;

                end else if (is_territory_box) begin
                    if      (cell_territory == 2'b01) {vga_r, vga_g, vga_b} = 12'h111;
                    else if (cell_territory == 2'b10) {vga_r, vga_g, vga_b} = 12'hFFF;
                    else                              {vga_r, vga_g, vga_b} = 12'hDB7;

                end else if (is_grid_line) begin
                    {vga_r, vga_g, vga_b} = 12'h000;

                end else if (is_board_bg) begin
                    {vga_r, vga_g, vga_b} = 12'hDB7;  // wood tan

                end else begin
                    {vga_r, vga_g, vga_b} = 12'h888;  // grey border
                end
            end
        end
    end
endmodule
