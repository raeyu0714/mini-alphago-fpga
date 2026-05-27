`timescale 1ns / 1ps

// Weight and bias ROM modules for the Mini AlphaGo Zero CNN.
//
// Synthesis hints:
//   Large ROMs (>= 512 entries) use RAMB18 block RAM.
//   Small ROMs (< 512 entries)  use distributed (LUT) RAM.
//
// All $readmemb paths are relative to the project root.
// Update the path prefix if your Vivado project is in a different location.

// ------------------------------------------------------------
// Entry Conv weights  (8-ch input -> 64 output, 3x3)
// 8 * 64 * 9 = 4608 entries -> RAMB18
// ------------------------------------------------------------
// NOTE: Uncomment and update path when synthesising.
/*
module rom_entry_weight (
    input  logic        clka,
    input  logic [12:0] addra,
    output logic [7:0]  douta
);
    (* rom_style = "block" *) logic [7:0] mem [0:4607];
    initial $readmemb("weight/entry_0_weight.mem", mem);
    always_ff @(posedge clka) douta <= mem[addra];
endmodule
*/

// ------------------------------------------------------------
// Entry Conv bias  (64 entries) -> distributed
// ------------------------------------------------------------
module rom_entry_bias (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/entry_0_bias.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Tower 0 Conv1 bias  (64 entries) -> distributed
// ------------------------------------------------------------
module rom_tower0_conv1_bias (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/tower_0_net_0_bias.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Tower 0 Conv2 bias  (64 entries) -> distributed
// ------------------------------------------------------------
module rom_tower0_conv2_bias (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/tower_0_net_3_bias.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Tower 1 Conv1 bias  (64 entries) -> distributed
// ------------------------------------------------------------
module rom_tower1_conv1_bias (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/tower_1_net_0_bias.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Tower 1 Conv2 bias  (64 entries) -> distributed
// ------------------------------------------------------------
module rom_tower1_conv2_bias (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/tower_1_net_3_bias.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Policy Head Conv weights  (64->2, 1x1)
// 64 * 2 = 128 entries -> distributed
// ------------------------------------------------------------
module rom_policy_conv_weight (
    input  logic       clk,
    input  logic [6:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:127];
    initial $readmemb("weight/policy_head_0_weight.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Policy FC bias  (81 entries) -> distributed
// ------------------------------------------------------------
module rom_policy_fc_bias (
    input  logic       clk,
    input  logic [6:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:80];
    initial $readmemb("weight/policy_head_4_bias.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Value Head Conv weights  (64->1, 1x1)
// 64 * 1 = 64 entries -> distributed
// ------------------------------------------------------------
module rom_value_conv_weight (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/value_head_0_weight.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Value FC1 bias  (64 entries) -> distributed
// ------------------------------------------------------------
module rom_value_fc1_bias (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/value_head_4_bias.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule

// ------------------------------------------------------------
// Value FC2 weights  (64->1)
// 64 * 1 = 64 entries -> distributed
// ------------------------------------------------------------
module rom_value_fc2_weight (
    input  logic       clk,
    input  logic [5:0] addr,
    output logic [7:0] data
);
    (* rom_style = "distributed" *) logic [7:0] mem [0:63];
    initial $readmemb("weight/value_head_6_weight.mem", mem);
    always_ff @(posedge clk) data <= mem[addr];
endmodule
