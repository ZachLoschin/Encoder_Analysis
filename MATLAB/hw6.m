% hw06_q4.m
% MATLAB script for HW06 Question 4
clear; clc;

%% Define the given vectors (all 8×1)
a1 = [1; 1; 1; 1; 0; 0; 0; 0];
a2 = [1; 0; 1; 0; 1; 0; 1; 0];
a3 = [1; 1; 0; 0; 1; 1; 0; 0];

% We assume b2 and b3 have trailing zeros to make them 8×1
b1 = [1; 2;  0;  1;  0;  1; -1; 0];
b2 = [0; 0;  1; 1; -1;  -1;  0; 0];
b3 = [1; 0;  1;  0;  1;  0;  1; 0];

%%
A = [a1, a2, a3];
rank(A)
B = [b1, b2, b3]
results = rref([A B])

%%
dot(b1, b2)
dot(b2, b3)
dot(b1, b3)
dot(a1, a2)

%%
xA1 = 5
xA2 = 2
xA3 = 1

x_A = [xA1, xA2, xA3]'
v = A*x_A
x_B = linsolve(B, v)
B*x_B - A*x_A



%%
x1b = dot(b1, v)
x2b = dot(b2, v)
x3b = dot(b3,v)

x_B_1 = dot(v, b1) / dot(b1, b1)
x_B_2 = dot(v, b2) / dot(b2, b2)
x_B_3 = dot(v, b3) / dot(b3, b3)

x_B - [x_B_1, x_B_2, x_B_3]'








