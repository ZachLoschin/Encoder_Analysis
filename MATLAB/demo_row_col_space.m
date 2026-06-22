% demo_row_col_space.m
% Illustrate that row-ops preserve row-space but change column-space

clear; close all; clc

% 1) Define a simple tall matrix A (3×2, rank 2)
A = [1 2;
     3 4;
     5 6];

% 2) Row-reduce to RREF and get pivot-column indices
[R, pivCols] = rref(A);

% 3) Basis for Col(A): the pivot columns of the *original* A
Corig = A(:, pivCols);

%    Basis for Col(R): the pivot columns of R (shows how Col(A) is moved)
Cnew  = R(:, pivCols);

% 4) Basis for Row(A): the nonzero rows of R (since row-ops preserve row-space)
rowMask   = any(abs(R) > 1e-10, 2);  % which rows of R are nonzero
RowBasis  = R(rowMask, :);

%% 5) PLOTS

figure('Position',[100 100 900 600])

% 5a) Original columns of A in R^3
subplot(2,2,1)
quiver3(0,0,0, A(1,1),A(2,1),A(3,1), 'b','LineWidth',2); hold on
quiver3(0,0,0, A(1,2),A(2,2),A(3,2), 'b','LineWidth',2)
title('Original columns of A')
xlabel('x'); ylabel('y'); zlabel('z')
axis equal; grid on

% 5b) Columns of R = rref(A) in R^3
subplot(2,2,2)
quiver3(0,0,0, R(1,1),R(2,1),R(3,1), 'r','LineWidth',2); hold on
quiver3(0,0,0, R(1,2),R(2,2),R(3,2), 'r','LineWidth',2)
title('Columns of R = rref(A)')
xlabel('x'); ylabel('y'); zlabel('z')
axis equal; grid on

% 5c) Original rows of A in R^2
subplot(2,2,3)
hold on
for i=1:size(A,1)
    quiver(0,0, A(i,1),A(i,2), 'b','LineWidth',1.5)
end
title('Original rows of A (in R^2)')
xlabel('x_1'); ylabel('x_2')
axis equal; grid on

% 5d) Nonzero rows of R (basis of row-space)
subplot(2,2,4)
hold on
for i=1:size(RowBasis,1)
    quiver(0,0, RowBasis(i,1),RowBasis(i,2), 'r','LineWidth',2)
end
title('Nonzero rows of R (= basis of row-space)')
xlabel('x_1'); ylabel('x_2')
axis equal; grid on

%% 6) Display the chosen bases
disp('Pivot columns of A (basis for Col(A)):') 
disp(Corig)

disp('Nonzero rows of R (basis for Row(A)):')
disp(RowBasis)

%%
disp("Note that the columns of R do not span the same space as the columns of A, " + ...
     "so they cannot serve as a basis for Col(A).  Row reduction preserves only " + ...
     "the *indices* of the pivot columns, not the actual column subspace.")

disp("In contrast, row operations preserve the entire row‐space.  The nonzero rows " + ...
     "of RREF(A) form a convenient basis for Row(A), which is why we read the row-space " + ...
     "basis directly off the reduced matrix.")