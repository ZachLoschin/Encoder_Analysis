%% Problem 1a

% Define the coefficient matrix
A = [1 3 -3 7; 0 1 -4 5];

% Define the vectors from the parametric vector form
u = [-9; 4; 1; 0];
v = [8; -5; 0; 1];

% Set the parameters (note that these can be any real numbers!)
s = 10;
t = -1/2;

% Calculate the solution vector from parametric vector form
x = s*u + t*v;

% Verify that A*x gives the zero vector. Note that we
% want the zero vector to appear because we solved the
% homogeneous system Ax=0
z = A*x

%% Problem 1b
A = [1 5 2 -6 9 0;
    0 0 1 -7 4 -8;
    0 0 0 0 0 1;
    0 0 0 0 0 0];

u = [-5; 1; 0; 0; 0; 0];
v = [-8; 0; 7; 1; 0; 0];
w = [-1; 0; -4; 0; 1; 0];

s = -5;
t = 11;
q = 1;

x = s*u + t*v + q*w;

z = A*x


%% Problem 3a
A = [1 3 1;
    -4 -9 2;
    0 -3 -6];

u = [-2; 1; 0];
v = [5; -2; 1];

s = 1;
t = -5;

x = s*u + t*v;

% Note that here, we get the vector [1, -1, -3]
% because that is the b vector of the
% nonhomogeneous system!
z = A*x


%% Problem 3b
A = [1 0 -2 -5;
    -1 1 3 7;
    3 -2 2 1];

u = [1; 2; 0; 0];
v = [1; 0; -2; 1];

s = 5;

% Note that u doesn't get multiplied by anything here
x = u + s*v;

z = A*x



