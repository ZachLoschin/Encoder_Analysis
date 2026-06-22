% Construct magic matrix
n = 5;
A = magic(n)

% Verify proper creation
magic_number = n*(n*n+1)/2
row_sum = sum(A(:,1))
col_sum = sum(A(1,:))
diag_sum = sum(diag(A))

% Eig
[V,D] = eig(A)

x = V(:,1)
val1 = A*x
val2 = D(1,1)*x

x_b = [1;1;1;1;1];
A*x_b

%% Power method section
n=55



