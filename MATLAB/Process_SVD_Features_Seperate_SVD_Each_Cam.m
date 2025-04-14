%% Post-SVD Reconstruction with Cropped Mean Frame


%% Static Frame Movie SVD Features: 100 PCs
num_components = 10;

U = movMask_0(:, 1:num_components);      % [pixels x components]
V = movSVD_0(:, 1:num_components);       % [time x components]
S = diag(movSv(1:num_components));       % [components x components]

movie_features = V;              % [time x components]

%% Motion SVD Features: 100 PCs
num_components = 10;

U = motMask_0(:, 1:num_components);      % [pixels x components]
V = motSVD_0(:, 1:num_components);       % [time x components]
S = diag(motSv(1:num_components));       % [components x components]

motion_features = V;              % [time x components]

%% Combine the features and save them
SVD_features = [movie_features, motion_features];

save("SVD_Features_Cam0_TD13d_2024_11_12.mat", 'SVD_features')
