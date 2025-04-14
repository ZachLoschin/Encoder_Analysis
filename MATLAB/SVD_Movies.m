%% Post-SVD Reconstruction with Cropped Mean Frame

% Parameters
num_components = 100;  % Use as many components as needed

% Extract the first 'num_components' from each matrix
U = movMask_0(:, 1:num_components);      % [pixels x components]
V = movSVD_0(:, 1:num_components);       % [time x components]
S = diag(movSv(1:num_components));       % [components x components]

% Reconstruct motion energy (space x time)
reconstruction = U * S * V';              % [pixels x time]

% Get the spatial dimensions for reshaping
[Lybin, Lxbin] = deal(size(movMask_reshape_0, 1), size(movMask_reshape_0, 2));  % Get spatial dims

% Add back the mean value of each pixel
% reconstruction = reconstruction + avgframe_0(:);

% Preallocate the movie array
movie_array = zeros(Lybin, Lxbin, size(reconstruction, 2));  % [Lybin x Lxbin x time]

% Fill the movie array with the reconstructed frames
for t = 1:size(reconstruction, 2)
    movie_array(:, :, t) = reshape(reconstruction(:, t), Lxbin, Lybin)';  % Transposed reshape
end

implay(movie_array, 100)


%% Play the movie svd with color
% Normalize the movie
minVal = min(movie_array(:));
maxVal = max(movie_array(:));
norm_movie = (movie_array - minVal) / (maxVal - minVal);  % [0, 1]

% Choose colormap
cmap = jet(256);  % Or whatever you like

% Preallocate RGB movie
rgbMovie = zeros(size(movie_array, 1), size(movie_array, 2), 3, size(movie_array, 3));

% Apply colormap to each frame
for t = 1:size(movie_array, 3)
    frame = norm_movie(:, :, t);
    frame_rgb = ind2rgb(gray2ind(frame, 256), cmap);
    rgbMovie(:, :, :, t) = frame_rgb;
end

% Play RGB movie
implay(rgbMovie, 100);


%% Create the motion svd

% Parameters
num_components = 500;  % Use as many components as needed

% Extract the first 'num_components' from each matrix
U = motMask_0(:, 1:num_components);      % [pixels x components]
V = motSVD_0(:, 1:num_components);       % [time x components]
S = diag(motSv(1:num_components));       % [components x components]

% Reconstruct motion energy (space x time)
reconstruction = U * S * V';              % [pixels x time]

% Get the spatial dimensions for reshaping
[Lybin, Lxbin] = deal(size(movMask_reshape_0, 1), size(movMask_reshape_0, 2));  % Get spatial dims

% Add back the mean value of each pixel
reconstruction = reconstruction + avgframe_0(:);

% Preallocate the movie array
movie_array = zeros(Lybin, Lxbin, size(reconstruction, 2));  % [Lybin x Lxbin x time]

% Fill the movie array with the reconstructed frames
for t = 1:size(reconstruction, 2)
    movie_array(:, :, t) = reshape(reconstruction(:, t), Lxbin, Lybin)';  % Transposed reshape
end

implay(movie_array, 100)


%% play the motion SVD video with color

% Normalize the movie
minVal = min(movie_array(:));
maxVal = max(movie_array(:));
norm_movie = (movie_array - minVal) / (maxVal - minVal);  % [0, 1]

% Choose colormap
cmap = jet(256);  % Or whatever you like

% Preallocate RGB movie
rgbMovie = zeros(size(movie_array, 1), size(movie_array, 2), 3, size(movie_array, 3));

% Apply colormap to each frame
for t = 1:size(movie_array, 3)
    frame = norm_movie(:, :, t);
    frame_rgb = ind2rgb(gray2ind(frame, 256), cmap);
    rgbMovie(:, :, :, t) = frame_rgb;
end

% Play RGB movie
implay(rgbMovie, 100);


