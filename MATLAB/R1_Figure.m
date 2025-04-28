% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% January 2025
% Preprocessing file for disengagement analysis

% Heatmap creation for imported state labels and tongue kinematics from
% julia HMM-GLM analysis (R1 only)

% Read in the data, construct heatmap, overlay lick kinematics
clear;
clc;
close all

%% Import the state inference and tongue data
R1_States = readmatrix("C:\Research\Encoder_Modeling\Encoder_Analysis\Results_423\TD1d_02_22\Jaw2PC\R1_States_Reg.csv");  % Load the R1_States matrix
R1_Tongue = readmatrix("C:\Research\Encoder_Modeling\Encoder_Analysis\Results_423\TD1d_02_22\Jaw2PC\R1_Tongue_Reg.csv");

%% Normalize the kinematic data
R1_Tongue_norm = (R1_Tongue - nanmin(R1_Tongue, [], 2)) ./ (nanmax(R1_Tongue, [], 2) - nanmin(R1_Tongue, [], 2));
R1_Tongue_norm(R1_Tongue_norm == 0) = NaN;

%% Transpose for plotting
R1_Tongue = R1_Tongue_norm';
R1_States = R1_States;

%% Initialize the figure
figure;
hold on;

% Plotting parameters
lw = 0.75;  % Line width
px = 75; py = 75;
width = 700; height = 800;
set(gcf, 'Position', [px, py, width, height]);

% Plot the tongue traces
for i = 1:size(R1_Tongue, 2)
    plot(1:length(R1_Tongue(:, i)), i-1 + R1_Tongue(:, i), 'k', 'LineWidth', lw);
end

% Overlay heatmap
h = imagesc(1:size(R1_States, 2), 1:size(R1_States, 1), R1_States);

% Adjust the colormap
colormap("jet");

% Adjust the heatmap transparency
set(h, 'AlphaData', 0.5);

% Add a colorbar
colorbar;

% Set axes limits and labels
set(gca, 'YTick', 0:10:260);
xlabel('Time (s)');
ylabel('Trial Number');

% Adjust the x-axis ticks and labels to reflect time in seconds
xticks([0 100 200 300 400 500]);
xticklabels({'0', '1', '2', '3', '4', '5'});

% Set y-limits
ylim([0 size(R1_Tongue, 2)+5]);  % a little padding

% Remove grid lines and tighten the axes
box off;
axis tight;

% Customize ticks and labels to match your style
set(gca, 'TickLength', [0 0]);
xlim([0 200])

% Title
title("Single Trial State Estimates: R1");

hold off;

%% Helper functions
function All_Tongue_norm = range_normalize_with_nans(All_Tongue)
    [nRows, nCols] = size(All_Tongue);  % Get the number of rows and columns
    All_Tongue_norm = NaN(nRows, nCols);  % Initialize the output matrix with NaNs
    
    % Loop through each row to perform range normalization
    for i = 1:nRows
        row = All_Tongue(i, :);  % Get the i-th row
        
        % Get the non-NaN elements and compute the min and max
        valid_elements = row(~isnan(row));
        if numel(valid_elements) > 1  % Ensure there are more than one valid element
            row_min = min(valid_elements);
            row_max = max(valid_elements);
            
            % Apply range normalization (min-max normalization)
            All_Tongue_norm(i, :) = (row - row_min) / (row_max - row_min);
        end
    end
end
