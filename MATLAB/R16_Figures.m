% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% January 2025
% Preprocessing file for disengagement analysis

% Heatmap creation for imported state labels and tongue kinematics from
% julia HMM-GLM analysis

% Read in the data, construct heatmap, overlay lick kinematics
clear;
clc;
close all
%% Import the state inference and tongue data

R1_States = readmatrix("C:\Research\Encoder_Modeling\Encoder_Analysis\Results_423\YH1_05_08\Jaw2PC\R1_States_Vit_Reg.csv");  % Load the R1_States matrix
R4_States = readmatrix("C:\Research\Encoder_Modeling\Encoder_Analysis\Results_423\YH1_05_08\Jaw2PC\R16_States_Vit_Reg.csv");  % Load the R4_States matrix
R1_Tongue = readmatrix("C:\Research\Encoder_Modeling\Encoder_Analysis\Results_423\YH1_05_08\Jaw2PC\R1_Tongue_Reg.csv");
R4_Tongue = readmatrix("C:\Research\Encoder_Modeling\Encoder_Analysis\Results_423\YH1_05_08\Jaw2PC\R16_Tongue_Reg.csv");

% close all
% R1_Tongue = R1_Tongue';
% R4_Tongue = R4_Tongue';

% %% Chop the data to relevant periods
% start = 400;
% stop = 800;
% 
% R1_States_chopped = R1_States(:, start:stop);
% R4_States_chopped = R4_States(:, start:stop);
% R1_Tongue = R1_Tongue(:, start:stop);
% R4_Tongue = R4_Tongue(:, start:stop);

%% -- Combine the R4 and R1 Datasets and Heatmaps -- %%
% All_States = exp([R4_States; R1_States]);
All_States = [R4_States; R1_States];
All_Tongue = [R4_Tongue; R1_Tongue];

%% Normalize the kinametic data
All_Tongue = range_normalize_with_nans(All_Tongue);
All_Tongue(All_Tongue == 0) = NaN;

% Normalize each row for R4_Tongue and R1_Tongue
R4_Tongue_norm = (R4_Tongue - nanmin(R4_Tongue, [], 2)) ./ (nanmax(R4_Tongue, [], 2) - nanmin(R4_Tongue, [], 2));
R4_Tongue_norm(R4_Tongue_norm == 0) = NaN;

R1_Tongue_norm = (R1_Tongue - nanmin(R1_Tongue, [], 2)) ./ (nanmax(R1_Tongue, [], 2) - nanmin(R1_Tongue, [], 2));
R1_Tongue_norm(R1_Tongue_norm == 0) = NaN;

%%
% Initialize the figure

R1_Tongue = R1_Tongue_norm';
R4_Tongue = R4_Tongue_norm';
R1_States = R1_States';
R4_States = R4_States';
% All_States = All_States';
figure;
hold on;

% Plotting parameters
lw = 0.75;  % Line width
px = 75; py = 75;
width = 700; height = 800;
set(gcf, 'Position', [px, py, width, height]);

% Define colors for the conditions
R4_color = 'k'; % Black for R4
R1_color = 'k'; % Black for R1

% Plotting line plots
for i = 1:size(R4_Tongue, 2)
    plot(1:length(R4_Tongue(:, i)), i-1 + R4_Tongue(:, i), R4_color, 'LineWidth', lw);
end

for i = 1:size(R1_Tongue, 2)
    plot(1:length(R1_Tongue(:, i)), i-1 + size(R4_Tongue, 2) + R1_Tongue(:, i), R1_color, 'LineWidth', lw);
end

% Overlay heatmap
h = imagesc(1:size(All_States, 2), 1:size(All_States, 1), All_States);

% Adjust the colormap to suit your data range
colormap("jet");  % You can use 'jet', 'hot', 'cool', or any other colormap
% colormap(flipud(linspecer));  % Flip the colormap if needed

% Adjust the heatmap properties
set(h, 'AlphaData', 0.5); % Adjust transparency to see the line plots underneath

% Add a colorbar
colorbar;

% Add a horizontal line to separate the conditions
yline(size(R4_Tongue, 2), 'k-', 'LineWidth', 3);

% Add labels or annotations
text(-30, size(R4_Tongue, 2) + 2, 'R1', 'FontSize', 12, 'Color', "k");
text(-30, size(R4_Tongue, 2) - 2, 'R16', 'FontSize', 12, 'Color', "k");

% Set axes limits and labels
set(gca, 'YTick', 0:10:260);
xlabel('Time (s)');
ylabel('Trial Number');

% Adjust the x-axis ticks and labels to reflect time in seconds
xticks([0 100 200 300 400 500]); % Position of ticks
xticklabels({'0', '1', '2', '3', '4', '5'}); % Labels corresponding to time in seconds


ylim([0 139]);  % Number of trials


% Remove grid lines and tighten the axes
box off;
axis tight;

% Customize ticks and labels to match your style
set(gca, 'TickLength', [0 0]);
xlim([0 200])
% Hold off to finish the plot
hold off;

title("Single Trial State Estimates: R1 and R16");


%% NEW FIGURE %%

engaged_durations = [];
disengaged_durations = [];
engaged_licks = {};
disengaged_licks = {};

% Transpose to [timepoints x trials]
All_Tongue = All_Tongue';
All_States = All_States';

nTrials = size(All_Tongue, 2);

figure; hold on;
set(gcf, 'Position', [100, 100, 800, 800]);

% Loop through each trial
for trial = 1:nTrials
    tongue = All_Tongue(:, trial)';
    states = All_States(:, trial)';
    
    % Find lick bouts: stretches of non-NaN values
    is_valid = ~isnan(tongue);
    diff_valid = diff([0 is_valid 0]);
    lick_starts = find(diff_valid == 1);
    lick_ends   = find(diff_valid == -1) - 1;
    
    for i = 1:length(lick_starts)
        idx = lick_starts(i):lick_ends(i);
        bout = tongue(idx);
        bout_states = states(idx);
        
        % Color based on average state during the bout
        color = 'r';
        if mean(bout_states == 1) <= 0.5
            color = 'b';
        end
        
        % Save the duration
        duration = length(idx);
        
        % Classify based on average state
        if mean(bout_states == 1) <= 0.5
            engaged_durations(end+1) = duration;
            engaged_licks{end+1} = bout;
        else
            disengaged_durations(end+1) = duration;
            disengaged_licks{end+1} = bout;
        end


        plot(idx, (trial - 1) + bout, color, 'LineWidth', 1.2);
    end
end

xlabel('Time (s)');
ylabel('Trial #');
title('Engaged (Red) vs Disengaged (Blue) Licks');
xticks([0 100 200]);
% xticklabels({'0', '1', '2'});
% set(gca, 'TickLength', [0 0]);
% box off;
% ylim([0 nTrials + 1]);

%% Boxplot of Lick Bout Durations
%% Boxplot of Lick Bout Durations
figure;
boxplot([engaged_durations'; disengaged_durations'], ...
        [repmat({'Engaged'}, length(engaged_durations), 1); ...
         repmat({'Disengaged'}, length(disengaged_durations), 1)]);
ylabel('Lick Duration (ms)');
title('Comparison of Engaged vs Disengaged Lick Durations');

% Convert y-axis ticks from frames (10ms each) to milliseconds
yt = yticks;
yticklabels(string(yt * 10));


%% Example plots of licks
% How many examples to show
nExamples = 10;

% Random indices (protect against too few licks)
nEngaged = min(nExamples, length(engaged_licks));
nDisengaged = min(nExamples, length(disengaged_licks));

% Plot engaged licks
figure;
for i = 1:nEngaged
    subplot(2, nExamples, i);
    plot(engaged_licks{i}, 'r', 'LineWidth', 1.5);
    title(['Engaged ' num2str(i)]);
    xlabel('Time'); ylabel('Lick amplitude');
end

% Plot disengaged licks
for i = 1:nDisengaged
    subplot(2, nExamples, nExamples + i);
    plot(disengaged_licks{i}, 'b', 'LineWidth', 1.5);
    title(['Disengaged ' num2str(i)]);
    xlabel('Time'); ylabel('Lick amplitude');
end

sgtitle('Example Engaged (Red) vs Disengaged (Blue) Licks');




function normalized_data = normalize_trials(data, method)
    % Replace NaNs with zeros
    data(isnan(data)) = 0;

    % Concatenate trials into a single vector
    concatenated = data(:);

    % Normalize based on the specified method
    switch lower(method)
        case 'zscore'
            % Z-score normalization
            normalized_concatenated = (concatenated - mean(concatenated)) / std(concatenated);
        case 'range'
            % Min-max normalization
            normalized_concatenated = normalize(concatenated, 'range');
        otherwise
            error("Invalid method. Choose 'zscore' or 'range'.");
    end

    % Reshape back to original dimensions
    normalized_data = reshape(normalized_concatenated, size(data));
end

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


