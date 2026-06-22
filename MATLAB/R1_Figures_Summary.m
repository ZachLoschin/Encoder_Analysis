% Read in the data, construct heatmap, overlay lick kinematics
clear;
clc;
close all
%% Import the state inference and tongue data
% base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_Aug_Learning\L2';
% summary_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_Aug_Learning\Summary_L2';
% alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\Learning';
base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R1_Aug';
summary_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R1_Aug\Summary_Figs_R1';
alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R1';

subfolder = '';

% Get list of all subfolders in base_dir
session_dirs = dir(base_dir);
session_dirs = session_dirs([session_dirs.isdir]);  % Keep only directories
session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));  % Remove . and ..

% Initialize storage for each session's data
R1_INF_MEAN = {};
R1_INF_MEAN_FC = {};
R1_D_HM = [];
avgDisengageTime = [];      % will be N_sessions x 1
sessionList      = {};      % will be N_sessions x 1

for ij = 1:length(session_dirs)
    session_name = session_dirs(ij).name;
    sessionList{end+1} = session_name;   % store the name
    save_dir = fullfile(base_dir, session_name, subfolder);
    
    % Look for the same folder name in the alternate base directory
    alt_session_dir = fullfile(alt_base_dir, session_name);



    if isfolder(alt_session_dir)
        fprintf('Found matching folder for %s in alt_base_dir.\n', session_name);
        % Now you can use alt_session_dir for further processing
    else
        warning('No matching folder for %s in alt_base_dir.', session_name);
        continue;
    end

    % Load files
    R1_States   = readmatrix(fullfile(save_dir, 'R1_States_Reg.csv'));
    R1_Tongue   = readmatrix(fullfile(save_dir, 'R1_Tongue_Reg.csv'));
    PC          = readmatrix(fullfile(save_dir, 'R1_PC_R2_Reg.csv'));
    % R1_Trial_Track = readmatrix(fullfile(alt_session_dir, 'R1_Trial_Track.csv'));
    % R4_Trial_Track = readmatrix(fullfile(alt_session_dir, 'R4_Trial_Track.csv'));
    
    % right after you load R1_States:
    R1_States = readmatrix(fullfile(save_dir,'R1_States_Reg.csv'));
    
    % list of sessions whose 0/1 you want to invert
    flip_sessions = {
      'TD26d_2025_07_30_P1'
      'TD26d_2025_08_05_P1'
      'TD26d_2025_08_06_P1'
      'TD26d_2025_08_07_P1'
      'TD27d_2025_07_25_P2'
      'TD9si_2024_07_08_P1'
      'TD9si_2024_07_10_P1'
    };
    
    % check and flip
    if ismember(session_name, flip_sessions)
      fprintf(' Flipping states for %s\n', session_name);
      % first compute probabilities
      p = exp(R1_States);        % p is in [0,1]
      % invert them
      p = 1 - p;                 % now inverted
    else
      p = exp(R1_States);
    end
    
    R1_States = log(p);
    % now use p in place of exp(R1_States)
    All_States = p;               % size nTrials x T, values in [0,1]

    %% Neural Probe Data Importation and Chopping
    prb = alt_session_dir(end);
    
    R1_Path = alt_session_dir + "\Probe" + prb + "_R1_Uncut.csv";


    % Read the matrices
    R1_Neural = readmatrix(R1_Path);


    [x, xx] = size(R1_Neural);

    % Reshape into trials
    % T x N_trials x N_neurons
    R1_Neural_Trials = reshape(R1_Neural, 600, [], xx);  % size: 600 x 95 x 54
    R1_Neural_Trials = permute(R1_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
    % R1_Neural_Trials = zscore_pregc(R1_Neural_Trials, 100);
    
 
    
    %% -- Combine the R4 and R1 Datasets and Heatmaps -- %%
    All_States = exp([R1_States]);

    % All_States = [R4_States; R1_States];
    All_Tongue = [R1_Tongue];
    
    %% Normalize the kinametic data
    All_Tongue = range_normalize_with_nans(All_Tongue);
    % All_Tongue(All_Tongue == 0) = NaN;
    
    R1_Tongue_norm = (R1_Tongue - nanmin(R1_Tongue, [], 2)) ./ (nanmax(R1_Tongue, [], 2) - nanmin(R1_Tongue, [], 2));
    R1_Tongue_norm(R1_Tongue_norm == 0) = NaN;
    
    %%      
    R1_Tongue = R1_Tongue_norm';
    R1_States = R1_States';  



     %% Trial averaged inference plots
    % Trial averaged inference plots
    R1_Inf_Mean = mean(exp(R1_States'), 1);
    R1_Inf_Std  = std(exp(R1_States'), 0, 1);

    R1_INF_MEAN{end+1} = R1_Inf_Mean;


    x = 1:length(R1_Inf_Mean)



    %% Aligned to FC
    FCs_R1 = readmatrix(fullfile(alt_session_dir, 'FCs_R1.csv')) - 100;
    % Parameters
    window_size = 211;
    pre_points = 10;
    post_points = 200;
    
    % ---- R1 ----
    n_trials = size(R1_States, 2);
    R1_FC_aligned = nan(window_size, n_trials);

     for i = 1:n_trials
        align_point = round(FCs_R1(i));
        start_idx = align_point - pre_points;
        end_idx = align_point + post_points;
    
        % Determine valid indices
        valid_start = max(start_idx, 1);
        valid_end = min(end_idx, size(R1_States, 1));
    
        % Corresponding indices in the aligned window
        insert_start = valid_start - start_idx + 1;
        insert_end = insert_start + (valid_end - valid_start);
    
        % Fill with actual data
        R1_FC_aligned(insert_start:insert_end, i) = R1_States(valid_start:valid_end, i);
     end



     % Trial averaged inference plots
    R1_Inf_Mean = nanmean(exp(R1_FC_aligned'), 1);
    R1_Inf_Std  = nanstd(exp(R1_FC_aligned'), 0, 1);

    % Replace NaNs in mean and std with 0.0
    R1_Inf_Mean(isnan(R1_Inf_Mean)) = 0.0;
    R1_Inf_Std(isnan(R1_Inf_Std)) = 0.0;

    R1_INF_MEAN_FC{end+1} = R1_Inf_Mean;


    R1_disengage = (estimate_disengage_times(exp(R1_States'), 0.5, 2, 20) - 10)*10;


    R1_d_hm = R1_disengage / 10;

    R1_D_HM = [R1_D_HM; R1_d_hm];

    avgDisengageTime(end+1) = nanmean(R1_disengage);


    %% Align the neural activity to the disengagement times
    R1_d_hm = R1_disengage / 10;
    
    R1_Neural_Trials;  % 600 x 106 x 38

    n_trials = size(R1_Neural_Trials, 2);
    n_neurons = size(R1_Neural_Trials, 3);
    window_size = 101;
    pre_points = 50;
    post_points = 50;
    
    R1_d_aligned = nan(window_size, n_trials, n_neurons);
    
    for i = 1:n_trials
        align_point = round(R1_d_hm(i)) + 100;  % add offset
        start_idx = align_point - pre_points;
        end_idx = align_point + post_points;
    
        % Check bounds
        if start_idx >= 1 && end_idx <= size(R1_Neural_Trials, 1)
            R1_d_aligned(:, i, :) = R1_Neural_Trials(start_idx:end_idx, i, :);
        else
            warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
        end
    end



    engaged_durations = [];
    disengaged_durations = [];
    engaged_licks = {};
    disengaged_licks = {};
    
    % Transpose to [timepoints x trials]
    All_Tongue = All_Tongue';
    All_States = All_States';
    
    nTrials = size(All_Tongue, 2);


    engaged_ms = (engaged_durations' *10);
    disengaged_ms = (disengaged_durations' *10);
    close all; 
end




%% Session Wide Average Inference
disp("Plotting Started")

% Stack into matrices: [num_sessions x timepoints]
R1_all = cat(1, R1_INF_MEAN{:});  % [num_sessions x T]


% Compute mean and std across sessions
R1_Inf_Mean = mean(R1_all, 1);  % [1 x T]
R1_Inf_Std  = std(R1_all, 0, 1);



x = 1:length(R1_Inf_Mean);  % time axis, adapt if needed

figure;
hold on

% Shaded std region for R1
fill([x, fliplr(x)], ...
     [R1_Inf_Mean + R1_Inf_Std, fliplr(R1_Inf_Mean - R1_Inf_Std)], ...
     [0.6 0.8 1], ...       % light blue color
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);

% Plot mean traces
plot(x, R1_Inf_Mean, 'b', 'LineWidth', 2);

ylabel("State 1 Probability");
xlabel("Time (s)");
xticks([0 60 110 160 210]);
xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'});
xlim([0, 200]);
title("Trial Averaged Inference");

xline(11, "--k", "LineWidth", 1);


legend(["R1 Std", "R1 Mean"]);

saveas(gcf, fullfile(summary_dir, 'Ave_Inference.png'));
saveas(gcf, fullfile(summary_dir, 'Ave_Inference.fig'));


%% Trial averaged inference aligned to FC

% Stack into matrices: [num_sessions x timepoints]
R1_all = cat(1, R1_INF_MEAN_FC{:});  % [num_sessions x T]

% Compute mean and std across sessions
R1_Inf_Mean = nanmean(R1_all, 1);  % [1 x T]
R1_Inf_Std  = nanstd(R1_all, 0, 1);


x = 1:length(R1_Inf_Mean);  % time axis, adapt if needed

figure
hold on

% Clamp shaded areas between 0 and 1
R1_Shade_Upper = min(R1_Inf_Mean + R1_Inf_Std, 1);
R1_Shade_Lower = max(R1_Inf_Mean - R1_Inf_Std, 0);




% Shaded std region for R1
r1_s = fill([x, fliplr(x)], ...
     [R1_Shade_Upper, fliplr(R1_Shade_Lower)], ...
     [0.6 0.8 1], ...
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);



% Plot mean traces
r1_p = plot(x, R1_Inf_Mean, 'b', 'LineWidth', 2);

ylabel("State 1 Probability")
xlabel("Time (s)")
xticks([0 60 110 160 210]); % Position of ticks
xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'}); % Labels corresponding to time in seconds
xlim([0, 200])
ylim([0, 1])
title("Trial Averaged Inference")

xline(11, "--k", "LineWidth", 1)

legend([r1_p, r1_s], {'R1 Mean', 'R1 Std'} )
saveas(gcf, fullfile(summary_dir, 'Ave_Inference_FCAligned.png'))
saveas(gcf, fullfile(summary_dir, 'Ave_Inference_FCAligned.fig'))

%% Session Wide Dt Histogram
edges = 0:50:2000; % bin in 10-frame increments
figure
histogram(R1_D_HM*10, edges, 'FaceColor', 'blue', 'FaceAlpha', 0.5)
hold on
xlabel('Disengagement Time (ms)')
ylabel('Trial Count')
legend('R1')
title('Disengagement Time Distribution')
xticks(0:250:2000);                          % Tick positions in ms
xticklabels(string((0:250:2000)/1000));      % Convert to seconds as labels
saveas(gcf, fullfile(summary_dir, 'Dt_Histogram.png'))
saveas(gcf, fullfile(summary_dir, 'Dt_Histogram.fig'))

%% Session Wide Dt boxplot
disengage_times = [R1_D_HM*10];

% Create matching group labels
group_labels = [repmat({'R1'}, length(R1_D_HM), 1)];

% Make the box plot
figure
boxplot(disengage_times, group_labels)
ylabel('Disengagement Time (s)')  % Units: seconds
yticks(0:250:2000);                          % Tick positions in ms
yticklabels(string((0:250:2000)/1000));
title('Disengagement Time Distribution')

saveas(gcf, fullfile(summary_dir, 'Dt_Boxplot.png'))
saveas(gcf, fullfile(summary_dir, 'Dt_Boxplot.fig'))


%% Plots for illustratos
% Disengagement times in ms
R1_ms = R1_D_HM * 10;
edges = 0:100:2000;
xticks_ms = -250:250:2000;
xticklabels_sec = string(xticks_ms / 1000);

%% 1. R1 Boxplot
figure;
boxplot(R1_ms, 'Orientation', 'vertical', 'Colors', 'b', 'Symbol', '');
ylim([-250 2000])
yticks(xticks_ms)
yticklabels(xticklabels_sec)
ylabel('Disengagement Time (s)')
title('R1 Boxplot')
set(gca, 'XTickLabel', {})  % Hide y-axis labels for Illustrator
saveas(gcf, fullfile(summary_dir, 'R1_Dt_Boxplot.png'))
saveas(gcf, fullfile(summary_dir, 'R1_Dt_Boxplot.fig'))

%% 3. R1 Histogram
% Scale factor
scale_factor = 1;

% Histogram binning
[counts, centers] = histcounts(R1_ms, edges);
centers = edges(1:end-1) + diff(edges)/2;

% Scale the bar heights
scaled_counts = counts / scale_factor;

% Plot using bar
figure;
barh(centers, scaled_counts, 1, 'FaceColor', 'b', 'FaceAlpha', 0.6)
ylim([-250 2000])
yticks(-250:250:2000)
yticklabels(string((-250:250:2000)/1000))
ylabel('Disengagement Time (s)')
xlabel(['Scaled Trial Count (/ ' num2str(scale_factor) ')'])
title('R1 Histogram (Scaled)')
saveas(gcf, fullfile(summary_dir, 'R1_Dt_Histogram_Scaled.png'))
saveas(gcf, fullfile(summary_dir, 'R1_Dt_Histogram_Scaled.fig'))


cleanLabels = strrep(sessionList, '_', ' ');

% After the loop: plot average disengagement time across sessions
figure;
plot(avgDisengageTime, '-o', 'LineWidth', 2);
grid on;
xlabel('Session (in chronological order)');
ylabel('Average disengagement time (ms)');
title('Mean disengagement time per session');
xticks(1:numel(avgDisengageTime));
xticklabels(cleanLabels);
xtickangle(45);
saveas(gcf, fullfile(summary_dir, 'Learning_dt.png'))
saveas(gcf, fullfile(summary_dir, 'Learning_dt.fig'))


%%
function zscored_data = zscore_pregc(data, pre_gc_points)
    % data: T x N_trials x N_neurons
    % pre_gc_points: number of timepoints before GC, e.g., 100
    
    % Extract pre-GC time window
    data_pregc = data(1:pre_gc_points, :, :);  % size: pre_gc_points x N_trials x N_neurons

    % Reshape to [pre_gc_points * N_trials, N_neurons]
    [~, N_trials, N_neurons] = size(data_pregc);
    concatenated_data = reshape(data_pregc, [], N_neurons);  % size: (pre_gc_points*N_trials) x N_neurons

    % Compute mean and std per neuron across all timepoints and trials
    neuron_means = mean(concatenated_data, 1);  % size: 1 x N_neurons
    neuron_stds = std(concatenated_data, 0, 1);  % size: 1 x N_neurons

    % Z-score the full data
    % broadcasting over time and trials
    zscored_data = (data - reshape(neuron_means, 1, 1, [])) ./ ...
                   reshape(neuron_stds, 1, 1, []);

    % Trim to post-preGC window
    zscored_data = zscored_data(pre_gc_points - 100 + 1:end, :, :);
end


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


