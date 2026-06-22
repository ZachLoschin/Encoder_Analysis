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
base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R1_Aug';
subfolder = '';

% Get list of all subfolders in base_dir
session_dirs = dir(base_dir);
session_dirs = session_dirs([session_dirs.isdir]);  % Keep only directories
session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));  % Remove . and ..

for i = 1:length(session_dirs)

    nColors = 256;

    % Fully saturated red and blue
    pure_blue = [0, 0, 1];
    pure_red  = [1, 0, 0];
    white = [1, 1, 1];
    
    % Blend strength toward white (0 = fully pure color, 1 = white)
    blend_factor = 0.5;
    
    % Generate linearly spaced RGB values and blend each with white
    blue_side = (1 - blend_factor) * pure_blue + blend_factor * white;
    red_side  = (1 - blend_factor) * pure_red  + blend_factor * white;
    
    custom_cmap = [linspace(blue_side(1), red_side(1), nColors)', ...
                   linspace(blue_side(2), red_side(2), nColors)', ...
                   linspace(blue_side(3), red_side(3), nColors)'];

    blue = [0/255, 0/255, 153/255]; white = [255/255, 255/255, 255/255]; red = [255/255, 51/255, 0/255];
    numColors = 100; blue = [0/255, 0/255, 153/255]; white = [255/255, 255/255, 255/255];
    blueToWhite = [linspace(blue(1), white(1), numColors)', linspace(blue(2), white(2), numColors)', linspace(blue(3), white(3), numColors)'];


    session_name = session_dirs(i).name;
    save_dir = fullfile(base_dir, session_name, subfolder);
    alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R1';
    subfolder = '';

    if session_name=="Summary_Figs_R1"
        continue
    end

    % Load files
    R1_States   = readmatrix(fullfile(save_dir, 'R1_States_Reg.csv'));
    R1_Tongue   = readmatrix(fullfile(save_dir, 'R1_Tongue_Reg.csv'));
    PC          = readmatrix(fullfile(save_dir, 'R1_PC_R2_Reg.csv'));

    
    %% Plot PSTHs for visualization
    alt_session_dir = fullfile(alt_base_dir, session_name);
    prb = alt_session_dir(end)
    R1_Path = alt_session_dir + "\Probe" + prb + "_R1_Uncut.csv";
    FCs_R1 = readmatrix(fullfile(alt_session_dir, 'FCs_R1.csv')) - 100;
    % Read the matrices
    R1_Neural = readmatrix(R1_Path);

    [x, xx] = size(R1_Neural);

    % Reshape into trials
    % T x N_trials x N_neurons
    R1_Neural_Trials = reshape(R1_Neural, 600, [], xx);  % size: 600 x 95 x 54
    R1_Neural_Trials = permute(R1_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
    % R1_Neural_Trials = zscore_pregc(R1_Neural_Trials, 100);
    

   % Average across trials
    R1_PSTH = squeeze(mean(R1_Neural_Trials, 3));  % size: 600 x 54
    
    % Transpose for heatmap: neurons on Y, time on X
    imagesc(R1_PSTH');  % now size is xx x 600
    xlabel('Time (samples)');
    ylabel('Neuron');
    title('R1 PSTH (Trial-Averaged)');
    colorbar;
    
    % Optional: improve visualization
    set(gca, 'YDir', 'normal');
    
    
    %% -- Combine the R1 Datasets and Heatmaps -- %%
    % All_States = exp(R1_States);

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



    All_Tongue = R1_Tongue;
    %% Normalize the kinametic data
    All_Tongue = range_normalize_with_nans(All_Tongue);
    % All_Tongue(All_Tongue == 0) = NaN;
    
    R1_Tongue_norm = (R1_Tongue - nanmin(R1_Tongue, [], 2)) ./ (nanmax(R1_Tongue, [], 2) - nanmin(R1_Tongue, [], 2));
    R1_Tongue_norm(R1_Tongue_norm == 0) = NaN;
    
    %%
    % Initialize the figure
    
    R1_Tongue = R1_Tongue_norm';
    R1_States = R1_States';
    figure;
    hold on;
    
    % Plotting parameters
    lw = 0.75;  % Line width
    px = 75; py = 75;
    width = 700; height = 800;
    set(gcf, 'Position', [px, py, width, height]);
    
    R1_color = 'k'; % Black for R1
    
    % Overlay heatmap
    h = imagesc(1:size(All_States, 2), 1:size(All_States, 1), All_States);

    for j = 1:size(R1_Tongue, 2)
        plot(1:length(R1_Tongue(:, j)), j-1 + R1_Tongue(:, j), 'k', 'LineWidth', lw);
    end
    
     % Adjust the colormap to suit your data range
    colormap(custom_cmap);  % You can use 'jet', 'hot', 'cool', or any other colormap
    % colormap(linspecer);  % Flip the colormap if needed
    
    % Adjust the heatmap properties
    % set(h, 'AlphaData', 0.5); % Adjust transparency to see the line plots underneath
    
    % Add a colorbar
    cc = colorbar;

    % Set axes limits and labels
    set(gca, 'YTick', 0:10:260);
    xlabel('Time (s)');
    ylabel('Trial Number');
    
     % Adjust the x-axis ticks and labels to reflect time in seconds
    xticks([0 60 110 160 210 260 310 360 410 460 510]); % Position of ticks
    xticklabels({'-0.1', '0.5', '1', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0'}); % Labels corresponding to time in seconds
    
    nTrials = size(All_States, 1);
    ylim([0 nTrials]);  % Number of trials

    % Remove grid lines and tighten the axes
    box off;
    axis tight;
    
    % Customize ticks and labels to match your style
    set(gca, 'TickLength', [0 0]);
    xlim([0 210])
    % Hold off to finish the plot
    hold off;
    
    title("Single Trial State Estimates: R1");
    xline(11, '--k', 'LineWidth', 1);  % Add vertical line for GC
    text(11, -5, 'GC', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 12, ...
        'Color', 'k');
    
    saveas(gcf, fullfile(save_dir, 'Inference_Heatmap.png'));  % Save current figure as PNG
    saveas(gcf, fullfile(save_dir, 'Inference_Heatmap.fig'));  % Save as MATLAB .fig file
    
    
    %% PC prediction R2 values
    PC = PC(1, 1:10);
    bar(PC)
    title("Neural PC Encoding R^2")
    xlabel("Neural PC")
    ylabel("R^2")
    ylim([0,1])
    saveas(gcf, fullfile(save_dir, 'PC_Encoding_R2.png'));

    %% Trial averaged inference plots
    lb = [0 0.76 1];

    % Trial averaged inference plots
    R1_Inf_Mean = mean(exp(R1_States'), 1);
    R1_Inf_Std  = std(exp(R1_States'), 0, 1);
    

    x = 1:length(R1_Inf_Mean);
    
    figure
    hold on
    
    % Shaded std region for R1
    fill([x, fliplr(x)], ...
         [R1_Inf_Mean + R1_Inf_Std, fliplr(R1_Inf_Mean - R1_Inf_Std)], ...
         [0.6 0.8 1], ...       % light blue color
         'EdgeColor', 'none', ...
         'FaceAlpha', 0.7);
    
    % Plot mean traces
    plot(x, R1_Inf_Mean, 'b', 'LineWidth', 2)
    
    ylabel("State 1 Probability")
    xlabel("Time (s)")
    xticks([0 60 110 160 210]); % Position of ticks
    xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'}); % Labels corresponding to time in seconds
    xlim([0, 200])
    ylim([0, 1])
    title("Trial Averaged Inference")
    
    xline(11, "--k", "LineWidth", 1)
    legend(["R1 Std", "R1 Mean"])
    saveas(gcf, fullfile(save_dir, 'Ave_Inference.png'))
    saveas(gcf, fullfile(save_dir, 'Ave_Inference.fig'))

    %% Trial averaged inference aligned to FC

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
    
    x = 1:length(R1_Inf_Mean);
    
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
    
    legend(["R1 Std", "R1 Mean"])
    saveas(gcf, fullfile(save_dir, 'Ave_Inference_FCAligned.png'))
    saveas(gcf, fullfile(save_dir, 'Ave_Inference_FCAligned.fig'))

    % Compute disengagement times for R1
    R1_disengage = (estimate_disengage_times(exp(R1_States'), 0.5, 2, 20) - 10) * 10;
    csvwrite(fullfile(save_dir, "R1_dt.csv"), R1_disengage);
    
    % Convert back to trial indices for heatmap overlay
    R1_d_hm = R1_disengage / 10;
    
    % ---- Plot Heatmap ----
    figure; hold on;
    
    % Figure settings
    lw = 0.75;             % Line width for tongue traces
    set(gcf, 'Position', [75, 75, 700, 800]); % px, py, width, height
    
    % Overlay heatmap
    h = imagesc(1:size(All_States, 2), 1:size(All_States, 1), All_States);
    colormap(custom_cmap);
    cc = colorbar;
    
    % Plot tongue traces
    for j = 1:size(R1_Tongue, 2)
        plot(1:size(R1_Tongue,1), j-1 + R1_Tongue(:, j), 'k', 'LineWidth', lw);
    end
    
    % Plot disengagement time markers
    for i = 1:length(R1_d_hm)
        if ~isnan(R1_d_hm(i))
            plot(R1_d_hm(i)+10, i, 'wo', 'MarkerSize', 4, 'LineWidth', 1.2);
        end
    end
    
    % Axis settings
    set(gca, 'YTick', 0:10:260, 'TickLength', [0 0]);
    xlabel('Time (s)'); ylabel('Trial Number');
    
    xticks([0 60 110 160 210 260 310 360 410 460 510]);
    xticklabels({'-0.1', '0.5', '1', '1.5', '2.0', '2.5', ...
                 '3.0', '3.5', '4.0', '4.5', '5.0'});
    
    ylim([0 size(All_States,1)]);
    xlim([0 210]);
    box off; axis tight;
    
    % Title and reference line
    title("Single Trial State Estimates: R1");
    xline(11, '--k', 'LineWidth', 1);  
    text(11, -5, 'GC', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 12, 'Color', 'k');
    
    % Save figure
    saveas(gcf, fullfile(save_dir, 'R1_Inference_Heatmap_dt.png'));
    saveas(gcf, fullfile(save_dir, 'R1_Inference_Heatmap_dt.fig'));


    % %% NEW FIGURE %%
    % 
    % engaged_durations = [];
    % disengaged_durations = [];
    % engaged_licks = {};
    % disengaged_licks = {};
    % 
    % % Transpose to [timepoints x trials]
    % All_Tongue = All_Tongue';
    % All_States = All_States';
    % 
    % nTrials = size(All_Tongue, 2);
    % 
    % figure; hold on;
    % set(gcf, 'Position', [100, 100, 800, 800]);
    % 
    % % Loop through each trial
    % for trial = 1:nTrials
    %     tongue = All_Tongue(:, trial)';
    %     states = All_States(:, trial)';
    % 
    %     % Find lick bouts: stretches of non-NaN values
    %     is_valid = ~isnan(tongue);
    %     diff_valid = diff([0 is_valid 0]);
    %     lick_starts = find(diff_valid == 1);
    %     lick_ends   = find(diff_valid == -1) - 1;
    % 
    %     for i = 1:length(lick_starts)
    %         idx = lick_starts(i):lick_ends(i);
    % 
    %         % Check minimum length requirement
    %         if length(idx) < 2
    %             continue;  % Skip this bout if it's too short
    %         end
    % 
    %         bout = tongue(idx);
    %         bout_states = states(idx);
    % 
    %         % Color based on average state during the bout
    %         color = 'r';
    %         if mean(bout_states == 1) <= 0.5
    %             color = 'b';
    %         end
    % 
    %         % Save the duration
    %         duration = length(idx);
    % 
    %         % Classify based on average state
    %         if mean(bout_states == 1) >= 0.5
    %             engaged_durations(end+1) = duration;
    %             engaged_licks{end+1} = bout;
    %         else
    %             disengaged_durations(end+1) = duration;
    %             disengaged_licks{end+1} = bout;
    %         end
    % 
    % 
    %         plot(idx, (trial - 1) + bout, color, 'LineWidth', 1.2);
    %     end
    % end
    % 
    % xlabel('Time (s)');
    % ylabel('Trial #');
    % title('Engaged (Red) vs Disengaged (Blue) Licks');
    % xticks([0 100 200]);
    % % xticklabels({'0', '1', '2'});
    % % set(gca, 'TickLength', [0 0]);
    % % box off;
    % % ylim([0 nTrials + 1]);
    % 
    % saveas(gcf, fullfile(save_dir, 'Labeled_Licks.png'));  % Save current figure as PNG
    % saveas(gcf, fullfile(save_dir, 'Labeled_Licks.fig'));  % Save as MATLAB .fig file
    % 
    close all


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


