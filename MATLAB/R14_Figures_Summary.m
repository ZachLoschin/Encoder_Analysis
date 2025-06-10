% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% 2025
% Preprocessing file for disengagement analysis

% Figure creation: Single trial inference, trial averaged inference, PC
% encoding accuracy, state-labeled licks.

% Read in the data, construct heatmap, overlay lick kinematics
clear;
clc;
close all
%% Import the state inference and tongue data
base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R14_UpdatedSigma';
summary_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R14_UpdatedSigma\Summary_Figs';
alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R14_529_4thlick';
subfolder = '';

% Get list of all subfolders in base_dir
session_dirs = dir(base_dir);
session_dirs = session_dirs([session_dirs.isdir]);  % Keep only directories
session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));  % Remove . and ..

% Initialize storage for each session's data
R1_INF_MEAN = {};
R4_INF_MEAN = {};
R1_D_HM = [];
R4_D_HM = [];


for ij = 1:length(session_dirs)
    try
        session_name = session_dirs(ij).name;
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
        R4_States   = readmatrix(fullfile(save_dir, 'R14_States_Reg.csv'));
        R1_Tongue   = readmatrix(fullfile(save_dir, 'R1_Tongue_Reg.csv'));
        R4_Tongue   = readmatrix(fullfile(save_dir, 'R14_Tongue_Reg.csv'));
        PC          = readmatrix(fullfile(save_dir, 'R14_PC_R2_Reg.csv'));
        % R1_Trial_Track = readmatrix(fullfile(alt_session_dir, 'R1_Trial_Track.csv'));
        % R4_Trial_Track = readmatrix(fullfile(alt_session_dir, 'R4_Trial_Track.csv'));
        
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
    
        %% Neural Probe Data Importation and Chopping
        prb = alt_session_dir(end);
        
        R1_Path = alt_session_dir + "\Probe" + prb + "_R1_Uncut.csv";
        R4_Path = alt_session_dir + "\Probe" + prb + "_R4_Uncut.csv";
    
        % Read the matrices
        R1_Neural = readmatrix(R1_Path);
        R4_Neural = readmatrix(R4_Path);
    
        [x, xx] = size(R1_Neural);
    
        % Reshape into trials
        % T x N_trials x N_neurons
        R1_Neural_Trials = reshape(R1_Neural, 600, [], xx);  % size: 600 x 95 x 54
        R1_Neural_Trials = permute(R1_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
        % R1_Neural_Trials = zscore_pregc(R1_Neural_Trials, 100);
        
        R4_Neural_Trials = reshape(R4_Neural, 600, [], xx);  % size: 600 x 95 x 54
        R4_Neural_Trials = permute(R4_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
        % R4_Neural_Trials = zscore_pregc(R4_Neural_Trials, 100);    
        
        %% -- Combine the R4 and R1 Datasets and Heatmaps -- %%
        All_States = exp([R4_States; R1_States]);
        % All_States = [R4_States; R1_States];
        All_Tongue = [R4_Tongue; R1_Tongue];
        
        %% Normalize the kinametic data
        All_Tongue = range_normalize_with_nans(All_Tongue);
        % All_Tongue(All_Tongue == 0) = NaN;
        
        % Normalize each row for R4_Tongue and R1_Tongue
        R4_Tongue_norm = (R4_Tongue - nanmin(R4_Tongue, [], 2)) ./ (nanmax(R4_Tongue, [], 2) - nanmin(R4_Tongue, [], 2));
        R4_Tongue_norm(R4_Tongue_norm == 0) = NaN;
        
        R1_Tongue_norm = (R1_Tongue - nanmin(R1_Tongue, [], 2)) ./ (nanmax(R1_Tongue, [], 2) - nanmin(R1_Tongue, [], 2));
        R1_Tongue_norm(R1_Tongue_norm == 0) = NaN;
        
        %%      
        R1_Tongue = R1_Tongue_norm';
        R4_Tongue = R4_Tongue_norm';
        R1_States = R1_States';
        R4_States = R4_States';    
        
        %% Trial averaged inference plots
        % Trial averaged inference plots
        R1_Inf_Mean = mean(exp(R1_States'), 1);
        R1_Inf_Std  = std(exp(R1_States'), 0, 1);
        
        R4_Inf_Mean = mean(exp(R4_States'), 1);
        R4_Inf_Std  = std(exp(R4_States'), 0, 1);

        R1_INF_MEAN{end+1} = R1_Inf_Mean;
        R4_INF_MEAN{end+1} = R4_Inf_Mean;


        x = 1:length(R1_Inf_Mean);
        
        % figure
        % hold on
        % 
        % % Shaded std region for R1
        % fill([x, fliplr(x)], ...
        %      [R1_Inf_Mean + R1_Inf_Std, fliplr(R1_Inf_Mean - R1_Inf_Std)], ...
        %      [0.8 0.8 1], ...       % light blue color
        %      'EdgeColor', 'none', ...
        %      'FaceAlpha', 0.4);
        % 
        % % Shaded std region for R4
        % fill([x, fliplr(x)], ...
        %      [R4_Inf_Mean + R4_Inf_Std, fliplr(R4_Inf_Mean - R4_Inf_Std)], ...
        %      [1 0.8 0.8], ...       % light red/pink color
        %      'EdgeColor', 'none', ...
        %      'FaceAlpha', 0.7);
        % 
        % % Plot mean traces
        % plot(x, R1_Inf_Mean, 'b', 'LineWidth', 2)
        % plot(x, R4_Inf_Mean, 'r', 'LineWidth', 2)
        % 
        % ylabel("State 1 Probability")
        % xlabel("Time (s)")
        % xticks([0 60 110 160 210]); % Position of ticks
        % xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'}); % Labels corresponding to time in seconds
        % xlim([0, 200])
        % title("Trial Averaged Inference")
        % 
        % xline(11, "--k", "LineWidth", 1)
        % text(11, min([R1_Inf_Mean - R1_Inf_Std, R4_Inf_Mean - R4_Inf_Std], [], 'all') - 0.02, 'GC', ...
        %     'HorizontalAlignment', 'center', ...
        %     'VerticalAlignment', 'top', ...
        %     'FontSize', 10);
        % 
        % legend(["R1 Std", "R4 Std", "R1 Mean", "R4 Mean"])
        % saveas(gcf, fullfile(save_dir, 'Ave_Inference.png'))
        % saveas(gcf, fullfile(save_dir, 'Ave_Inference.fig'))
    
        %% Disengagement time analysis
        % Minus 10 here to corect for GC time being 10 points into time series.
        R1_disengage = (estimate_disengage_times(exp(R1_States'), 0.5, 2, 20) - 10)*10;
        R4_disengage = (estimate_disengage_times(exp(R4_States'), 0.5, 2, 20) - 10)*10;
    
    
        % edges = 0:50:2000; % bin in 10-frame increments
        % figure
        % histogram(R1_disengage, edges, 'FaceColor', 'blue', 'FaceAlpha', 0.5)
        % hold on
        % histogram(R4_disengage, edges, 'FaceColor', 'red', 'FaceAlpha', 0.5)
        % xlabel('Disengagement Time (ms)')
        % ylabel('Trial Count')
        % legend('R1', 'R4')
        % title('Disengagement Time Distribution')
        % saveas(gcf, fullfile(save_dir, 'Dt_Histogram.png'))
        % saveas(gcf, fullfile(save_dir, 'Dt_Histogram.fig'))
    
        % % Combine disengagement times and group labels
        % figure
        % disengage_times = [R1_disengage(:); R4_disengage(:)];
        % group_labels = [repmat({'R1'}, length(R1_disengage), 1); repmat({'R4'}, length(R4_disengage), 1)];
        % 
        % % Make the box plot
        % figure
        % boxplot(disengage_times, group_labels)
        % ylabel('Disengagement Time (ms)')
        % title('Disengagement Time Distribution')
        % saveas(gcf, fullfile(save_dir, 'Dt_Boxplot.png'))
        % saveas(gcf, fullfile(save_dir, 'Dt_Boxplot.fig'))
    
        %% Heatmap with disengagement time estimates
        % Add back in 10 pre gc points for heatmap visualization
        R1_d_hm = R1_disengage / 10;
        R4_d_hm = R4_disengage / 10;

        R1_D_HM = [R1_D_HM; R1_d_hm];
        R4_D_HM = [R4_D_HM; R4_d_hm];
      
        %% Align the neural activity to the disengagement times
        R1_d_hm = R1_disengage / 10;
        R4_d_hm = R4_disengage / 10;
        
        R1_Neural_Trials;  % 600 x 106 x 38
        R4_Neural_Trials;  % 600 x 84 x 38
    
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
    
    
        n_trials = size(R4_Neural_Trials, 2);
        n_neurons = size(R4_Neural_Trials, 3);
        window_size = 101;
        pre_points = 50;
        post_points = 50;
        
        R4_d_aligned = nan(window_size, n_trials, n_neurons);
        
        for i = 1:n_trials
            align_point = round(R4_d_hm(i)) + 100;  % add offset
            start_idx = align_point - pre_points;
            end_idx = align_point + post_points;
        
            % Check bounds
            if start_idx >= 1 && end_idx <= size(R4_Neural_Trials, 1)
                R4_d_aligned(:, i, :) = R4_Neural_Trials(start_idx:end_idx, i, :);
            else
                warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
            end
        end
    
        % %% Test overlay
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % R1_PSTH = squeeze(mean(R1_d_aligned, 3));  % average across trials
        % imagesc(R1_PSTH');
        % xlabel('Time (s)');
        % ylabel('Neuron');
        % title('R1 PSTH: Dt Aligned');
        % colormap('jet');
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '--k', 'LineWidth', 1);     
        % text(51, 0, 'Dt', ...
        %     'HorizontalAlignment', 'center', ...
        %     'VerticalAlignment', 'bottom', ...
        %     'FontSize', 12, ...
        %     'Color', 'k');
        % 
        % % --- Overlay population average on a second axis ---
        % ax1 = gca;  % main axis
        % ax1_pos = ax1.Position;  % get position of original axis
        % 
        % % Create new transparent axis on top
        % ax2 = axes('Position', ax1_pos, ...
        %            'Color', 'none', ...
        %            'YAxisLocation', 'right', ...
        %            'XAxisLocation', 'bottom', ...
        %            'XColor', 'k', 'YColor', 'k', ...
        %            'Box', 'off');
        % 
        % hold(ax2, 'on');
        % pop_ave_R1 = mean(R1_PSTH, 2);  % 101 x 1
        % plot(ax2, 1:101, pop_ave_R1, 'k', 'LineWidth', 2);
        % 
        % % Match x-axis ticks with heatmap
        % set(ax2, 'XLim', [1, 101]);
        % set(ax2, 'XTick', 1:10:101);
        % set(ax2, 'XTickLabel', []);  % hide duplicate x labels
        % set(ax2, 'YLim', [min(pop_ave_R1)*0.9, max(pop_ave_R1)*1.1]);
        % set(ax2, 'YTickLabel', []);  % hide duplicate x labels
        % y1 = ylabel("Spike Rate");
        % y1.Position(1) = y1.Position(1)-1;
        % legend("Pop Ave")
        % % ylabel(ax2, 'FR');
        % saveas(gcf, fullfile(save_dir, 'R1_Ave_PSTH_Dt_Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'R1_Ave_PSTH_Dt_Aligned.fig'));
        % 
        % 
        % 
        % 
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % R4_PSTH = squeeze(mean(R4_d_aligned, 3));  % average across trials
        % imagesc(R4_PSTH');
        % xlabel('Time (s)');
        % ylabel('Neuron');
        % title('R4 PSTH: Dt Aligned');
        % colormap('jet');
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '--k', 'LineWidth', 1);     
        % text(51, 0, 'Dt', ...
        %     'HorizontalAlignment', 'center', ...
        %     'VerticalAlignment', 'bottom', ...
        %     'FontSize', 12, ...
        %     'Color', 'k');
        % 
        % % --- Overlay population average on a second axis ---
        % ax1 = gca;  % main axis
        % ax1_pos = ax1.Position;  % get position of original axis
        % 
        % % Create new transparent axis on top
        % ax2 = axes('Position', ax1_pos, ...
        %            'Color', 'none', ...
        %            'YAxisLocation', 'right', ...
        %            'XAxisLocation', 'bottom', ...
        %            'XColor', 'k', 'YColor', 'k', ...
        %            'Box', 'off');
        % 
        % hold(ax2, 'on');
        % pop_ave_R4 = mean(R4_PSTH, 2);  % 101 x 1
        % plot(ax2, 1:101, pop_ave_R4, 'k', 'LineWidth', 2);
        % 
        % % Match x-axis ticks with heatmap
        % set(ax2, 'XLim', [1, 101]);
        % set(ax2, 'XTick', 1:10:101);
        % set(ax2, 'XTickLabel', []);  % hide duplicate x labels
        % set(ax2, 'YLim', [min(pop_ave_R1)*0.9, max(pop_ave_R1)*1.1]);
        % set(ax2, 'YTickLabel', []);  % hide duplicate x labels
        % y1 = ylabel("Spike Rate");
        % y1.Position(1) = y1.Position(1)-1;
        % legend("Pop Ave")
        % % ylabel(ax2, 'FR');
        % saveas(gcf, fullfile(save_dir, 'R4_Ave_PSTH_Dt_Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'R4_Ave_PSTH_Dt_Aligned.fig'));
    
        %% NEW FIGURE %%
    
        engaged_durations = [];
        disengaged_durations = [];
        engaged_licks = {};
        disengaged_licks = {};
        
        % Transpose to [timepoints x trials]
        All_Tongue = All_Tongue';
        All_States = All_States';
        
        nTrials = size(All_Tongue, 2);
        
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
        % xticks([0 60 110 160 210]);
        % xticklabels({'-.10', '0.5', '1.0', '1.5', '2.0'});
        % % set(gca, 'TickLength', [0 0]);
        % % box off;
        % ylim([0 nTrials + 1]);
        % xlim([0 210])
        % xline(11, '--k', 'LineWidth', 1);  % Add vertical line for GC
        % text(11, -5, 'GC', ...
        %     'HorizontalAlignment', 'center', ...
        %     'VerticalAlignment', 'bottom', ...
        %     'FontSize', 12, ...
        %     'Color', 'k');
        % 
        % saveas(gcf, fullfile(save_dir, 'Labeled_Licks.png'));  % Save current figure as png
        % saveas(gcf, fullfile(save_dir, 'Labeled_Licks.fig'));  % Save as MATLAB .fig file
    
        
        %% Box plot of Lick Bout Durations
        % Convert durations to milliseconds
        engaged_ms = (engaged_durations' *10);
        disengaged_ms = (disengaged_durations' *10);
        
        % % Combine into a single vector and group labels
        % all_data = [engaged_ms; disengaged_ms];
        % group_labels = [repmat({'Engaged'}, length(engaged_ms), 1); ...
        %                 repmat({'Disengaged'}, length(disengaged_ms), 1)];
        % 
        % % Plot boxplot
        % figure;
        % boxplot(all_data, group_labels);
        % 
        % ylabel('Lick Duration (ms)');
        % title('Engaged vs Disengaged Lick Bout Durations');
        % set(gca, 'FontSize', 12);
        % 
        % % Save figure
        % saveas(gcf, fullfile(save_dir, 'Duration_Comparison_Boxplot.png'));
        % saveas(gcf, fullfile(save_dir, 'Duration_Comparison_Boxplot.fig'));
    
    
        % %% Histogram of lick lengths
        % figure;
        % all_ms = [engaged_ms; disengaged_ms];
        % histogram(engaged_ms, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'DisplayName', 'Engaged', 'BinWidth',10);
        % hold on
        % histogram(disengaged_ms, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'DisplayName', 'Disengaged', 'BinWidth',10);
        % title("Lick Duration Distributions")
        % xlabel("Lick Duration (ms)")
        % ylabel("(Count)")
        % legend(["Engaged", "Disengaged"])
        % % Save figure
        % saveas(gcf, fullfile(save_dir, 'EngDis_Histogram_Duration.png'));
        % saveas(gcf, fullfile(save_dir, 'EngDis_Histogram_Duration.fig'));
        close all;
    catch
        disp("Error")
    end
end

%% Session Wide Average Inference
disp("Plotting Started")

% Stack into matrices: [num_sessions x timepoints]
R1_all = cat(1, R1_INF_MEAN{:});  % [num_sessions x T]
R4_all = cat(1, R4_INF_MEAN{:});

% Compute mean and std across sessions
R1_Inf_Mean = mean(R1_all, 1);  % [1 x T]
R1_Inf_Std  = std(R1_all, 0, 1);

R4_Inf_Mean = mean(R4_all, 1);
R4_Inf_Std  = std(R4_all, 0, 1);

x = 1:length(R1_Inf_Mean);  % time axis, adapt if needed

figure;
hold on

% Shaded std region for R1
fill([x, fliplr(x)], ...
     [R1_Inf_Mean + R1_Inf_Std, fliplr(R1_Inf_Mean - R1_Inf_Std)], ...
     [0.6 0.8 1], ...       % light blue color
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);

% Shaded std region for R4
fill([x, fliplr(x)], ...
     [R4_Inf_Mean + R4_Inf_Std, fliplr(R4_Inf_Mean - R4_Inf_Std)], ...
     [0.7 0.9 1], ...       % light red/pink color
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);

% Plot mean traces
plot(x, R1_Inf_Mean, 'b', 'LineWidth', 2);
plot(x, R4_Inf_Mean, "Color",[0 0.76 1], 'LineWidth', 2);

ylabel("State 1 Probability");
xlabel("Time (s)");
xticks([0 60 110 160 210]);
xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'});
xlim([0, 200]);
title("Trial Averaged Inference");

xline(11, "--k", "LineWidth", 1);
text(11, min([R1_Inf_Mean - R1_Inf_Std, R4_Inf_Mean - R4_Inf_Std], [], 'all') - 0.02, ...
    'GC', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'top', ...
    'FontSize', 10);

legend(["R1 Std", "R4 Std", "R1 Mean", "R4 Mean"]);

saveas(gcf, fullfile(summary_dir, 'Ave_Inference.png'));
saveas(gcf, fullfile(summary_dir, 'Ave_Inference.fig'));

%% Session Wide Dt Histogram
edges = 0:50:2000; % bin in 10-frame increments
figure
histogram(R1_D_HM*10, edges, 'FaceColor', 'blue', 'FaceAlpha', 0.5)
hold on
histogram(R4_D_HM*10, edges, 'FaceColor', [0 0.76 1], 'FaceAlpha', 0.5)
xlabel('Disengagement Time (ms)')
ylabel('Trial Count')
legend('R1', 'R4')
title('Disengagement Time Distribution')
xticks(0:250:2000);                          % Tick positions in ms
xticklabels(string((0:250:2000)/1000));      % Convert to seconds as labels
saveas(gcf, fullfile(summary_dir, 'Dt_Histogram.png'))
saveas(gcf, fullfile(summary_dir, 'Dt_Histogram.fig'))

%% Session Wide Dt boxplot
disengage_times = [R1_D_HM*10; R4_D_HM*10];

% Create matching group labels
group_labels = [repmat({'R1'}, length(R1_D_HM), 1); ...
                repmat({'R4'}, length(R4_D_HM), 1)];

% Make the box plot
figure
boxplot(disengage_times, group_labels)
ylabel('Disengagement Time (s)')  % Units: seconds
yticks(0:250:2000);                          % Tick positions in ms
yticklabels(string((0:250:2000)/1000));
title('Disengagement Time Distribution')

saveas(gcf, fullfile(summary_dir, 'Dt_Boxplot.png'))
saveas(gcf, fullfile(summary_dir, 'Dt_Boxplot.fig'))









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


