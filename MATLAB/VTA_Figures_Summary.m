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
% Initialize storage for each session's data
Stim_INF_MEAN = {};
NoStim_INF_MEAN = {};
Stim_INF_MEAN_FC = {};
NoStim_INF_MEAN_FC = {};
Stim_D_HM = [];
NoStim_D_HM = [];

%% Import the state inference and tongue data
base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_VTA_Aug\';
alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\VTA_Stim';
summary_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_VTA_Aug\Summary_Figs_M1';
subfolder = '';

% Get list of all subfolders in base_dir
session_dirs = dir(base_dir);
session_dirs = session_dirs([session_dirs.isdir]);  % Keep only directories
session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));  % Remove . and ..

for ij = 1:length(session_dirs)
    % try
        session_name = session_dirs(ij).name;
        % 
        % if ~ismember(session_name, valid_session_names)
        %     fprintf('Skipping %s — not in metadata or not marked for inclusion.\n', session_name);
        %     continue;
        % end
        
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
        Stim_States   = readmatrix(fullfile(save_dir, 'Stim_States_Reg.csv'));
        NoStim_States   = readmatrix(fullfile(save_dir, 'NoStim_States_Reg.csv'));
        Stim_Tongue   = readmatrix(fullfile(save_dir, 'Stim_Tongue_Reg.csv'));
        NoStim_Tongue   = readmatrix(fullfile(save_dir, 'NoStim_Tongue_Reg.csv'));
        PC          = readmatrix(fullfile(save_dir, 'VTA_PC_R2_Reg.csv'));
        % Stim_Trial_Track = readmatrix(fullfile(alt_session_dir, 'Stim_Trial_Track.csv'));
        % NoStim_Trial_Track = readmatrix(fullfile(alt_session_dir, 'NoStim_Trial_Track.csv'));
        
        % close all
        % Stim_Tongue = Stim_Tongue';
        % NoStim_Tongue = NoStim_Tongue';
        
        % %% Chop the data to relevant periods
        % start = 400;
        % stop = 800;
        % 
        % Stim_States_chopped = Stim_States(:, start:stop);
        % NoStim_States_chopped = NoStim_States(:, start:stop);
        % Stim_Tongue = Stim_Tongue(:, start:stop);
        % NoStim_Tongue = NoStim_Tongue(:, start:stop);
    
        %% Neural Probe Data Importation and Chopping
        prb = alt_session_dir(end);
        
        Stim_Path = alt_session_dir + "\Probe" + prb + "_Stim_Uncut.csv";
        NoStim_Path = alt_session_dir + "\Probe" + prb + "_NoStim_Uncut.csv";
    
        % Read the matrices
        Stim_Neural = readmatrix(Stim_Path);
        NoStim_Neural = readmatrix(NoStim_Path);
    
        [x, xx] = size(Stim_Neural);
    
        % Reshape into trials
        % T x N_trials x N_neurons
        Stim_Neural_Trials = reshape(Stim_Neural, 600, [], xx);  % size: 600 x 95 x 54
        Stim_Neural_Trials = permute(Stim_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
        % Stim_Neural_Trials = zscore_pregc(Stim_Neural_Trials, 100);
        
        NoStim_Neural_Trials = reshape(NoStim_Neural, 600, [], xx);  % size: 600 x 95 x 54
        NoStim_Neural_Trials = permute(NoStim_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
        % NoStim_Neural_Trials = zscore_pregc(NoStim_Neural_Trials, 100);    
        
        %% -- Combine the NoStim and Stim Datasets and Heatmaps -- %%
        All_States = exp([NoStim_States; Stim_States]);
        % All_States = [NoStim_States; Stim_States];
        All_Tongue = [NoStim_Tongue; Stim_Tongue];
        
        %% Normalize the kinametic data
        All_Tongue = range_normalize_with_nans(All_Tongue);
        % All_Tongue(All_Tongue == 0) = NaN;
        
        % Normalize each row for NoStim_Tongue and Stim_Tongue
        NoStim_Tongue_norm = (NoStim_Tongue - nanmin(NoStim_Tongue, [], 2)) ./ (nanmax(NoStim_Tongue, [], 2) - nanmin(NoStim_Tongue, [], 2));
        NoStim_Tongue_norm(NoStim_Tongue_norm == 0) = NaN;
        
        Stim_Tongue_norm = (Stim_Tongue - nanmin(Stim_Tongue, [], 2)) ./ (nanmax(Stim_Tongue, [], 2) - nanmin(Stim_Tongue, [], 2));
        Stim_Tongue_norm(Stim_Tongue_norm == 0) = NaN;
        
        %%      
        Stim_Tongue = Stim_Tongue_norm';
        NoStim_Tongue = NoStim_Tongue_norm';
        Stim_States = Stim_States';
        NoStim_States = NoStim_States';    
        
        %% Trial averaged inference plots
        % Trial averaged inference plots
        Stim_Inf_Mean = mean(exp(Stim_States'), 1);
        Stim_Inf_Std  = std(exp(Stim_States'), 0, 1);
        
        NoStim_Inf_Mean = mean(exp(NoStim_States'), 1);
        NoStim_Inf_Std  = std(exp(NoStim_States'), 0, 1);

        Stim_INF_MEAN{end+1} = Stim_Inf_Mean;
        NoStim_INF_MEAN{end+1} = NoStim_Inf_Mean;


        x = 1:length(Stim_Inf_Mean);

        %% Aligned to FC
        FCs_Stim = readmatrix(fullfile(alt_session_dir, 'FCs_Stim.csv')) - 100;
        FCs_NoStim = readmatrix(fullfile(alt_session_dir, 'FCs_NoStim.csv'))- 100;
        
        % Parameters
        window_size = 211;
        pre_points = 10;
        post_points = 200;
        
        % ---- Stim ----
        n_trials = size(Stim_States, 2);
        Stim_FC_aligned = nan(window_size, n_trials);
        
        for i = 1:n_trials
            align_point = round(FCs_Stim(i));
            start_idx = align_point - pre_points;
            end_idx = align_point + post_points;
        
            % Determine valid indices
            valid_start = max(start_idx, 1);
            valid_end = min(end_idx, size(Stim_States, 1));
        
            % Corresponding indices in the aligned window
            insert_start = valid_start - start_idx + 1;
            insert_end = insert_start + (valid_end - valid_start);
        
            % Fill with actual data
            Stim_FC_aligned(insert_start:insert_end, i) = Stim_States(valid_start:valid_end, i);
        end
        
        % ---- NoStim ----
        n_trials = size(NoStim_States, 2);
        NoStim_FC_aligned = nan(window_size, n_trials);
        
        for i = 1:n_trials
            align_point = round(FCs_NoStim(i));
            start_idx = align_point - pre_points;
            end_idx = align_point + post_points;
        
            valid_start = max(start_idx, 1);
            valid_end = min(end_idx, size(NoStim_States, 1));
        
            insert_start = valid_start - start_idx + 1;
            insert_end = insert_start + (valid_end - valid_start);
        
            NoStim_FC_aligned(insert_start:insert_end, i) = NoStim_States(valid_start:valid_end, i);
        end

        


        % Trial averaged inference plots
        Stim_Inf_Mean = nanmean(exp(Stim_FC_aligned'), 1);
        Stim_Inf_Std  = nanstd(exp(Stim_FC_aligned'), 0, 1);
        
        NoStim_Inf_Mean = nanmean(exp(NoStim_FC_aligned'), 1);
        NoStim_Inf_Std  = nanstd(exp(NoStim_FC_aligned'), 0, 1);

        % Replace NaNs in mean and std with 0.0
        Stim_Inf_Mean(isnan(Stim_Inf_Mean)) = 0.0;
        Stim_Inf_Std(isnan(Stim_Inf_Std)) = 0.0;
        
        NoStim_Inf_Mean(isnan(NoStim_Inf_Mean)) = 0.0;
        NoStim_Inf_Std(isnan(NoStim_Inf_Std)) = 0.0;

        Stim_INF_MEAN_FC{end+1} = Stim_Inf_Mean;
        NoStim_INF_MEAN_FC{end+1} = NoStim_Inf_Mean;

        % figure
        % hold on
        % 
        % % Shaded std region for Stim
        % fill([x, fliplr(x)], ...
        %      [Stim_Inf_Mean + Stim_Inf_Std, fliplr(Stim_Inf_Mean - Stim_Inf_Std)], ...
        %      [0.8 0.8 1], ...       % light blue color
        %      'EdgeColor', 'none', ...
        %      'FaceAlpha', 0.4);
        % 
        % % Shaded std region for NoStim
        % fill([x, fliplr(x)], ...
        %      [NoStim_Inf_Mean + NoStim_Inf_Std, fliplr(NoStim_Inf_Mean - NoStim_Inf_Std)], ...
        %      [1 0.8 0.8], ...       % light red/pink color
        %      'EdgeColor', 'none', ...
        %      'FaceAlpha', 0.7);
        % 
        % % Plot mean traces
        % plot(x, Stim_Inf_Mean, 'b', 'LineWidth', 2)
        % plot(x, NoStim_Inf_Mean, 'r', 'LineWidth', 2)
        % 
        % ylabel("State 1 Probability")
        % xlabel("Time (s)")
        % xticks([0 60 110 160 210]); % Position of ticks
        % xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'}); % Labels corresponding to time in seconds
        % xlim([0, 200])
        % title("Trial Averaged Inference")
        % 
        % xline(11, "--k", "LineWidth", 1)
        % text(11, min([Stim_Inf_Mean - Stim_Inf_Std, NoStim_Inf_Mean - NoStim_Inf_Std], [], 'all') - 0.02, 'GC', ...
        %     'HorizontalAlignment', 'center', ...
        %     'VerticalAlignment', 'top', ...
        %     'FontSize', 10);
        % 
        % legend(["Stim Std", "NoStim Std", "Stim Mean", "NoStim Mean"])
        % saveas(gcf, fullfile(save_dir, 'Ave_Inference.png'))
        % saveas(gcf, fullfile(save_dir, 'Ave_Inference.fig'))
    
        %% Disengagement time analysis
        % Minus 10 here to corect for GC time being 10 points into time series.
        Stim_disengage = (estimate_disengage_times(exp(Stim_States'), 0.5, 2, 20) - 10)*10;
        NoStim_disengage = (estimate_disengage_times(exp(NoStim_States'), 0.5, 2, 20) - 10)*10;
    
    
        % edges = 0:50:2000; % bin in 10-frame increments
        % figure
        % histogram(Stim_disengage, edges, 'FaceColor', 'blue', 'FaceAlpha', 0.5)
        % hold on
        % histogram(NoStim_disengage, edges, 'FaceColor', 'red', 'FaceAlpha', 0.5)
        % xlabel('Disengagement Time (ms)')
        % ylabel('Trial Count')
        % legend('Stim', 'NoStim')
        % title('Disengagement Time Distribution')
        % saveas(gcf, fullfile(save_dir, 'Dt_Histogram.png'))
        % saveas(gcf, fullfile(save_dir, 'Dt_Histogram.fig'))
    
        % % Combine disengagement times and group labels
        % figure
        % disengage_times = [Stim_disengage(:); NoStim_disengage(:)];
        % group_labels = [repmat({'Stim'}, length(Stim_disengage), 1); repmat({'NoStim'}, length(NoStim_disengage), 1)];
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
        Stim_d_hm = Stim_disengage / 10;
        NoStim_d_hm = NoStim_disengage / 10;

        Stim_D_HM = [Stim_D_HM; Stim_d_hm];
        NoStim_D_HM = [NoStim_D_HM; NoStim_d_hm];
      
        %% Align the neural activity to the disengagement times
        Stim_d_hm = Stim_disengage / 10;
        NoStim_d_hm = NoStim_disengage / 10;
        
        Stim_Neural_Trials;  % 600 x 106 x 38
        NoStim_Neural_Trials;  % 600 x 84 x 38
    
        n_trials = size(Stim_Neural_Trials, 2);
        n_neurons = size(Stim_Neural_Trials, 3);
        window_size = 101;
        pre_points = 50;
        post_points = 50;
        
        Stim_d_aligned = nan(window_size, n_trials, n_neurons);
        
        for i = 1:n_trials
            align_point = round(Stim_d_hm(i)) + 100;  % add offset
            start_idx = align_point - pre_points;
            end_idx = align_point + post_points;
        
            % Check bounds
            if start_idx >= 1 && end_idx <= size(Stim_Neural_Trials, 1)
                Stim_d_aligned(:, i, :) = Stim_Neural_Trials(start_idx:end_idx, i, :);
            else
                warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
            end
        end
    
    
        n_trials = size(NoStim_Neural_Trials, 2);
        n_neurons = size(NoStim_Neural_Trials, 3);
        window_size = 101;
        pre_points = 50;
        post_points = 50;
        
        NoStim_d_aligned = nan(window_size, n_trials, n_neurons);
        
        for i = 1:n_trials
            align_point = round(NoStim_d_hm(i)) + 100;  % add offset
            start_idx = align_point - pre_points;
            end_idx = align_point + post_points;
        
            % Check bounds
            if start_idx >= 1 && end_idx <= size(NoStim_Neural_Trials, 1)
                NoStim_d_aligned(:, i, :) = NoStim_Neural_Trials(start_idx:end_idx, i, :);
            else
                warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
            end
        end
    
        % %% Test overlay
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % Stim_PSTH = squeeze(mean(Stim_d_aligned, 3));  % average across trials
        % imagesc(Stim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Neuron');
        % title('Stim PSTH: Dt Aligned');
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
        % pop_ave_Stim = mean(Stim_PSTH, 2);  % 101 x 1
        % plot(ax2, 1:101, pop_ave_Stim, 'k', 'LineWidth', 2);
        % 
        % % Match x-axis ticks with heatmap
        % set(ax2, 'XLim', [1, 101]);
        % set(ax2, 'XTick', 1:10:101);
        % set(ax2, 'XTickLabel', []);  % hide duplicate x labels
        % set(ax2, 'YLim', [min(pop_ave_Stim)*0.9, max(pop_ave_Stim)*1.1]);
        % set(ax2, 'YTickLabel', []);  % hide duplicate x labels
        % y1 = ylabel("Spike Rate");
        % y1.Position(1) = y1.Position(1)-1;
        % legend("Pop Ave")
        % % ylabel(ax2, 'FR');
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_PSTH_Dt_Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_PSTH_Dt_Aligned.fig'));
        % 
        % 
        % 
        % 
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % NoStim_PSTH = squeeze(mean(NoStim_d_aligned, 3));  % average across trials
        % imagesc(NoStim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Neuron');
        % title('NoStim PSTH: Dt Aligned');
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
        % pop_ave_NoStim = mean(NoStim_PSTH, 2);  % 101 x 1
        % plot(ax2, 1:101, pop_ave_NoStim, 'k', 'LineWidth', 2);
        % 
        % % Match x-axis ticks with heatmap
        % set(ax2, 'XLim', [1, 101]);
        % set(ax2, 'XTick', 1:10:101);
        % set(ax2, 'XTickLabel', []);  % hide duplicate x labels
        % set(ax2, 'YLim', [min(pop_ave_Stim)*0.9, max(pop_ave_Stim)*1.1]);
        % set(ax2, 'YTickLabel', []);  % hide duplicate x labels
        % y1 = ylabel("Spike Rate");
        % y1.Position(1) = y1.Position(1)-1;
        % legend("Pop Ave")
        % % ylabel(ax2, 'FR');
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_PSTH_Dt_Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_PSTH_Dt_Aligned.fig'));
    
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
    % catch
    %     disp("Error")
    % end
end

%% Session Wide Average Inference
disp("Plotting Started")

% Stack into matrices: [num_sessions x timepoints]
Stim_all = cat(1, Stim_INF_MEAN{:});  % [num_sessions x T]
NoStim_all = cat(1, NoStim_INF_MEAN{:});

% Compute mean and std across sessions
Stim_Inf_Mean = mean(Stim_all, 1);  % [1 x T]
Stim_Inf_Std  = std(Stim_all, 0, 1);

NoStim_Inf_Mean = mean(NoStim_all, 1);
NoStim_Inf_Std  = std(NoStim_all, 0, 1);

x = 1:length(Stim_Inf_Mean);  % time axis, adapt if needed

figure;
hold on

% Shaded std region for Stim
fill([x, fliplr(x)], ...
     [Stim_Inf_Mean + Stim_Inf_Std, fliplr(Stim_Inf_Mean - Stim_Inf_Std)], ...
     [0.6 0.8 1], ...       % light blue color
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);

% Shaded std region for NoStim
fill([x, fliplr(x)], ...
     [NoStim_Inf_Mean + NoStim_Inf_Std, fliplr(NoStim_Inf_Mean - NoStim_Inf_Std)], ...
     [0.7 0.9 1], ...       % light red/pink color
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);

% Plot mean traces
plot(x, Stim_Inf_Mean, 'b', 'LineWidth', 2);
plot(x, NoStim_Inf_Mean, "Color",[0 0.76 1], 'LineWidth', 2);

ylabel("State 1 Probability");
xlabel("Time (s)");
xticks([0 60 110 160 210]);
xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'});
xlim([0, 200]);
title("Trial Averaged Inference");

xline(11, "--k", "LineWidth", 1);
text(11, min([Stim_Inf_Mean - Stim_Inf_Std, NoStim_Inf_Mean - NoStim_Inf_Std], [], 'all') - 0.02, ...
    'GC', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'top', ...
    'FontSize', 10);

legend(["Stim Std", "NoStim Std", "Stim Mean", "NoStim Mean"]);

saveas(gcf, fullfile(summary_dir, 'Ave_Inference.png'));
saveas(gcf, fullfile(summary_dir, 'Ave_Inference.fig'));


%% Trial averaged inference aligned to FC

% Stack into matrices: [num_sessions x timepoints]
Stim_all = cat(1, Stim_INF_MEAN_FC{:});  % [num_sessions x T]
NoStim_all = cat(1, NoStim_INF_MEAN_FC{:});

% Compute mean and std across sessions
Stim_Inf_Mean = nanmean(Stim_all, 1);  % [1 x T]
Stim_Inf_Std  = nanstd(Stim_all, 0, 1);

NoStim_Inf_Mean = nanmean(NoStim_all, 1);
NoStim_Inf_Std  = nanstd(NoStim_all, 0, 1);

x = 1:length(Stim_Inf_Mean);  % time axis, adapt if needed

figure
hold on

% Clamp shaded areas between 0 and 1
Stim_Shade_Upper = min(Stim_Inf_Mean + Stim_Inf_Std, 1);
Stim_Shade_Lower = max(Stim_Inf_Mean - Stim_Inf_Std, 0);

NoStim_Shade_Upper = min(NoStim_Inf_Mean + NoStim_Inf_Std, 1);
NoStim_Shade_Lower = max(NoStim_Inf_Mean - NoStim_Inf_Std, 0);


% Shaded std region for Stim
Stim_s = fill([x, fliplr(x)], ...
     [Stim_Shade_Upper, fliplr(Stim_Shade_Lower)], ...
     [0.6 0.8 1], ...
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);

% Shaded std region for NoStim
NoStim_s = fill([x, fliplr(x)], ...
     [NoStim_Shade_Upper, fliplr(NoStim_Shade_Lower)], ...
     [0.7 0.9 1], ...
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.7);


% Plot mean traces
Stim_p = plot(x, Stim_Inf_Mean, 'b', 'LineWidth', 2);
NoStim_p = plot(x, NoStim_Inf_Mean, 'Color', [0 0.76 1], 'LineWidth', 2);

ylabel("State 1 Probability")
xlabel("Time (s)")
xticks([0 60 110 160 210]); % Position of ticks
xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'}); % Labels corresponding to time in seconds
xlim([0, 200])
ylim([0, 1])
title("Trial Averaged Inference")

xline(11, "--k", "LineWidth", 1)
text(11, min([Stim_Inf_Mean - Stim_Inf_Std, NoStim_Inf_Mean - NoStim_Inf_Std], [], 'all') - 0.02, 'GC', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'top', ...
    'FontSize', 10);

legend([Stim_p, NoStim_p, Stim_s, NoStim_s], {'Stim Mean', 'NoStim Mean', 'Stim Std', 'NoStim Std'} )
saveas(gcf, fullfile(summary_dir, 'Ave_Inference_FCAligned.png'))
saveas(gcf, fullfile(summary_dir, 'Ave_Inference_FCAligned.fig'))

%% Session Wide Dt Histogram
edges = 0:50:2000; % bin in 10-frame increments
figure
histogram(Stim_D_HM*10, edges, 'FaceColor', 'blue', 'FaceAlpha', 0.5)
hold on
histogram(NoStim_D_HM*10, edges, 'FaceColor', [0 0.76 1], 'FaceAlpha', 0.5)
xlabel('Disengagement Time (ms)')
ylabel('Trial Count')
legend('Stim', 'NoStim')
title('Disengagement Time Distribution')
xticks(0:250:2000);                          % Tick positions in ms
xticklabels(string((0:250:2000)/1000));      % Convert to seconds as labels
saveas(gcf, fullfile(summary_dir, 'Dt_Histogram.png'))
saveas(gcf, fullfile(summary_dir, 'Dt_Histogram.fig'))

%% Session Wide Dt boxplot
disengage_times = [Stim_D_HM*10; NoStim_D_HM*10];

% Create matching group labels
group_labels = [repmat({'Stim'}, length(Stim_D_HM), 1); ...
                repmat({'NoStim'}, length(NoStim_D_HM), 1)];

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
Stim_ms = Stim_D_HM * 10;
NoStim_ms = NoStim_D_HM * 10;
edges = 0:100:2000;
xticks_ms = -250:250:2000;
xticklabels_sec = string(xticks_ms / 1000);

%% 1. Stim Boxplot
figure;
boxplot(Stim_ms, 'Orientation', 'vertical', 'Colors', 'b', 'Symbol', '');
ylim([-250 2000])
yticks(xticks_ms)
yticklabels(xticklabels_sec)
ylabel('Disengagement Time (s)')
title('Stim Boxplot')
set(gca, 'XTickLabel', {})  % Hide y-axis labels for Illustrator
saveas(gcf, fullfile(summary_dir, 'Stim_Dt_Boxplot.png'))
saveas(gcf, fullfile(summary_dir, 'Stim_Dt_Boxplot.fig'))

%% 2. NoStim Boxplot
figure;
boxplot(NoStim_ms, 'Orientation', 'vertical', 'Colors', [0 0.76 1], 'Symbol', '');
ylim([-250 2000])
yticks(xticks_ms)
yticklabels(xticklabels_sec)
ylabel('Disengagement Time (s)')
title('NoStim Boxplot')
set(gca, 'XTickLabel', {})
saveas(gcf, fullfile(summary_dir, 'NoStim_Dt_Boxplot.png'))
saveas(gcf, fullfile(summary_dir, 'NoStim_Dt_Boxplot.fig'))

%% 3. Stim Histogram
% Scale factor
scale_factor = 1;

% Histogram binning
[counts, centers] = histcounts(Stim_ms, edges);
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
title('Stim Histogram (Scaled)')
saveas(gcf, fullfile(summary_dir, 'Stim_Dt_Histogram_Scaled.png'))
saveas(gcf, fullfile(summary_dir, 'Stim_Dt_Histogram_Scaled.fig'))

%% 4. NoStim Histogram
[counts, centers] = histcounts(NoStim_ms, edges);
centers = edges(1:end-1) + diff(edges)/2;
scaled_counts = counts / scale_factor;

figure;
barh(centers, scaled_counts, 1, 'FaceColor', [0 0.76 1], 'FaceAlpha', 0.6)
ylim([-250 2000])
yticks(-250:250:2000)
yticklabels(string((-250:250:2000)/1000))
ylabel('Disengagement Time (s)')
xlabel(['Scaled Trial Count (/ ' num2str(scale_factor) ')'])
title('NoStim Histogram (Scaled)')
saveas(gcf, fullfile(summary_dir, 'NoStim_Dt_Histogram_Scaled.png'))
saveas(gcf, fullfile(summary_dir, 'NoStim_Dt_Histogram_Scaled.fig'))








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


