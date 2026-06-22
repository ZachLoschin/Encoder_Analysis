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
base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_VTA_Aug\';
alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\VTA_Stim';
subfolder = '';

% Get list of all subfolders in base_dir
session_dirs = dir(base_dir);
session_dirs = session_dirs([session_dirs.isdir]);  % Keep only directories
session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));  % Remove . and ..

for ij = 1:length(session_dirs)
    % try
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
    
        % Import relevant lick contact times
        Stim_Lick1_Path = alt_session_dir + "\FCs_Stim.csv";
        % NoStim_Lick4_Path = alt_session_dir + "\Fourth_C_NoStim.csv";

        FCs_Stim = readmatrix(fullfile(alt_session_dir, 'FCs_Stim.csv')) - 100;
        FCs_NoStim = readmatrix(fullfile(alt_session_dir, 'FCs_NoStim.csv'))- 100;

        % - 100 takes out pre gc period. Now times are relative to GC
        Stim_Lick1 = readmatrix(Stim_Lick1_Path) - 100;
        % NoStim_Lick4 = readmatrix(NoStim_Lick4_Path) - 100;


        %% Plot PSTHs for visualization
       % Average across trials
        Stim_PSTH = squeeze(mean(Stim_Neural_Trials, 3));  % size: 600 x 54
        
        % Transpose for heatmap: neurons on Y, time on X
        imagesc(Stim_PSTH');  % now size is xx x 600
        xlabel('Time (samples)');
        ylabel('Neuron');
        title('Stim PSTH (Trial-Averaged)');
        colorbar;
        
        % Optional: improve visualization
        set(gca, 'YDir', 'normal');
    
    
        %% Average across trials
        NoStim_PSTH = squeeze(mean(NoStim_Neural_Trials, 3));  % size: 600 x 54
        % NoStim_PSTH = NoStim_PSTH(90:200, :);
        
        % Transpose for heatmap: neurons on Y, time on X
        imagesc(NoStim_PSTH');  % now size is xx x 600
        xlabel('Time (samples)');
        ylabel('Neuron');
        title('NoStim PSTH (Trial-Averaged)');
        colorbar;
        
        % Optional: improve visualization
        set(gca, 'YDir', 'normal');
    
        
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
        % Initialize the figure
        
        Stim_Tongue = Stim_Tongue_norm';
        NoStim_Tongue = NoStim_Tongue_norm';
        Stim_States = Stim_States';
        NoStim_States = NoStim_States';
        % All_States = All_States';
        figure;
        hold on;
        
        % Plotting parameters
        lw = 0.75;  % Line width
        px = 75; py = 75;
        width = 700; height = 800;
        set(gcf, 'Position', [px, py, width, height]);
        
        % Define colors for the conditions
        NoStim_color = 'k'; % Black for NoStim
        Stim_color = 'k'; % Black for Stim
    
        % Overlay heatmap
        h = imagesc(1:size(All_States, 2), 1:size(All_States, 1), All_States);
        
        % Plotting line plots
        for j = 1:size(NoStim_Tongue, 2)
            plot(1:length(NoStim_Tongue(:, j)), j-1 + NoStim_Tongue(:, j), NoStim_color, 'LineWidth', lw);
        end
        
        for j = 1:size(Stim_Tongue, 2)
            plot(1:length(Stim_Tongue(:, j)), j-1 + size(NoStim_Tongue, 2) + Stim_Tongue(:, j), Stim_color, 'LineWidth', lw);
        end
            
        
        % Adjust the colormap to suit your data range
        colormap(custom_cmap);  % You can use 'jet', 'hot', 'cool', or any other colormap
        % colormap(linspecer);  % Flip the colormap if needed
        
        % Adjust the heatmap properties
        % set(h, 'AlphaData', 0.5); % Adjust transparency to see the line plots underneath
        
        % Add a colorbar
        cc = colorbar;
        
        % Add a horizontal line to separate the conditions
        yline(size(NoStim_Tongue, 2), 'k-', 'LineWidth', 3);
        
        % Add labels or annotations
        text(-30, size(NoStim_Tongue, 2) + 2, 'Stim', 'FontSize', 12, 'Color', "k");
        text(-30, size(NoStim_Tongue, 2) - 2, 'NoStim', 'FontSize', 12, 'Color', "k");
        
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
        
        title("Single Trial State Estimates: Stim and NoStim");
        xline(11, '--k', 'LineWidth', 1);  % Add vertical line for GC
        text(11, -5, 'GC', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', 'k');
        saveas(gcf, fullfile(save_dir, 'Inference_Heatmap.png'));  % Save current figure as png
        saveas(gcf, fullfile(save_dir, 'Inference_Heatmap.fig'));  % Save as MATLAB .fig file
        
        
        
        %% Trial averaged inference plots
        lb = [0 0.76 1];

        % Trial averaged inference plots
        Stim_Inf_Mean = mean(exp(Stim_States'), 1);
        Stim_Inf_Std  = std(exp(Stim_States'), 0, 1);
        
        NoStim_Inf_Mean = mean(exp(NoStim_States'), 1);
        NoStim_Inf_Std  = std(exp(NoStim_States'), 0, 1);
        
        x = 1:length(Stim_Inf_Mean);
        
        figure
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
        plot(x, Stim_Inf_Mean, 'b', 'LineWidth', 2)
        plot(x, NoStim_Inf_Mean, 'Color', [0 0.76 1], 'LineWidth', 2)
        
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
        
        legend(["Stim Std", "NoStim Std", "Stim Mean", "NoStim Mean"])
        saveas(gcf, fullfile(save_dir, 'Ave_Inference.png'))
        saveas(gcf, fullfile(save_dir, 'Ave_Inference.fig'))

        %% Trial averaged inference aligned to FC

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

        
        x = 1:length(Stim_Inf_Mean);
        
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
        saveas(gcf, fullfile(save_dir, 'Ave_Inference_FCAligned.png'))
        saveas(gcf, fullfile(save_dir, 'Ave_Inference_FCAligned.fig'))
    
            
        %% Disengagement time analysis
        % Minus 10 here to corect for GC time being 10 points into time series.
        Stim_disengage = (estimate_disengage_times(exp(Stim_States'), 0.5, 2, 20) - 10)*10;
        NoStim_disengage = (estimate_disengage_times(exp(NoStim_States'), 0.5, 2, 20) - 10)*10;
        
        csvwrite(fullfile(save_dir, "Stim_dt.csv"), Stim_disengage);
        csvwrite(fullfile(save_dir, "NoStim_dt.csv"), NoStim_disengage);
    
        % edges = 0:50:2000; % bin in 10-frame increments
        % figure
        % histogram(Stim_disengage, edges, 'FaceColor', 'blue', 'FaceAlpha', 0.8)
        % hold on
        % histogram(NoStim_disengage, edges, 'FaceColor', [0 0.76 1], 'FaceAlpha', 0.8)
        % xlabel('Disengagement Time (ms)')
        % ylabel('Trial Count')
        % legend('Stim', 'NoStim')
        % title('Disengagement Time Distribution')
        % saveas(gcf, fullfile(save_dir, 'Dt_Histogram.png'))
        % saveas(gcf, fullfile(save_dir, 'Dt_Histogram.fig'))
        % 
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
        % 
        % % %% New plot
        % % % Stim_Trial_Track: list of actual trial numbers for Stim trials
        % % % Stim_disengage: disengagement times, same length as Stim_Trial_Track
        % % % Assumes trial numbers in Stim_Trial_Track correspond directly to Stim_disengage
        % % 
        % % 
        % % window_size = 5;
        % % rel_positions = -window_size:window_size;
        % % 
        % % % Get block switch trials (those whose trial number mod 10 == 1)
        % % block_switch_trials = Stim_Trial_Track(mod(Stim_Trial_Track, 10) == 1);
        % % 
        % % % Preallocate disengagement matrix
        % % disengage_matrix = NaN(length(block_switch_trials), length(rel_positions));
        % % 
        % % % Loop over each block switch trial
        % % for b = 1:length(block_switch_trials)
        % %     switch_trial = block_switch_trials(b);
        % % 
        % %     for w = 1:length(rel_positions)
        % %         trial_number = switch_trial + rel_positions(w);
        % % 
        % %         % Find index of this trial number in Stim_Trial_Track
        % %         idx = find(Stim_Trial_Track == trial_number);
        % % 
        % %         if ~isempty(idx)
        % %             disengage_matrix(b, w) = Stim_disengage(idx);
        % %         end
        % %     end
        % % end
        % % 
        % % % Compute average and standard deviation
        % % mean_disengage = nanmean(disengage_matrix, 1);
        % % std_disengage = nanstd(disengage_matrix, 0, 1);  % std across rows
        % % 
        % % % Compute upper and lower bounds for shading
        % % upper = mean_disengage + std_disengage;
        % % lower = mean_disengage - std_disengage;
        % % 
        % % % Plot
        % % figure;
        % % hold on;
        % % 
        % % % Shaded standard deviation area
        % % fill([rel_positions, fliplr(rel_positions)], ...
        % %      [upper, fliplr(lower)], ...
        % %      [0.8 0.8 1], ...       % light blue fill color
        % %      'EdgeColor', 'none', ...
        % %      'FaceAlpha', 0.4);
        % % 
        % % % Mean line
        % % plot(rel_positions, mean_disengage, '-o', 'LineWidth', 2, 'Color', [0 0 0.8]);
        % % 
        % % % Labels and formatting
        % % xlabel('Trial position relative to block switch');
        % % ylabel('Disengagement Time (ms)');
        % % title('Stim disengagement time around block switches');
        % % grid on;
        % % hold off;
        % % saveas(gcf, fullfile(save_dir, 'BlockSwitch_Dt.png'))
        % % saveas(gcf, fullfile(save_dir, 'BlockSwitch_Dt.fig'))
        % 
        %% Heatmap with disengagement time estimates

        % Add back in 10 pre gc points for heatmap visualization
        Stim_d_hm = Stim_disengage / 10;
        NoStim_d_hm = NoStim_disengage / 10;


        % All_States = All_States';
        figure;
        hold on;

        % Plotting parameters
        lw = 0.75;  % Line width
        px = 75; py = 75;
        width = 700; height = 800;
        set(gcf, 'Position', [px, py, width, height]);

        % Define colors for the conditions
        NoStim_color = 'k'; % Black for NoStim
        Stim_color = 'k'; % Black for Stim

        % Overlay heatmap
        h = imagesc(1:size(All_States, 2), 1:size(All_States, 1), All_States);

        % Plotting line plots
        for j = 1:size(NoStim_Tongue, 2)
            plot(1:length(NoStim_Tongue(:, j)), j-1 + NoStim_Tongue(:, j), NoStim_color, 'LineWidth', lw);
        end

        for j = 1:size(Stim_Tongue, 2)
            plot(1:length(Stim_Tongue(:, j)), j-1 + size(NoStim_Tongue, 2) + Stim_Tongue(:, j), Stim_color, 'LineWidth', lw);
        end

        % Plot disengagement time markers
        for i = 1:length(NoStim_d_hm)
            if ~isnan(NoStim_d_hm(i))
                plot(NoStim_d_hm(i)+10, i, 'wo', 'MarkerSize', 4, 'LineWidth', 1.2);
            end
        end

        for i = 1:length(Stim_d_hm)
            if ~isnan(Stim_d_hm(i))
                y = i + size(NoStim_Tongue, 2);  % Offset for Stim trials
                plot(Stim_d_hm(i)+10, y, 'wo', 'MarkerSize', 4, 'LineWidth', 1.2);
            end
        end    

        % Adjust the colormap to suit your data range
        colormap(custom_cmap);  % You can use 'jet', 'hot', 'cool', or any other colormap
        % colormap(linspecer);  % Flip the colormap if needed

        % Adjust the heatmap properties
        % set(h, 'AlphaData', 0.5); % Adjust transparency to see the line plots underneath

        % Add a colorbar
        colorbar;

        % Add a horizontal line to separate the conditions
        yline(size(NoStim_Tongue, 2), 'k-', 'LineWidth', 3);

        % Add labels or annotations
        text(-30, size(NoStim_Tongue, 2) + 2, 'Stim', 'FontSize', 12, 'Color', "k");
        text(-30, size(NoStim_Tongue, 2) - 2, 'NoStim', 'FontSize', 12, 'Color', "k");

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

        title("Single Trial State Estimates: Stim and NoStim");
        xline(11, '--k', 'LineWidth', 1);  % Add vertical line for GC
        text(11, -5, 'GC', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', 'k');


        disp("done")
        % saveas(gcf, fullfile(save_dir, 'Dt_Heatmap.png'));  % Save current figure as png
        % saveas(gcf, fullfile(save_dir, 'Dt_Heatmap.fig'));  % Save as MATLAB .fig file
        % 
        % 
        % 
        % % %% Block swtich Stim4
        % % rel_positions = -window_size:window_size;
        % % 
        % % % Normalize disengagement times by subtracting trial-type-specific means
        % % Stim_mean = mean(Stim_disengage, 'omitnan');
        % % NoStim_mean = mean(NoStim_disengage, 'omitnan');
        % % Stim_disengage_norm = Stim_disengage - Stim_mean;
        % % NoStim_disengage_norm = NoStim_disengage - NoStim_mean;
        % % 
        % % % Combine trial numbers and disengagement values
        % % all_trials = [Stim_Trial_Track(:); NoStim_Trial_Track(:)];
        % % all_disengage = [Stim_disengage_norm(:); NoStim_disengage_norm(:)];
        % % 
        % % % Identify block switch trials (based on Stim trials only)
        % % block_switch_trials = Stim_Trial_Track(mod(Stim_Trial_Track, 10) == 1);
        % % 
        % % % Preallocate matrix
        % % disengage_matrix = NaN(length(block_switch_trials), length(rel_positions));
        % % 
        % % % Loop through each block switch trial
        % % for b = 1:length(block_switch_trials)
        % %     switch_trial = block_switch_trials(b);
        % % 
        % %     for w = 1:length(rel_positions)
        % %         trial_number = switch_trial + rel_positions(w);
        % % 
        % %         % Look for this trial number in all trials (Stim or NoStim)
        % %         idx = find(all_trials == trial_number);
        % %         if ~isempty(idx)
        % %             disengage_matrix(b, w) = all_disengage(idx);
        % %         end
        % %     end
        % % end
        % % 
        % % % Compute mean and std
        % % mean_disengage = nanmean(disengage_matrix, 1);
        % % std_disengage = nanstd(disengage_matrix, 0, 1);
        % % 
        % % % Shaded region bounds
        % % upper = mean_disengage + std_disengage;
        % % lower = mean_disengage - std_disengage;
        % % 
        % % % Plot
        % % figure;
        % % hold on;
        % % fill([rel_positions, fliplr(rel_positions)], ...
        % %      [upper, fliplr(lower)], ...
        % %      [0.8 0.9 1], 'EdgeColor', 'none', 'FaceAlpha', 0.4);  % shaded std
        % % 
        % % plot(rel_positions, mean_disengage, '-o', 'LineWidth', 2, 'Color', [0 0.2 0.8]);
        % % 
        % % xlabel('Trial position relative to block switch');
        % % ylabel('Normalized disengagement time (ms)');
        % % title('Disengagement around block switches (Stim & NoStim, mean-centered)');
        % % grid on;
        % % hold off;
        % % 
        % % % Save
        % % saveas(gcf, fullfile(save_dir, 'BlockSwitch_Dt_Normalized.png'));
        % % saveas(gcf, fullfile(save_dir, 'BlockSwitch_Dt_Normalized.fig'));
        % % 
        % 
        % % %% Histogram of block swtich disengagement times
        % % % Identify block switch trial numbers
        % % block_switch_mask = mod(Stim_Trial_Track, 10) == 1;
        % % 
        % % % Get disengagement times
        % % block_disengage = Stim_disengage(block_switch_mask);
        % % nonblock_disengage = Stim_disengage(~block_switch_mask);
        % % 
        % % % Plot histograms
        % % figure;
        % % hold on;
        % % histogram(nonblock_disengage, 'BinWidth', 2, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'none', 'DisplayName', 'Other Trials');
        % % histogram(block_disengage, 'BinWidth', 2, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'none', 'DisplayName', 'Block Switch Trials');
        % % 
        % % xlabel('Disengagement time (ms)');
        % % ylabel('Count');
        % % title('Disengagement Time Distribution: Block Switch vs Other Stim Trials');
        % % legend('show');
        % % grid on;
        % % hold off;
        % % 
        % % % Save
        % % saveas(gcf, fullfile(save_dir, 'Stim_Block_vs_Other_Hist.png'));
        % % saveas(gcf, fullfile(save_dir, 'Stim_Block_vs_Other_Hist.fig'));
        % 
        % 
        % %% Align the neural activity to the disengagement times
        % Stim_d_hm = Stim_disengage / 10;
        % NoStim_d_hm = NoStim_disengage / 10;
        % 
        % Stim_Neural_Trials;  % 600 x 106 x 38
        % NoStim_Neural_Trials;  % 600 x 84 x 38
        % 
        % n_trials = size(Stim_Neural_Trials, 2);
        % n_neurons = size(Stim_Neural_Trials, 3);
        % window_size = 151;
        % pre_points = 100;
        % post_points = 50;
        % 
        % Stim_d_aligned = nan(window_size, n_trials, n_neurons);
        % 
        % for i = 1:n_trials
        %     align_point = round(Stim_d_hm(i)) + 100;  % add offset
        %     start_idx = align_point - pre_points;
        %     end_idx = align_point + post_points;
        % 
        %     % Initialize a temporary window with NaNs
        %     temp_window = nan(window_size, n_neurons);
        % 
        %     % Compute actual indices within bounds
        %     valid_start = max(start_idx, 1);
        %     valid_end = min(end_idx, size(Stim_Neural_Trials, 1));
        % 
        %     % Indices in temp_window to place valid data
        %     insert_start = valid_start - start_idx + 1;
        %     insert_end = insert_start + (valid_end - valid_start);
        % 
        %     % Only copy data if there's an overlapping window
        %     if insert_start <= window_size && insert_end <= window_size
        %         temp_window(insert_start:insert_end, :) = ...
        %             Stim_Neural_Trials(valid_start:valid_end, i, :);
        %     end
        % 
        %     Stim_d_aligned(:, i, :) = temp_window;
        % end
        % 
        % 
        % n_trials = size(NoStim_Neural_Trials, 2);
        % n_neurons = size(NoStim_Neural_Trials, 3);
        % window_size = 151;
        % pre_points = 100;
        % post_points = 50;
        % 
        % NoStim_d_aligned = nan(window_size, n_trials, n_neurons);
        % 
        % for i = 1:n_trials
        %     align_point = round(NoStim_d_hm(i)) + 100;  % add offset
        %     start_idx = align_point - pre_points;
        %     end_idx = align_point + post_points;
        % 
        %     % Initialize a temporary window with NaNs
        %     temp_window = nan(window_size, n_neurons);
        % 
        %     % Compute actual indices within bounds
        %     valid_start = max(start_idx, 1);
        %     valid_end = min(end_idx, size(NoStim_Neural_Trials, 1));
        % 
        %     % Indices in temp_window to place valid data
        %     insert_start = valid_start - start_idx + 1;
        %     insert_end = insert_start + (valid_end - valid_start);
        % 
        %     % Only copy data if there's an overlapping window
        %     if insert_start <= window_size && insert_end <= window_size
        %         temp_window(insert_start:insert_end, :) = ...
        %             NoStim_Neural_Trials(valid_start:valid_end, i, :);
        %     end
        % 
        %     NoStim_d_aligned(:, i, :) = temp_window;
        % end
        % 
        % 
        % 
        % %% Align activity to the relevant lick time (L1 or L4)
        % 
        % n_trials = size(Stim_Neural_Trials, 2);
        % n_neurons = size(Stim_Neural_Trials, 3);
        % window_size = 101;
        % pre_points = 50;
        % post_points = 50;
        % 
        % Stim_L1_aligned = nan(window_size, n_trials, n_neurons);
        % 
        % for i = 1:n_trials
        %     align_point = round(Stim_Lick1(i)) + 100;  % add offset
        %     start_idx = align_point - pre_points;
        %     end_idx = align_point + post_points;
        % 
        %     % Check bounds
        %     if start_idx >= 1 && end_idx <= size(Stim_Neural_Trials, 1)
        %         Stim_L1_aligned(:, i, :) = Stim_Neural_Trials(start_idx:end_idx, i, :);
        %     else
        %         warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
        %     end
        % end
        % 
        % 
        % n_trials = size(NoStim_Neural_Trials, 2);
        % n_neurons = size(NoStim_Neural_Trials, 3);
        % window_size = 101;
        % pre_points = 50;
        % post_points = 50;
        % 
        % NoStim_L4_aligned = nan(window_size, n_trials, n_neurons);
        % 
        % for i = 1:n_trials
        %     align_point = round(NoStim_Lick4(i)) + 100;  % add offset
        %     start_idx = align_point - pre_points;
        %     end_idx = align_point + post_points;
        % 
        %     % Check bounds
        %     if start_idx >= 1 && end_idx <= size(NoStim_Neural_Trials, 1)
        %         NoStim_L4_aligned(:, i, :) = NoStim_Neural_Trials(start_idx:end_idx, i, :);
        %     else
        %         warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
        %     end
        % end
        % 
        % 
        % 
        % 
        % %% Align activity to the go cue
        % 
        % n_trials = size(Stim_Neural_Trials, 2);
        % n_neurons = size(Stim_Neural_Trials, 3);
        % window_size = 101;
        % pre_points = 50;
        % post_points = 50;
        % 
        % Stim_GC_aligned = nan(window_size, n_trials, n_neurons);
        % 
        % for i = 1:n_trials
        %     align_point = 100;  % time of GC
        %     start_idx = align_point - pre_points;
        %     end_idx = align_point + post_points;
        % 
        %     % Check bounds
        %     if start_idx >= 1 && end_idx <= size(Stim_Neural_Trials, 1)
        %         Stim_GC_aligned(:, i, :) = Stim_Neural_Trials(start_idx:end_idx, i, :);
        %     else
        %         warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
        %     end
        % end
        % 
        % 
        % n_trials = size(NoStim_Neural_Trials, 2);
        % n_neurons = size(NoStim_Neural_Trials, 3);
        % window_size = 101;
        % pre_points = 50;
        % post_points = 50;
        % 
        % NoStim_GC_aligned = nan(window_size, n_trials, n_neurons);
        % 
        % for i = 1:n_trials
        %     align_point = 100;  % time of GC
        %     start_idx = align_point - pre_points;
        %     end_idx = align_point + post_points;
        % 
        %     % Check bounds
        %     if start_idx >= 1 && end_idx <= size(NoStim_Neural_Trials, 1)
        %         NoStim_GC_aligned(:, i, :) = NoStim_Neural_Trials(start_idx:end_idx, i, :);
        %     else
        %         warning('Trial %d out of bounds (start: %d, end: %d)', i, start_idx, end_idx);
        %     end
        % end
        % 
        % %% Neural Activity Relevant Lick aligned
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % Stim_PSTH = squeeze(mean(Stim_L1_aligned, 2));  % average across neurons
        % imagesc(Stim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Neuron');
        % title('Stim Trial Averaged: Lick 1 Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '-r', 'LineWidth', 1);     
        % % text(51, 0, 'Dt', ...
        % %     'HorizontalAlignment', 'center', ...
        % %     'VerticalAlignment', 'bottom', ...
        % %     'FontSize', 12, ...
        % %     'Color', 'k');
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
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_PSTH_L1_Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_PSTH_L1_Aligned.fig'));
        % 
        % 
        % 
        % 
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % NoStim_PSTH = squeeze(mean(NoStim_L4_aligned, 2));   % average across neurons
        % imagesc(NoStim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Neuron');
        % title('NoStim Trial Averaged: Lick4 Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '-r', 'LineWidth', 2);     
        % % text(51, 0, 'Dt', ...
        % %     'HorizontalAlignment', 'center', ...
        % %     'VerticalAlignment', 'bottom', ...
        % %     'FontSize', 12, ...
        % %     'Color', 'r');
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
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_PSTH_L4_Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_PSTH_L4_Aligned.fig'));
        % 
        % 
        % %% Activity GC aligned
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % Stim_PSTH = squeeze(mean(Stim_GC_aligned, 3));  % average across trials
        % imagesc(Stim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Trial');
        % title('Stim Trial Averaged: GC Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '-r', 'LineWidth', 1);     
        % % text(51, 0, 'Dt', ...
        % %     'HorizontalAlignment', 'center', ...
        % %     'VerticalAlignment', 'bottom', ...
        % %     'FontSize', 12, ...
        % %     'Color', 'k');
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
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_TrialvsTime_GCAligned.png'));
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_TrialvsTime_GCAligned.fig'));
        % 
        % 
        % 
        % 
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % NoStim_PSTH = squeeze(mean(NoStim_GC_aligned, 3));   % average across trials
        % imagesc(NoStim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Trial');
        % title('NoStim Trial Averaged: GC Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '-r', 'LineWidth', 2);     
        % % text(51, 0, 'Dt', ...
        % %     'HorizontalAlignment', 'center', ...
        % %     'VerticalAlignment', 'bottom', ...
        % %     'FontSize', 12, ...
        % %     'Color', 'r');
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
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_TrialvsTime_GCAligned.png'));
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_TrialvsTime_GCAligned.fig'));
        % 
        % 
        % 
        % %% Relevant lick aligned
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % Stim_PSTH = squeeze(mean(Stim_L1_aligned, 3));  % average across neurons
        % imagesc(Stim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Trial');
        % title('Stim Trial Averaged: Lick 1 Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '-r', 'LineWidth', 1);     
        % % text(51, 0, 'Dt', ...
        % %     'HorizontalAlignment', 'center', ...
        % %     'VerticalAlignment', 'bottom', ...
        % %     'FontSize', 12, ...
        % %     'Color', 'k');
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
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_TrialvsTime_L1Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_TrialvsTime_L1Aligned.fig'));
        % 
        % 
        % 
        % 
        % 
        % % --- Main PSTH heatmap ---
        % figure;
        % NoStim_PSTH = squeeze(mean(NoStim_L4_aligned, 3));   % average across neurons
        % imagesc(NoStim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Trial');
        % title('NoStim Trial Averaged: Lick4 Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis for time in ms
        % xticks(1:10:101);                     
        % xticklabels(-.5:.1:.5);            
        % xline(51, '-r', 'LineWidth', 2);     
        % % text(51, 0, 'Dt', ...
        % %     'HorizontalAlignment', 'center', ...
        % %     'VerticalAlignment', 'bottom', ...
        % %     'FontSize', 12, ...
        % %     'Color', 'r');
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
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_TrialvsTime_L4Aligned.png'));
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_TrialvsTime_L4Aligned.fig'));
        % 
        % %% DT aligned
        % 
        % % --- Stim Main PSTH heatmap ---
        % figure;
        % Stim_PSTH = squeeze(nanmean(Stim_d_aligned, 3));  % average across neurons
        % imagesc(Stim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Trial');
        % title('Stim Trial Averaged: Dt Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis
        % xticks(1:10:151);
        % xticklabels(-1.0:0.1:0.5);
        % xline(101, '-r', 'LineWidth', 1);  % updated center index
        % 
        % % --- Overlay population average on second axis ---
        % ax1 = gca;
        % ax1_pos = ax1.Position;
        % 
        % ax2 = axes('Position', ax1_pos, ...
        %            'Color', 'none', ...
        %            'YAxisLocation', 'right', ...
        %            'XAxisLocation', 'bottom', ...
        %            'XColor', 'k', 'YColor', 'k', ...
        %            'Box', 'off');
        % 
        % hold(ax2, 'on');
        % pop_ave_Stim = nanmean(Stim_PSTH, 2);  % 151 x 1
        % plot(ax2, 1:151, pop_ave_Stim, 'k', 'LineWidth', 2);
        % 
        % set(ax2, 'XLim', [1, 151]);
        % set(ax2, 'XTick', 1:10:151);
        % set(ax2, 'XTickLabel', []);
        % set(ax2, 'YLim', [min(pop_ave_Stim)*0.9, max(pop_ave_Stim)*1.1]);
        % set(ax2, 'YTickLabel', []);
        % y1 = ylabel("Spike Rate");
        % y1.Position(1) = y1.Position(1) - 1;
        % legend("Pop Ave");
        % 
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_TrialvsTime.png'));
        % saveas(gcf, fullfile(save_dir, 'Stim_Ave_TrialvsTime.fig'));
        % 
        % 
        % % --- NoStim Main PSTH heatmap ---
        % figure;
        % NoStim_PSTH = squeeze(nanmean(NoStim_d_aligned, 3));  % average across neurons
        % imagesc(NoStim_PSTH');
        % xlabel('Time (s)');
        % ylabel('Trial');
        % title('NoStim Trial Averaged: Dt Aligned');
        % colormap(blueToWhite)
        % colorbar;
        % set(gca, 'YDir', 'normal');
        % 
        % % Format x-axis
        % xticks(1:10:151);
        % xticklabels(-1.0:0.1:0.5);
        % xline(101, '-r', 'LineWidth', 2);
        % 
        % % --- Overlay population average on second axis ---
        % ax1 = gca;
        % ax1_pos = ax1.Position;
        % 
        % ax2 = axes('Position', ax1_pos, ...
        %            'Color', 'none', ...
        %            'YAxisLocation', 'right', ...
        %            'XAxisLocation', 'bottom', ...
        %            'XColor', 'k', 'YColor', 'k', ...
        %            'Box', 'off');
        % 
        % hold(ax2, 'on');
        % pop_ave_NoStim = nanmean(NoStim_PSTH, 2);  % 151 x 1
        % plot(ax2, 1:151, pop_ave_NoStim, 'k', 'LineWidth', 2);
        % 
        % set(ax2, 'XLim', [1, 151]);
        % set(ax2, 'XTick', 1:10:151);
        % set(ax2, 'XTickLabel', []);
        % set(ax2, 'YLim', [min(pop_ave_NoStim)*0.9, max(pop_ave_NoStim)*1.1]);
        % set(ax2, 'YTickLabel', []);
        % y1 = ylabel("Spike Rate");
        % y1.Position(1) = y1.Position(1) - 1;
        % legend("Pop Ave");
        % 
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_TrialvsTime.png'));
        % saveas(gcf, fullfile(save_dir, 'NoStim_Ave_TrialvsTime.fig'));
        % 
        % 
        % 
        % %% PC prediction R2 values
        % figure
        % PC = PC(1, 1:10);
        % bar(PC)
        % title("Neural PC Encoding R^2")
        % xlabel("Neural PC")
        % ylabel("R^2")
        % ylim([0,1])
        % saveas(gcf, fullfile(save_dir, 'PC_Encoding_R2.png'));
        % saveas(gcf, fullfile(save_dir, 'PC_Encoding_R2.fig'));
        % 
        % % %% Single trial tongue and Dt line
        % % gc_frame = 11;
        % % 
        % % Stim_dir = fullfile(save_dir, 'Stim_trials');
        % % NoStim_dir = fullfile(save_dir, 'NoStim_trials');
        % % if ~exist(Stim_dir, 'dir'), mkdir(Stim_dir); end
        % % if ~exist(NoStim_dir, 'dir'), mkdir(NoStim_dir); end
        % % 
        % % 
        % % Stim_Tongue(isnan(Stim_Tongue)) = 0;
        % % NoStim_Tongue(isnan(NoStim_Tongue)) = 0;
        % % 
        % % 
        % % % Set up subplot grid
        % % rows = 2; cols = 4;
        % % n_per_fig = rows * cols;
        % % 
        % % % --- Stim ---
        % % nStim = size(Stim_Tongue, 2);
        % % for i = 1:n_per_fig:nStim
        % %     figure;
        % %     for j = 1:min(n_per_fig, nStim - i + 1)
        % %         idx = i + j - 1;
        % %         subplot(rows, cols, j);
        % %         plot(Stim_Tongue(:, idx), 'b'); hold on;
        % %         xline(11, '--k'); % GC line
        % %         xline(Stim_disengage(idx), '--r'); % disengage line
        % %         title(['Stim Trial ' num2str(idx)]);
        % %         xlim([0 200]);
        % %     end
        % %     saveas(gcf, fullfile(Stim_dir, ['Stim_Tongue_Subplot_' num2str(i) '.png']));
        % %     close;
        % % end
        % % 
        % % % --- NoStim ---
        % % nNoStim = size(NoStim_Tongue, 2);
        % % for i = 1:n_per_fig:nNoStim
        % %     figure;
        % %     for j = 1:min(n_per_fig, nNoStim - i + 1)
        % %         idx = i + j - 1;
        % %         subplot(rows, cols, j);
        % %         plot(NoStim_Tongue(:, idx), 'r'); hold on;
        % %         xline(11, '--k'); % GC line
        % %         xline(NoStim_disengage(idx), '--b'); % disengage line
        % %         title(['NoStim Trial ' num2str(idx)]);
        % %         xlim([0 200]);
        % %     end
        % %     saveas(gcf, fullfile(NoStim_dir, ['NoStim_Tongue_Subplot_' num2str(i) '.png']));
        % %     close;
        % % end
        % % 
        % % % Stim mean and long disengagers
        % % 
        % % Stim_Tongue(isnan(Stim_Tongue)) = 0;
        % % NoStim_Tongue(isnan(NoStim_Tongue)) = 0;
        % % 
        % % Stim_median = median(Stim_disengage);
        % % Stim_std = std(Stim_disengage);
        % % 
        % % % Indices for grouping
        % % typical_idx = find(Stim_disengage <= Stim_median + Stim_std);
        % % long_idx    = find(Stim_disengage > Stim_median + Stim_std);
        % % 
        % % Stim_typical_dir = fullfile(save_dir, 'Stim_typical_trials');
        % % Stim_long_dir = fullfile(save_dir, 'Stim_long_trials');
        % % if ~exist(Stim_typical_dir, 'dir'), mkdir(Stim_typical_dir); end
        % % if ~exist(Stim_long_dir, 'dir'), mkdir(Stim_long_dir); end
        % % 
        % % plot_disengage_examples(Stim_Tongue, Stim_disengage, typical_idx, Stim_typical_dir, 'Stim_typical');
        % % plot_disengage_examples(Stim_Tongue, Stim_disengage, long_idx, Stim_long_dir, 'Stim_long');
        % % 
        % % 
        % % % %% NoStim mean and long disengagers
        % % % % Indices for grouping
        % % % typical_idx = find(NoStim_disengage <= NoStim_median + NoStim_std);
        % % % long_idx    = find(NoStim_disengage > NoStim_median + NoStim_std);
        % % % 
        % % % NoStim_typical_dir = fullfile(save_dir, 'NoStim_typical_trials');
        % % % NoStim_long_dir = fullfile(save_dir, 'NoStim_long_trials');
        % % % if ~exist(NoStim_typical_dir, 'dir'), mkdir(Stim_typical_dir); end
        % % % if ~exist(NoStim_long_dir, 'dir'), mkdir(Stim_long_dir); end
        % % 
        % % 
        % 
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
        % 
        % 
        % %% Box plot of Lick Bout Durations
        % % Convert durations to milliseconds
        % engaged_ms = (engaged_durations' *10);
        % disengaged_ms = (disengaged_durations' *10);
        % 
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
        % 
        % 
        % %% Histogram of lick lengths
        % figure;
        % all_ms = [engaged_ms; disengaged_ms];
        % histogram(engaged_ms, 'FaceColor', red_side, 'FaceAlpha', 0.5, 'DisplayName', 'Engaged', 'BinWidth',10);
        % hold on
        % histogram(disengaged_ms, 'FaceColor', blue_side, 'FaceAlpha', 0.5, 'DisplayName', 'Disengaged', 'BinWidth',10);
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
    %     disp(save_dir)
    %     close all;
    % end


end

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


