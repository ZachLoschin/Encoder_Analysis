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
% base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\R14_2026_SNR\';
% alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R14_2026_SNR';
% 
% base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\TD10sa';
% alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R14_2026';

base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\TDsa12_ShankFiltered';
alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\TDsa12';


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
        
       %  R1_Path = alt_session_dir + "\Probe" + prb + "_R1_Uncut.csv";
       %  R4_Path = alt_session_dir + "\Probe" + prb + "_R4_Uncut.csv";
       % 
       %  % Read the matrices
       %  R1_Neural = readmatrix(R1_Path);
       %  R4_Neural = readmatrix(R4_Path);
       % 
       %  [x, xx] = size(R1_Neural);
       % 
       %  % Reshape into trials
       %  % T x N_trials x N_neurons
        SR_neural = 1400;
        pregc = 200;
       % 
       %  % SR_neural = 600;
        % % pregc = 100;
       % 
       %  R1_Neural_Trials = reshape(R1_Neural, SR_neural, [], xx);  % size: 600 x 95 x 54
       %  R1_Neural_Trials = permute(R1_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
       %  % R1_Neural_Trials = zscore_pregc(R1_Neural_Trials, 100);
       % 
       %  R4_Neural_Trials = reshape(R4_Neural, SR_neural, [], xx);  % size: 600 x 95 x 54
       %  R4_Neural_Trials = permute(R4_Neural_Trials, [1,2,3]);  % size: 95 x 600 x 54
       %  % R4_Neural_Trials = zscore_pregc(R4_Neural_Trials, 100);
       % 
        % Import relevant lick contact times
        R1_Lick1_Path = alt_session_dir + "\FCs_R1.csv";
        R4_Lick4_Path = alt_session_dir + "\Fourth_C_R4.csv";

        FCs_R1 = readmatrix(fullfile(alt_session_dir, 'FCs_R1.csv')) - pregc;
        FCs_R4 = readmatrix(fullfile(alt_session_dir, 'FCs_R4.csv'))- pregc;

        % - 100 takes out pre gc period. Now times are relative to GC
        R1_Lick1 = readmatrix(R1_Lick1_Path) - pregc;
        % R4_Lick4 = readmatrix(R4_Lick4_Path) - pregc;
       % 
       % 
       %  %% Plot PSTHs for visualization
       % % Average across trials
       %  R1_PSTH = squeeze(mean(R1_Neural_Trials, 3));  % size: 600 x 54
       % 
       %  % Transpose for heatmap: neurons on Y, time on X
       %  imagesc(R1_PSTH');  % now size is xx x 600
       %  xlabel('Time (samples)');
       %  ylabel('Neuron');
       %  title('R1 PSTH (Trial-Averaged)');
       %  colorbar;
       % 
       %  % Optional: improve visualization
       %  set(gca, 'YDir', 'normal');
       % 
       % 
       %  %% Average across trials
       %  R4_PSTH = squeeze(mean(R4_Neural_Trials, 3));  % size: 600 x 54
       %  % R4_PSTH = R4_PSTH(90:200, :);
       % 
       %  % Transpose for heatmap: neurons on Y, time on X
       %  imagesc(R4_PSTH');  % now size is xx x 600
       %  xlabel('Time (samples)');
       %  ylabel('Neuron');
       %  title('R4 PSTH (Trial-Averaged)');
       %  colorbar;
       % 
       %  % Optional: improve visualization
       %  set(gca, 'YDir', 'normal');
    
        
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
        % Initialize the figure
        GCtime = 11;  % This is 11 because I am only saving 50ms before gc
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
        
        [trs, timepoints] = size(All_States);

        % Overlay heatmap
        h = imagesc(1:size(All_States, 2), 1:size(All_States, 1), All_States);
        
        % Plotting line plots
        for j = 1:size(R4_Tongue, 2)
            plot(1:length(R4_Tongue(:, j)), j-1 + R4_Tongue(:, j), R4_color, 'LineWidth', lw);
        end
        
        for j = 1:size(R1_Tongue, 2)
            plot(1:length(R1_Tongue(:, j)), j-1 + size(R4_Tongue, 2) + R1_Tongue(:, j), R1_color, 'LineWidth', lw);
        end
            
        
        % Adjust the colormap to suit your data range
        colormap(custom_cmap);  % You can use 'jet', 'hot', 'cool', or any other colormap
        % colormap(linspecer);  % Flip the colormap if needed
        
        % Adjust the heatmap properties
        % set(h, 'AlphaData', 0.5); % Adjust transparency to see the line plots underneath
        
        % Add a colorbar
        cc = colorbar;
        
        % Add a horizontal line to separate the conditions
        yline(size(R4_Tongue, 2), 'k-', 'LineWidth', 3);
        
        % Add labels or annotations
        text(-30, size(R4_Tongue, 2) + 2, 'R1', 'FontSize', 12, 'Color', "k");
        text(-30, size(R4_Tongue, 2) - 2, 'R4', 'FontSize', 12, 'Color', "k");
        
        % Set axes limits and labels
        set(gca, 'YTick', 0:10:trs);
        xlabel('Time (s)');
        ylabel('Trial Number');
        
        % % Adjust the x-axis ticks and labels to reflect time in seconds
        % xticks([0 60 110 160 210 260 310 360 410 460 510]); % Position of ticks
        % xticklabels({'-0.1', '0.5', '1', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0'}); % Labels corresponding to time in seconds
        xticks([0 60 110 160 210 260 310 360 410]); % Position of ticks
        xticklabels({'-0.05', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0'}); % Labels corresponding to time in seconds
        nTrials = size(All_States, 1);
        ylim([0 nTrials]);  % Number of trials
        
        
        % Remove grid lines and tighten the axes
        box off;
        axis tight;
        
        % Customize ticks and labels to match your style
        set(gca, 'TickLength', [0 0]);
        % xlim([0 210])
        % Hold off to finish the plot
        hold off;
        
        title("Single Trial State Estimates: R1 and R4");
        xline(GCtime, '--k', 'LineWidth', 1);  % Add vertical line for GC
        text(GCtime, -5, 'GC', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', 'k');
        saveas(gcf, fullfile(save_dir, 'Inference_Heatmap.png'));  % Save current figure as png
        saveas(gcf, fullfile(save_dir, 'Inference_Heatmap.fig'));  % Save as MATLAB .fig file
        
        
        
        %% Trial averaged inference plots
        lb = [0 0.76 1];

        % Trial averaged inference plots
        R1_Inf_Mean = mean(exp(R1_States'), 1);
        R1_Inf_Std  = std(exp(R1_States'), 0, 1);
        
        R4_Inf_Mean = mean(exp(R4_States'), 1);
        R4_Inf_Std  = std(exp(R4_States'), 0, 1);
        
        x = 1:length(R1_Inf_Mean);
        
        figure
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
        plot(x, R1_Inf_Mean, 'b', 'LineWidth', 2)
        plot(x, R4_Inf_Mean, 'Color', [0 0.76 1], 'LineWidth', 2)
        
        ylabel("State 1 Probability")
        xlabel("Time (s)")
        % xticks([0 60 110 160 210]); % Position of ticks
        xticks([0 60 110 160 210 260 310 360 410])
        % xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'}); % Labels corresponding to time in seconds
        xticklabels({'-0.05', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0'});
        % xlim([0, 200])
        ylim([0, 1])
        title("Trial Averaged Inference")
        
        xline(GCtime, "--k", "LineWidth", 1)
        text(GCtime, min([R1_Inf_Mean - R1_Inf_Std, R4_Inf_Mean - R4_Inf_Std], [], 'all') - 0.02, 'GC', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 10);
        
        legend(["R1 Std", "R4 Std", "R1 Mean", "R4 Mean"])
        saveas(gcf, fullfile(save_dir, 'Ave_Inference.png'))
        saveas(gcf, fullfile(save_dir, 'Ave_Inference.fig'))

        %% Trial averaged inference aligned to FC

        % Parameters
        % window_size = 211;
        % pre_points = 10;
        % post_points = 200;

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
        
        % ---- R4 ----
        n_trials = size(R4_States, 2);
        R4_FC_aligned = nan(window_size, n_trials);
        
        for i = 1:n_trials
            align_point = round(FCs_R4(i));
            start_idx = align_point - pre_points;
            end_idx = align_point + post_points;
        
            valid_start = max(start_idx, 1);
            valid_end = min(end_idx, size(R4_States, 1));
        
            insert_start = valid_start - start_idx + 1;
            insert_end = insert_start + (valid_end - valid_start);
        
            R4_FC_aligned(insert_start:insert_end, i) = R4_States(valid_start:valid_end, i);
        end

        


        % Trial averaged inference plots
        R1_Inf_Mean = nanmean(exp(R1_FC_aligned'), 1);
        R1_Inf_Std  = nanstd(exp(R1_FC_aligned'), 0, 1);
        
        R4_Inf_Mean = nanmean(exp(R4_FC_aligned'), 1);
        R4_Inf_Std  = nanstd(exp(R4_FC_aligned'), 0, 1);

        % Replace NaNs in mean and std with 0.0
        R1_Inf_Mean(isnan(R1_Inf_Mean)) = 0.0;
        R1_Inf_Std(isnan(R1_Inf_Std)) = 0.0;
        
        R4_Inf_Mean(isnan(R4_Inf_Mean)) = 0.0;
        R4_Inf_Std(isnan(R4_Inf_Std)) = 0.0;

        
        x = 1:length(R1_Inf_Mean);
        
        figure
        hold on

        % Clamp shaded areas between 0 and 1
        R1_Shade_Upper = min(R1_Inf_Mean + R1_Inf_Std, 1);
        R1_Shade_Lower = max(R1_Inf_Mean - R1_Inf_Std, 0);
        
        R4_Shade_Upper = min(R4_Inf_Mean + R4_Inf_Std, 1);
        R4_Shade_Lower = max(R4_Inf_Mean - R4_Inf_Std, 0);

        
       % Shaded std region for R1
        r1_s = fill([x, fliplr(x)], ...
             [R1_Shade_Upper, fliplr(R1_Shade_Lower)], ...
             [0.6 0.8 1], ...
             'EdgeColor', 'none', ...
             'FaceAlpha', 0.7);
        
        % Shaded std region for R4
        r4_s = fill([x, fliplr(x)], ...
             [R4_Shade_Upper, fliplr(R4_Shade_Lower)], ...
             [0.7 0.9 1], ...
             'EdgeColor', 'none', ...
             'FaceAlpha', 0.7);

        
        % Plot mean traces
        r1_p = plot(x, R1_Inf_Mean, 'b', 'LineWidth', 2);
        r4_p = plot(x, R4_Inf_Mean, 'Color', [0 0.76 1], 'LineWidth', 2);
        
        ylabel("State 1 Probability")
        xlabel("Time (s)")
        xticks([0 60 110 160 210])
        % xticklabels({'-0.1', '0.5', '1.0', '1.5', '2.0'}); % Labels corresponding to time in seconds
        xticklabels({'-0.05', '0.25', '0.5', '0.75', '1.0'});
        % xlim([0, 200])
        % xlim([0, 200])
        ylim([0, 1])
        title("Trial Averaged Inference FC Aligned")
        
        xline(GCtime, "--k", "LineWidth", 1)
        text(GCtime, min([R1_Inf_Mean - R1_Inf_Std, R4_Inf_Mean - R4_Inf_Std], [], 'all') - 0.02, 'GC', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 10);
        
        legend([r1_p, r4_p, r1_s, r4_s], {'R1 Mean', 'R4 Mean', 'R1 Std', 'R4 Std'} )
        saveas(gcf, fullfile(save_dir, 'Ave_Inference_FCAligned.png'))
        saveas(gcf, fullfile(save_dir, 'Ave_Inference_FCAligned.fig'))
    
            
        %% Disengagement time analysis
        % Minus 10 here to corect for GC time being 10 points into time series.
        R1_disengage = (estimate_disengage_times(exp(R1_States'), 0.5, 4, 40) - 10)*.005;
        R4_disengage = (estimate_disengage_times(exp(R4_States'), 0.5, 4, 40) - 10)*.005;
        
        figure()
        histogram(R1_disengage)
        hold on
        histogram(R4_disengage)
        
        csvwrite(fullfile(save_dir, "R1_dt.csv"), R1_disengage);
        csvwrite(fullfile(save_dir, "R4_dt.csv"), R4_disengage);        


         % Initialize the figure
        GCtime = 11;  % This is 11 because I am only saving 50ms before gc
        R1_Tongue = R1_Tongue_norm';
        R4_Tongue = R4_Tongue_norm';
        R1_States = R1_States';
        R4_States = R4_States';
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
        [trs, timepoints] = size(All_States);
% Overlay heatmap
        h = imagesc(1:size(All_States, 2), 1:size(All_States, 1), All_States);
% Plotting line plots
        for j = 1:size(R4_Tongue, 2)
            plot(1:length(R4_Tongue(:, j)), j-1 + R4_Tongue(:, j), R4_color, 'LineWidth', lw);
        end
        for j = 1:size(R1_Tongue, 2)
            plot(1:length(R1_Tongue(:, j)), j-1 + size(R4_Tongue, 2) + R1_Tongue(:, j), R1_color, 'LineWidth', lw);
        end

% Overlay disengagement time dots
        dt_fs = 0.005;  % 200 Hz sampling interval
        nR4 = size(R4_Tongue, 2);
        nR1 = size(R1_Tongue, 2);

        R4_disengage_x = GCtime + R4_disengage / dt_fs;
        R1_disengage_x = GCtime + R1_disengage / dt_fs;

        for j = 1:nR4
            if ~isnan(R4_disengage_x(j))
                scatter(R4_disengage_x(j), j - 0.5, 20, 'w', 'filled', 'MarkerEdgeColor', 'none');
            end
        end
        for j = 1:nR1
            if ~isnan(R1_disengage_x(j))
                scatter(R1_disengage_x(j), nR4 + j - 0.5, 20, 'w', 'filled', 'MarkerEdgeColor', 'none');
            end
        end

% Adjust the colormap to suit your data range
        colormap(custom_cmap);
% Add a colorbar
        cc = colorbar;
% Add a horizontal line to separate the conditions
        yline(size(R4_Tongue, 2), 'k-', 'LineWidth', 3);
% Add labels or annotations
        text(-30, size(R4_Tongue, 2) + 2, 'R1', 'FontSize', 12, 'Color', "k");
        text(-30, size(R4_Tongue, 2) - 2, 'R4', 'FontSize', 12, 'Color', "k");
% Set axes limits and labels
        set(gca, 'YTick', 0:10:trs);
        xlabel('Time (s)');
        ylabel('Trial Number');
        xticks([0 60 110 160 210 260 310 360 410]);
        xticklabels({'-0.05', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0'});
        nTrials = size(All_States, 1);
        ylim([0 nTrials]);
        box off;
        axis tight;
        set(gca, 'TickLength', [0 0]);
        hold off;
        title("Single Trial State Estimates: R1 and R4");
        xline(GCtime, '--k', 'LineWidth', 1);
        text(GCtime, -5, 'GC', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', 'k');
        saveas(gcf, fullfile(save_dir, 'Inference_Heatmap_DT.png'));
        saveas(gcf, fullfile(save_dir, 'Inference_Heatmap_DT.fig'));


        % %% --- Single‐trial heatmap aligned to first contact (FC) + tongue overlay ---
        % % window_size = 171;    % pre_points + post_points + 1
        % % pre_points   = 20;
        % % post_points  = 150;
        % 
        % window_size = 371;    % pre_points + post_points + 1
        % pre_points   = 20;
        % post_points  = 300;
        % 
        % 
        % 
        % nR1 = size(R1_States, 2);
        % nR4 = size(R4_States, 2);
        % 
        % % align the states (shifted one sample earlier)
        % R1_states_FC = nan(window_size, nR1);
        % for i = 1:nR1
        %     ap0 = round(FCs_R1(i));
        %     ap  = ap0 - 1;                  % shift alignment one sample earlier
        %     si = ap - pre_points;
        %     ei = ap + post_points;
        %     vs = max(si,1); ve = min(ei, size(R1_States,1));
        %     ws = vs - si + 1; we = ws + (ve-vs);
        %     R1_states_FC(ws:we, i) = R1_States(vs:ve, i);
        % end
        % 
        % R4_states_FC = nan(window_size, nR4);
        % for i = 1:nR4
        %     ap0 = round(FCs_R4(i));
        %     ap  = ap0 - 1;
        %     si = ap - pre_points;
        %     ei = ap + post_points;
        %     vs = max(si,1); ve = min(ei, size(R4_States,1));
        %     ws = vs - si + 1; we = ws + (ve-vs);
        %     R4_states_FC(ws:we, i) = R4_States(vs:ve, i);
        % end
        % 
        % % align the tongue traces (same shift)
        % R1_tongue_FC = nan(window_size, nR1);
        % for i = 1:nR1
        %     ap0 = round(FCs_R1(i));
        %     ap  = ap0 - 1;
        %     si = ap - pre_points;
        %     ei = ap + post_points;
        %     vs = max(si,1); ve = min(ei, size(R1_Tongue,1));
        %     ws = vs - si + 1; we = ws + (ve-vs);
        %     R1_tongue_FC(ws:we, i) = R1_Tongue(vs:ve, i);
        % end
        % 
        % R4_tongue_FC = nan(window_size, nR4);
        % for i = 1:nR4
        %     ap0 = round(FCs_R4(i));
        %     ap  = ap0 - 1;
        %     si = ap - pre_points;
        %     ei = ap + post_points;
        %     vs = max(si,1); ve = min(ei, size(R4_Tongue,1));
        %     ws = vs - si + 1; we = ws + (ve-vs);
        %     R4_tongue_FC(ws:we, i) = R4_Tongue(vs:ve, i);
        % end
        % 
        % % Convert disengagement times (seconds) back to x-coordinates in the FC-aligned window
        % % disengage times are relative to sample 10 of the original States array,
        % % but States are FC-aligned, so x = pre_points + 1 + t_sec/0.005
        % dt_fs = 0.005;  % 200 Hz
        % 
        % R4_disengage_x = pre_points + 1 + R4_disengage / dt_fs;
        % R1_disengage_x = pre_points + 1 + R1_disengage / dt_fs;
        % 
        % 
        % % combine and exponentiate states
        % All_states_FC = exp([ R4_states_FC.' ; R1_states_FC.' ]);
        % 
        % % Initialize the figure
        % figure;
        % hold on;
        % 
        % % Plotting parameters
        % lw = 0.75;           % Line width
        % px = 75; py = 75;    % Position on screen
        % width = 700; height = 800;
        % set(gcf, 'Position', [px, py, width, height]);
        % 
        % % Define colors for the tongue traces
        % R4_color = 'k';      % Black for R4
        % R1_color = 'k';      % Black for R1
        % 
        % % Overlay heatmap
        % h = imagesc(1:window_size, 1:size(All_states_FC,1), All_states_FC);
        % 
        % % Overlay R4 tongue traces
        % for j = 1:size(R4_tongue_FC,2)
        %     plot(1:window_size, (j-1) + R4_tongue_FC(:,j), R4_color, 'LineWidth', lw);
        % end
        % 
        % % Overlay R1 tongue traces (offset by number of R4 trials)
        % for j = 1:size(R1_tongue_FC,2)
        %     plot(1:window_size, (j-1 + size(R4_tongue_FC,2)) + R1_tongue_FC(:,j), R1_color, 'LineWidth', lw);
        % end
        % 
        % 
        % % Overlay dots for R4 trials (rows 1:nR4 in the combined matrix)
        % for j = 1:nR4
        %     if ~isnan(R4_disengage_x(j))
        %         scatter(R4_disengage_x(j), j - 0.5, 20, 'w', 'filled', ...
        %             'MarkerEdgeColor', 'none');
        %     end
        % end
        % 
        % % Overlay dots for R1 trials (rows nR4+1 : end)
        % for j = 1:nR1
        %     if ~isnan(R1_disengage_x(j))
        %         scatter(R1_disengage_x(j), nR4 + j - 0.5, 20, 'w', 'filled', ...
        %             'MarkerEdgeColor', 'none');
        %     end
        % end
        % 
        % % Apply custom colormap
        % colormap(custom_cmap);
        % 
        % % Add colorbar
        % cc = colorbar;
        % 
        % % Draw horizontal separator between R4 and R1 trials
        % yline(size(R4_tongue_FC,2), 'k-', 'LineWidth', 3);
        % 
        % % Add annotations
        % text(-30, size(R4_tongue_FC,2) + 2, 'R1', 'FontSize', 12, 'Color', 'k');
        % text(-30, size(R4_tongue_FC,2) - 2, 'R4', 'FontSize', 12, 'Color', 'k');
        % 
        % % Axes labels and ticks
        % set(gca, 'YTick', 0:10:size(All_states_FC,1));
        % xlabel('Time (s)');
        % ylabel('Trial Number');
        % 
        % % X‐axis in seconds relative to FC (adjust tick positions as needed)
        % xticks([1 60 110 160 210 270 310 370]);
        % xticklabels({'-0.1','0.25','0.5','0.75','1.0', '1.25', '1.5'});
        % xlim([1 window_size]);
        % 
        % % Vertical FC line (one sample earlier)
        % xline(pre_points, '--k', 'LineWidth', 1);
        % text(pre_points, -5, 'FC', ...
        %      'HorizontalAlignment', 'center', ...
        %      'VerticalAlignment', 'bottom', ...
        %      'FontSize', 12, ...
        %      'Color', 'k');
        % 
        % % Y‐limits
        % nTrials = size(All_states_FC,1);
        % ylim([0 nTrials]);
        % 
        % % Clean up appearance
        % box off;
        % axis tight;
        % set(gca, 'TickLength', [0 0]);
        % 
        % % Title and save
        % title('Single Trial State Estimates + Tongue Traces Aligned to FC');
        % saveas(gcf, fullfile(save_dir, 'Inference_Heatmap_FC_with_Tongue.png'));
        % saveas(gcf, fullfile(save_dir, 'Inference_Heatmap_FC_with_Tongue.fig'));

       


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

