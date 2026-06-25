%% -- Mutual Information Analysis -- %%
% % MI will detect the condition identity, which presumably is detected from
% % a state switch in that region brought on by the reward in R14 task.
% MI_P1 = calculate_MI(probe1_R1, probe1_R4);
% MI_P2 = calculate_MI(probe2_R1, probe2_R4);
% MI_P3 = calculate_MI(probe3_R1, probe3_R4);
% 
% %%
% % Mean MI across neurons
% figure;
% plot(mean(MI_P1, 2))
% hold on
% plot(mean(MI_P2, 2))
% plot(mean(MI_P3, 2))
% xline(SR, '--', 'Trial Start') % or wherever your event markers are
% xlabel('Time (samples)')
% ylabel('MI (bits)')
% legend("Probe 1", "Probe 2", "Probe 3")
% title('Probe 1 - Mutual Information over time')
% 
% % Mean Firing Rate for each condition + MI
% figure;
% yyaxis left
% plot(P1R1); hold on;
% plot(P1R4);
% ylabel('Mean Firing Rate')
% 
% yyaxis right
% plot(mean(MI_P1, 2));
% ylabel('Mutual Information (bits)')
% 
% legend("R1 Firing Rate", "R4 Firing Rate", "Mutual Information")
% xlabel("Time (samples)")
% 
% %%
% [MI_P1, MI_P1_shuffle] = calculate_MI(probe1_R1, probe1_R4);
% 
% MI_P1_shuffle_mean = mean(MI_P1_shuffle, 3); % mean over shuffles
% 
% %%
% [MI_P2, MI_P2_shuffle] = calculate_MI(probe2_R1, probe2_R4);
% 
% MI_P2_shuffle_mean = mean(MI_P2_shuffle, 3); % mean over shuffles
% 
% 
% %%
% % figure;
% % yyaxis left
% % plot(P1R1); hold on;
% % plot(P1R4);
% % ylabel('Mean Firing Rate')
% % 
% % yyaxis right
% % plot(mean(MI_P1, 2));
% % plot(mean(MI_P1_shuffle_mean, 2), '--');
% % ylabel('Mutual Information (bits)')
% % 
% % legend("R1 Firing Rate", "R4 Firing Rate", "MI", "MI Shuffle")
% % xlabel("Time (samples)")
% 
% figure;
% 
% ax1 = subplot(2, 1, 1)
% sgtitle('Motor Cortex Mutual Information')
% plot(P1R1); hold on;
% plot(P1R4);
% ylabel('Mean Firing Rate')
% legend('R1', 'R4')
% xlabel('Time (samples)')
% 
% ax2 = subplot(2, 1, 2)
% plot(mean(MI_P1, 2)); hold on;
% plot(mean(MI_P1_shuffle_mean, 2), '--');
% ylabel('Mutual Information (bits)')
% legend('MI', 'MI Shuffle')
% xlabel('Time (samples)')
% linkaxes([ax1, ax2], 'x')
% 
% 
% figure;
% 
% ax3 = subplot(2, 1, 1)
% sgtitle('SNr Mutual Information')
% title("SNr MI")
% plot(P2R1); hold on;
% plot(P2R4);
% ylabel('Mean Firing Rate')
% legend('R1', 'R4')
% xlabel('Time (samples)')
% 
% ax4 = subplot(2, 1, 2)
% plot(mean(MI_P2, 2)); hold on;
% plot(mean(MI_P2_shuffle_mean, 2), '--');
% ylabel('Mutual Information (bits)')
% legend('MI', 'MI Shuffle')
% xlabel('Time (samples)')
% linkaxes([ax3, ax4], 'x')
% 
% %% Single Trial Plots
% trial = 1;
% plot_trial_MI(probe1_R1, probe1_R4, MI_P1, MI_P1_shuffle, trial, 'R1', SR, 'Motor Cortex')
% plot_trial_MI(probe2_R1, probe2_R4, MI_P2, MI_P2_shuffle, trial, 'R4', SR, 'SNr')
% 
% %% Encoding latency
% min_consecutive = 5;
% latency_P1 = find_encoding_latency(MI_P1, MI_P1_shuffle, min_consecutive);
% latency_P2 = find_encoding_latency(MI_P2, MI_P2_shuffle, min_consecutive);
% 
% % Remove neurons that never crossed threshold
% latency_P1_valid = latency_P1(~isnan(latency_P1));
% latency_P2_valid = latency_P2(~isnan(latency_P2));
% 
% fprintf('Motor Cortex: %d / %d neurons reached threshold\n', length(latency_P1_valid), length(latency_P1));
% fprintf('SNr:          %d / %d neurons reached threshold\n', length(latency_P2_valid), length(latency_P2));
% 
% % Convert to ms if needed
% % latency_P1_valid = latency_P1_valid / SR * 1000;
% % latency_P2_valid = latency_P2_valid / SR * 1000;
% 
% figure;
% subplot(1, 3, 1)
% histogram(latency_P1_valid, 20, 'FaceAlpha', 0.6); hold on;
% histogram(latency_P2_valid, 20, 'FaceAlpha', 0.6);
% legend('Motor Cortex', 'SNr')
% xlabel('Latency (samples)')
% ylabel('Neuron count')
% title('Encoding Latency Distribution')
% 
% subplot(1, 3, 2)
% cdfplot(latency_P1_valid); hold on;
% cdfplot(latency_P2_valid);
% legend('Motor Cortex', 'SNr')
% xlabel('Latency (samples)')
% ylabel('Cumulative fraction')
% title('CDF of Encoding Latency')
% 
% subplot(1, 3, 3)
% boxplot([latency_P1_valid, latency_P2_valid], ...
%     [ones(1, length(latency_P1_valid)), 2*ones(1, length(latency_P2_valid))], ...
%     'Labels', {'Motor Cortex', 'SNr'})
% ylabel('Latency (samples)')
% title('Encoding Latency Boxplot')
% 
% % Statistical comparison
% MI_mean_P1 = mean(MI_P1, 2); % timepoints x 1
% MI_mean_P2 = mean(MI_P2, 2); % timepoints x 1
% 
% [p, h] = signrank(MI_mean_P1, MI_mean_P2);
% fprintf('Wilcoxon rank-sum test: p = %.4f\n', p)

%% Function definitions
function [MI, MI_shuffle] = calculate_MI(probe_R1, probe_R2, num_bins, num_shuffles)
% calculate_MI - Calculates mutual information at each timepoint and neuron
%
% Inputs:
%   probe_R1     - timepoints x neurons x trials array for condition 1
%   probe_R2     - timepoints x neurons x trials array for condition 2
%   num_bins     - number of bins for firing rate histogram (default: 20)
%   num_shuffles - number of shuffle iterations for null distribution (default: 100)
%
% Outputs:
%   MI         - timepoints x neurons array of mutual information (bits)
%   MI_shuffle - timepoints x neurons x num_shuffles array of null MI values

if nargin < 3
    num_bins = 20;
end
if nargin < 4
    num_shuffles = 50;
end

[num_timepoints, num_neurons, NumR1] = size(probe_R1);
[~, ~, NumR2] = size(probe_R2);
w1 = NumR1 / (NumR1 + NumR2);
w2 = NumR2 / (NumR1 + NumR2);

MI           = zeros(num_timepoints, num_neurons);
MI_shuffle   = zeros(num_timepoints, num_neurons, num_shuffles);

for t = 1:num_timepoints
    if mod(t, 50) == 0
        fprintf('Processing timepoint %d / %d\n', t, num_timepoints);
    end

    for n = 1:num_neurons
        r1    = squeeze(probe_R1(t, n, :));
        r2    = squeeze(probe_R2(t, n, :));
        all_r = [r1; r2];

        edges = linspace(min(all_r), max(all_r), num_bins + 1);

        % Real MI
        MI(t, n) = compute_MI_from_responses(r1, r2, edges, w1, w2);

        % Shuffle MI — randomly reassign trial labels each iteration
        for s = 1:num_shuffles
            shuffled    = all_r(randperm(length(all_r)));
            r1_shuf     = shuffled(1:NumR1);
            r2_shuf     = shuffled(NumR1+1:end);
            MI_shuffle(t, n, s) = compute_MI_from_responses(r1_shuf, r2_shuf, edges, w1, w2);
        end
    end
end
end

function MI = compute_MI_from_responses(r1, r2, edges, w1, w2)
    all_r      = [r1; r2];
    counts_all = histcounts(all_r, edges, 'Normalization', 'probability');
    H_response = -sum(counts_all(counts_all > 0) .* log2(counts_all(counts_all > 0)));

    counts_r1  = histcounts(r1, edges, 'Normalization', 'probability');
    counts_r2  = histcounts(r2, edges, 'Normalization', 'probability');
    H_r1       = -sum(counts_r1(counts_r1 > 0) .* log2(counts_r1(counts_r1 > 0)));
    H_r2       = -sum(counts_r2(counts_r2 > 0) .* log2(counts_r2(counts_r2 > 0)));

    MI = H_response - (w1 * H_r1 + w2 * H_r2);
end

function plot_trial_MI(probe_R1, probe_R2, MI, MI_shuffle, trial_num, condition, SR, title_str)
% plot_trial_MI - Plots single trial firing rate and MI with 95% shuffle threshold
%
% Inputs:
%   probe_R1    - timepoints x neurons x trials for condition 1
%   probe_R2    - timepoints x neurons x trials for condition 2
%   MI          - timepoints x neurons MI array
%   MI_shuffle  - timepoints x neurons x shuffles null distribution
%   trial_num   - which trial to plot
%   condition   - 'R1' or 'R4'
%   SR          - sample rate (for trial start marker)
%   title_str   - string for sgtitle

if strcmp(condition, 'R1')
    trial_FR = mean(probe_R1(:, :, trial_num), 2);
else
    trial_FR = mean(probe_R2(:, :, trial_num), 2);
end

% 95th percentile of shuffle at each timepoint (averaged over neurons)
shuffle_95   = prctile(mean(MI_shuffle, 2), 95, 3);
MI_mean      = mean(MI, 2);
above_thresh = MI_mean > shuffle_95;

figure;
sgtitle(title_str)

ax1 = subplot(2, 1, 1);
plot(trial_FR);
xline(SR, 'k--', 'Trial Start');
ylabel('Mean Firing Rate')
xlabel('Time (samples)')
title(sprintf('Single Trial Firing Rate - %s Trial %d', condition, trial_num))

ax2 = subplot(2, 1, 2);
hold on;

% Plot full MI trace in blue
plot(MI_mean, 'b');

% Overlay above-threshold segments in red
MI_above = nan(size(MI_mean));
MI_above(above_thresh) = MI_mean(above_thresh);
plot(MI_above, 'r', 'LineWidth', 2);

plot(shuffle_95, 'k--');
xline(SR, 'k--', 'Trial Start');
ylabel('Mutual Information (bits)')
xlabel('Time (samples)')
legend('MI', 'Above Threshold', '95% Shuffle')
title('Population MI with Threshold')

linkaxes([ax1, ax2], 'x')
end

function latency = find_encoding_latency(MI, MI_shuffle, min_consecutive)
% find_encoding_latency - Finds first timepoint of sustained MI above 95% shuffle
%
% Inputs:
%   MI              - timepoints x neurons
%   MI_shuffle      - timepoints x neurons x shuffles
%   min_consecutive - number of consecutive points required (default: 5)
%
% Outputs:
%   latency - 1 x neurons array of latencies (NaN if never crosses threshold)

if nargin < 3
    min_consecutive = 5;
end

num_neurons  = size(MI, 2);
num_timepoints = size(MI, 1);
latency      = nan(1, num_neurons);

shuffle_95   = prctile(MI_shuffle, 95, 3); % timepoints x neurons

for n = 1:num_neurons
    above_thresh = MI(:, n) > shuffle_95(:, n);
    for t = 1:num_timepoints - min_consecutive
        if all(above_thresh(t:t + min_consecutive - 1))
            latency(n) = t;
            break
        end
    end
end
end