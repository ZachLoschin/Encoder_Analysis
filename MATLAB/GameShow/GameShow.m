%% Create Example PSTHs for the game show of identifying real vs. synthetic neurons
PTdat = load("spikeData_PT.mat");


plot_spike_raster_scatter(PTdat, 10, 50);



function plot_spike_raster_scatter(PTdat, neuron_idx, num_trials)
    % Get spike timings for the specified neuron (all trials)
    spike_timings = PTdat.spikeData(neuron_idx).SpkTimings;
    
    % If num_trials not specified or exceeds available trials, use all
    if nargin < 3 || num_trials > length(spike_timings)
        num_trials = length(spike_timings);
    end
    
    % Collect all spike times and corresponding trial numbers
    all_times = [];
    all_trials = [];
    
    for trial = 1:num_trials
        spike_times = spike_timings{trial};
        
        % Skip empty trials
        if isempty(spike_times)
            continue;
        end
        
        % Append spike times and trial numbers
        all_times = [all_times; spike_times(:)];
        all_trials = [all_trials; trial * ones(length(spike_times), 1)];
    end
    
    % Create figure and plot all spikes at once
    figure;
    scatter(all_times, all_trials, 10, 'k', '.', 'LineWidth', 0.5);
    
    % Format the plot
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('Trial', 'FontSize', 12);
    title(sprintf('Spike Raster - Neuron %d (%d trials)', neuron_idx, num_trials), 'FontSize', 14);
    ylim([0.5, num_trials+0.5]);
    set(gca, 'YDir', 'reverse');
    % grid on;
end
