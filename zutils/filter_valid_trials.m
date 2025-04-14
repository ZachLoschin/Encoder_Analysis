function [valid_trials, valid_features, valid_times, valid_gc] = filter_valid_trials(trial_indices, features, times, gc, offset)
    % Initialize output arrays
    valid_trials = [];
    valid_features = {};  % Cell array to store valid feature matrices
    valid_times = {};     % Cell array to store valid times
    valid_gc = [];        % Vector to store valid Go Cue times

    % Loop through each trial
    for i = 1:length(trial_indices)
        % Get the trial's time matrix and Go Cue time
        adjusted_times = times{i} - offset;  % Adjust times by the offset
        gc_time = gc(i);
        
        % Check if the last adjusted time is before the Go Cue
        if adjusted_times(end) >= gc_time
            % If valid, store the trial
            valid_trials = [valid_trials, trial_indices(i)];
            valid_features{end+1} = features{i};  % Store the feature matrix in a cell array
            valid_times{end+1} = times{i};
            valid_gc = [valid_gc, gc_time];
        end
    end
end
