function trial_Features = extract_trial_features(Features, Times, GC, pre_frames, post_frames)
    % extract_trial_features extracts trial features around the Go Cue (GC)
    % Inputs:
    %   Features - Cell array where each cell contains a matrix of time x features
    %   Times - Cell array where each cell contains a vector of timepoints (in seconds)
    %   GC - Vector of Go Cue times (in seconds)
    %   pre_frames - Number of frames before the GC to extract
    %   post_frames - Number of frames after the GC to extract
    %
    % Outputs:
    %   trial_Features - Cell array of matrices containing the trial features for each trial,
    %                    each matrix has (pre_frames + post_frames + 1) x features.

    % Initialize cell array to store trial features
    trial_Features = cell(1, length(Features)); 

    % Loop over all trials
    for i = 1:length(Features)
        % Get the current trial's feature matrix and corresponding times
        trial_mat = Features{i};
        time_mat = Times{i};

        % Subtract 0.49 from each element in the time matrix
        adjusted_times = time_mat - 0.49;

        % Find the closest time to the corresponding GC time
        [~, idx] = min(abs(adjusted_times - GC(i)));

        % Get the start and end indices based on the specified frames
        start_idx = idx - pre_frames;
        end_idx = idx + post_frames;

        % Check for valid indices (to avoid out of bounds)
        if start_idx < 1
            start_idx = 1;
            end_idx = pre_frames + post_frames + 1;  % Adjust to include valid frames
        end
        if end_idx > length(time_mat)
            end_idx = length(time_mat);
            start_idx = end_idx - (pre_frames + post_frames);  % Adjust to include valid frames
        end

        % Extract the trial features around the go cue
        trial_Features{i} = trial_mat(start_idx:end_idx, :);
    end
end
