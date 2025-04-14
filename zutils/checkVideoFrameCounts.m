function [frameCounts, mismatchFlags] = checkVideoFrameCounts(folderPath, obj)
    % checkVideoFrameCounts
    % Compare video frame counts in AVI files with data in obj.me
    %
    % Inputs:
    %   folderPath - string, path to the folder containing .avi files
    %   obj        - struct, must contain field 'me' (cell array)
    %
    % Outputs:
    %   frameCounts   - vector of frame counts (either matching or video frame count if mismatch)
    %   mismatchFlags - vector of flags (1 if mismatch, 0 if match)
    
    % Get list of all .avi files
    aviFiles = dir(fullfile(folderPath, '*.avi'));

    % Check if number of files matches number of trials
    if length(aviFiles) ~= length(obj.me)
        warning('Number of AVI files (%d) does not match number of trials in obj.me (%d)', length(aviFiles), length(obj.me));
    end

    % Preallocate result vectors
    numTrials = length(obj.me);
    frameCounts = zeros(numTrials, 1);
    mismatchFlags = zeros(numTrials, 1);

    % Loop and compare frames for each trial
    for i = 1:numTrials
        % Get corresponding video file
        videoFile = fullfile(folderPath, aviFiles(i).name);
        
        % Read video metadata
        v = VideoReader(videoFile);
        frameCount = floor(v.Duration * v.FrameRate);  % Fast method

        % Get frame count from obj.me
        dataFrameCount = size(obj.me{i}, 2);
        
        % Compare and store results
        if frameCount ~= dataFrameCount
            fprintf('Mismatch at trial %d (%s): Video frames = %d, obj.me frames = %d\n', ...
                i, aviFiles(i).name, frameCount, dataFrameCount);
            
            frameCounts(i) = frameCount;
            mismatchFlags(i) = 1;
        else
            frameCounts(i) = dataFrameCount;
            mismatchFlags(i) = 0;
        end
    end
end
