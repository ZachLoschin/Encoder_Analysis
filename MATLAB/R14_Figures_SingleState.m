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
base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window\';
subfolder = '';

% Get list of all subfolders in base_dir
session_dirs = dir(base_dir);
session_dirs = session_dirs([session_dirs.isdir]);  % Keep only directories
session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));  % Remove . and ..

PC1_R2 = [];

for i = 1:length(session_dirs)
    session_name = session_dirs(i).name;
    save_dir = fullfile(base_dir, session_name, subfolder);
    
    % Load files
    R1_States   = readmatrix(fullfile(save_dir, 'R1_States_Reg.csv'));
    R4_States   = readmatrix(fullfile(save_dir, 'R14_States_Reg.csv'));
    R1_Tongue   = readmatrix(fullfile(save_dir, 'R1_Tongue_Reg.csv'));
    R4_Tongue   = readmatrix(fullfile(save_dir, 'R14_Tongue_Reg.csv'));
    PC          = readmatrix(fullfile(save_dir, 'R14_PC_R2_Reg.csv'));
    
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
    
    R1_Tongue = R1_Tongue_norm';
    R4_Tongue = R4_Tongue_norm';
    R1_States = R1_States';
    R4_States = R4_States';


    %% PC prediction R2 values
    figure
    PC = PC(1, 1:10);
    bar(PC)
    title("Neural PC Encoding R^2")
    xlabel("Neural PC")
    ylabel("R^2")
    ylim([0,1])
    saveas(gcf, fullfile(save_dir, 'PC_Encoding_R2.eps'));
    close all;

    PC1_R2 = [PC1_R2, PC(1)];


end

PC1_R2

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


