% Read in the data, construct heatmap, overlay lick kinematics
clear;
clc;
close all
%% Import the state inference and tongue data
% base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R14_ToInclude';
% alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R14';

base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R16';
alt_base_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R16';
subfolder = '';

% Get list of all subfolders in base_dir
session_dirs = dir(base_dir);
session_dirs = session_dirs([session_dirs.isdir]);  % Keep only directories
session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));  % Remove . and ..

for ij = 1:length(session_dirs)
    % try
        session_name = session_dirs(ij).name;
        save_dir = fullfile(base_dir, session_name, subfolder);
        
        % Look for the same folder name in the alternate base directory
        alt_session_dir = fullfile(alt_base_dir, session_name);
    
        if isfolder(alt_session_dir)
            fprintf('Found matching folder for %s in alt_base_dir.\n', session_name);
            % Now you can use alt_session_dir for further processing
        else
            warning('No matching folder for %s in alt_base_dir.', session_name);
        end
    
        % Load files
        R1_States   = readmatrix(fullfile(save_dir, 'R1_States_Reg.csv'));
        R4_States   = readmatrix(fullfile(save_dir, 'R14_States_Reg.csv'));
        R1_Tongue   = readmatrix(fullfile(save_dir, 'R1_Tongue_Reg.csv'));
        R4_Tongue   = readmatrix(fullfile(save_dir, 'R14_Tongue_Reg.csv'));
        PC_2State          = readmatrix(fullfile(save_dir, 'R14_PC_R2_Reg.csv'));
        
        R4_Tongue   = readmatrix(fullfile(save_dir, 'R14_Tongue_Reg.csv'));
        PC_1State         = readmatrix(fullfile(save_dir, 'Single_State_PC_Encoding_EngInit.csv'));

        % Compute the means
        mean_2state = mean(PC_2State(1:10));
        mean_1state = mean(PC_1State);
        
        % Plot the bar chart
        figure;
        bar([mean_1state, mean_2state]);
        set(gca, 'XTickLabel', {'One State Model', 'Two State Model'});
        ylabel('Mean Encoding R^2');
        title('Single- vs. Two-State Model PC Encoding');

        saveas(gcf, fullfile(save_dir, 'SingleState_PC_Comp.png'));  % Save current figure as png
        saveas(gcf, fullfile(save_dir, 'SingleState_PC_Comp.fig'));  % Save as MATLAB .fig file
    
        a = 1;

    % catch
    %     warning("Skipping")
    %     continue
    % end

end

