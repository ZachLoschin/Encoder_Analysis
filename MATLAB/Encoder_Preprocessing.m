% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% March 2025
% Cleaning up preprocessing file for MC engagement analysis

%% Notes for processing

% obj.me for each element has 1xTimepoints of the trial. It looks like this
% is sampled at the same rate as the video data, so 1/400. This could be
% used to extract the trials from the video data features.

% Bruh of course this has one more timepoint than the SVD figures fml
% Looks like I will have to loop through everything and find which one has
% a mismatch and go from there so that I don't shift the trials while
% cutting them up


%% Finding "Kinematic Modes"
clear,clc

d = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Economo-Lab-Preprocessing';
addpath(d)
addpath(genpath(fullfile(d,'utils')))
addpath(genpath(fullfile(d,'zutils')))
addpath(genpath(fullfile(d,'DataLoadingScripts')))
addpath(genpath(fullfile(d,'funcs')))
addpath("C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\disengagement\ObjVis")
%% PARAMETERS
params.alignEvent          = 'goCue'; % 'fourthLick' 'goCue'  'moveOnset'  'firstLick' 'thirdLick' 'lastLick' 'reward'

% time warping only operates on neural data for now.
params.behav_only          = 0;
params.timeWarp            = 0;  % piecewise linear time warping - each lick duration on each trial gets warped to median lick duration for that lick across trials
params.nLicks              = 20; % number of post go cue licks to calculate median lick duration for and warp individual trials to

params.lowFR               = 1; % remove clusters with firing rates across all trials less than this val

params.condition(1) = {'hit==1 | hit==0' };    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1' };    % left to right         % right hits, no stim, aw off

% Take this big window for z-scoring to baseline
params.tmin = -2;
params.tmax = 4;
params.dt = 1/200;
pre_gc_points = -params.tmin / params.dt;

% smooth with causal gaussian kernel
params.smooth = 10;

% cluster qualities to use
params.quality = {'good','fair','excellent'}; % accepts any cell array of strings - special character 'all' returns clusters of any quality

params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_nostril','bottom_nostril'}};
params.feat_varToExplain = 99;  % num factors for dim reduction of video features should explain this much variance
params.N_varToExplain = 80;     % keep num dims that explains this much variance in neural data (when doing n/p)
params.advance_movement = 0;

% Params for finding kinematic modes
params.fcut = 10;          % smoothing cutoff frequency
params.cond = 5;         % which conditions to use to find mode
params.method = 'xcorr';   % 'xcorr' or 'regress' (basically the same)
params.fa = false;         % if true, reduces neural dimensions to 10 with factor analysis
params.bctype = 'reflect'; % options are : reflect  zeropad  none

%% SPECIFY DATA TO LOAD
datapth = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\processed sessions\r14';
meta = [];
meta = loadTD(meta,datapth);
params.probe = {meta.probe}; 

%% LOAD DATA
[obj,params] = loadSessionData(meta,params,params.behav_only);
% [obj,params] = loadSessionData(meta,params);

for sessix = 1:numel(meta)
    me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
end

%% Check if video features for SVD have same lengths in obj.me
folderPath = 'C:\Research\Encoder_Modeling\TD13d\2024-11-12\cam1';
[frameCounts, mismatchFlags] = checkVideoFrameCounts(folderPath, obj);


%% Import the SVD feature matrix
load('C:\Research\Encoder_Modeling\Video_Features\TD13d\2024-11-12\SVD_Features_Cam0_TD13d_2024_11_12.mat');


%%
% Get each trial of SVD_features using the framcounts
trialFeatures = cell(length(frameCounts), 1);
t = [];
startIdx = 1;
for i = 1:length(frameCounts)
    endIdx = startIdx + frameCounts(i) - 1;
    trialFeatures{i} = SVD_features(startIdx:endIdx, :);
    t = [t, size(trialFeatures{i}, 1) / 400];
    startIdx = endIdx + 1;
end


%% Get kinematic data
%---------------------------------------------
% kin (struct array) - one entry per session
%---------------------------------------------
nSessions = numel(meta);
for sessix = 1:numel(meta)
    message = strcat('----Getting kinematic data for session',{' '},num2str(sessix), {' '},'out of',{' '},num2str(nSessions),'----');
    disp(message)
    kin(sessix) = getKinematics(obj(sessix), me(sessix), params(sessix));
end

% clearvars -except kin meta obj params


%% -- Extract All Trial Tongue Lengths -- %%
conds2use = [1];
condtrix = params(sessix).trialid{conds2use};                 % Get the trials from this condition
kinfeat = 'tongue_length';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1
sessix = 1;
kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
all_length = kin.dat(:,condtrix,kinix);

% trim off first second because its before the go cue
all_length2 = all_length(pre_gc_points-100+1:end, :);

%% -- Extract All Trial Lick Port Contacts -- %%
all_contacts = obj.bp.ev.lickL;

for i = 1:obj.bp.Ntrials
    contacts = obj.bp.ev.lickL{i, 1};
    gc = obj.bp.ev.goCue(i);
    contacts = contacts - gc;
    all_contacts{i} = contacts;
end

%% -- Filter Tongue Length and LP Contacts by Trial Type -- %%
% Get trial IDs
R1_Trials = params.trialid{8};
R4_Trials = params.trialid{9};

% Filter Tongue Length
% Remove first second of data because its before gc
R1_Tongue = all_length(pre_gc_points-200+1:end, R1_Trials);  % 100 before gc to end of trial (600 points)
R4_Tongue = all_length(pre_gc_points-200+1:end, R4_Trials);  % 100 before gc to end of trial (600 points)

% Filter LP Contacts
R1_Contacts = all_contacts(R1_Trials);  % these contacts are relative to the gc
R4_Contacts = all_contacts(R4_Trials);  % these contacts are relative to the gc

%% Get SVD features by trial
R1_Features = trialFeatures(R1_Trials);
R4_Features = trialFeatures(R4_Trials);

R1_Times = {obj.traj{1,1}(R1_Trials).frameTimes};
R4_Times = {obj.traj{1,1}(R4_Trials).frameTimes};

gc = obj.bp.ev.goCue(:);
R1_GC = gc(R1_Trials);
R4_GC = gc(R4_Trials);


% Remove trials where the GC is past the end of the video
[R1_Valid_Trials, R1_Valid_Features, R1_Valid_Times, R1_Valid_GC] = ...
    filter_valid_trials(R1_Trials, R1_Features, R1_Times, R1_GC, 0.49);

[R4_Valid_Trials, R4_Valid_Features, R4_Valid_Times, R4_Valid_GC] = ...
    filter_valid_trials(R4_Trials, R4_Features, R4_Times, R4_GC, 0.49);

% Now pass the valid trials to the feature extraction function
R1_Trial_Features = extract_trial_features(R1_Valid_Features, R1_Valid_Times, R1_Valid_GC, 399, 2000);
R4_Trial_Features = extract_trial_features(R4_Valid_Features, R4_Valid_Times, R4_Valid_GC, 399, 2000);

%% Get into format for storage
% Assuming R1_Trial_Features and R4_Trial_Features are cell arrays of trial matrices
% Each matrix in the cell array has dimensions 2400 x 200 (timepoints x features)

% Initialize empty arrays to store the reshaped features
reshaped_R1_Trial_Features = [];
reshaped_R4_Trial_Features = [];

% Loop through the trials in R1_Trial_Features and R4_Trial_Features
for i = 1:length(R1_Trial_Features)
    % Reshape each trial matrix into (2400*trials) x 200 by concatenating across the time dimension
    reshaped_R1_Trial_Features = [reshaped_R1_Trial_Features; R1_Trial_Features{i}];
end

for i =1:length(R4_Trial_Features)
    reshaped_R4_Trial_Features = [reshaped_R4_Trial_Features; R4_Trial_Features{i}];
end


%% -- Get Region Specific Neural Data and Filter It-- %%
% Tudor code for separating probe 1 and 2
Ncells = size(obj.psth, 2);
probe1 = 1:numel(params.cluid{1, 1});
probe2 = size(params.cluid{1, 1},1)+1:Ncells;

probe1_trialdat = obj.trialdat(:,probe1, :);
probe2_trialdat = obj.trialdat(:,probe2, :);

% Limit to the desired trials
probe1_R4 = probe1_trialdat(:,:,R4_Valid_Trials);  % -2 to 5s 200Hz
probe2_R4 = probe2_trialdat(:,:,R4_Valid_Trials);

probe1_R1 = probe1_trialdat(:,:,R1_Valid_Trials);
probe2_R1 = probe2_trialdat(:,:,R1_Valid_Trials);

% %% Normalize the neural data to the baseline period
probe1_R4 = zscore_pregc(probe1_R4, pre_gc_points);
probe1_R1 = zscore_pregc(probe1_R1, pre_gc_points);
probe2_R4 = zscore_pregc(probe2_R4, pre_gc_points);
probe2_R1 = zscore_pregc(probe2_R1, pre_gc_points);

%% Get into format for storage
% timepoints x neurons x trials
% puts into timepoines x trials x neurons and then reshapes
concatenated_probe1_R4 = reshape(permute(probe1_R4, [1, 3, 2]), [], size(probe1_R4, 2));
concatenated_probe1_R1 = reshape(permute(probe1_R1, [1, 3, 2]), [], size(probe1_R1, 2));
concatenated_probe2_R4 = reshape(permute(probe2_R4, [1, 3, 2]), [], size(probe2_R4, 2));
concatenated_probe2_R1 = reshape(permute(probe2_R1, [1, 3, 2]), [], size(probe2_R1, 2));


%% -- Construct the Output File -- %%
sessionName = meta.anm;
sessionDate = meta.date;

% Construct the output folder path
outputFolder = fullfile( ...
    'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder', ...
    [sessionName '_' sessionDate]);

% Create the output folder if it does not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Save R4 Files
csvwrite(fullfile(outputFolder, "Probe1_R4.csv"), concatenated_probe1_R4);
csvwrite(fullfile(outputFolder, "Probe2_R4.csv"), concatenated_probe2_R4);
csvwrite(fullfile(outputFolder, "R4_Features.csv"), reshaped_R4_Trial_Features);


% Save R1 Files
csvwrite(fullfile(outputFolder, "Probe1_R1.csv"), concatenated_probe1_R1);
csvwrite(fullfile(outputFolder, "Probe2_R1.csv"), concatenated_probe2_R1);
csvwrite(fullfile(outputFolder, "R1_Features.csv"), reshaped_R1_Trial_Features);


% Save metadata as a .txt file for record-keeping
metadataFile = fullfile(outputFolder, 'metadata.txt');
fid = fopen(metadataFile, 'w');
fprintf(fid, 'Processing Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Script Name: %s\n', mfilename('fullpath'));
fprintf(fid, 'Session ID: %s\n', sessionName);
fprintf(fid, 'Session Date: %s\n', sessionDate);


%% Function to normalize trials across all k
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


function zscored_data = zscore_pregc(data, pre_gc_points)
    data_pregc = data(1:800,:,:);
    % Concatenate trials along the second dimension (time dimension)
    concatenated_data = reshape(data_pregc, [], size(data_pregc, 2));
    
    % Compute mean and standard deviation for each neuron
    neuron_means = mean(concatenated_data, 1);
    neuron_stds = std(concatenated_data, 0, 1);
    
    % Z-score the full dataset
    zscored_data = (data - neuron_means ./ ...
                        neuron_stds);

    % chop the data to the gc to end
    zscored_data = zscored_data(pre_gc_points-400+1:end, :, :);
end
