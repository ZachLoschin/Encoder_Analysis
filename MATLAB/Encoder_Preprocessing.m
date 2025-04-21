% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% March 2025
% Cleaning up preprocessing file for MC engagement analysis


%% -- CHECK THESE ASPECTS BEFORE RUNNING -- %%
resp = input("Did you check that the files in loadTD line up with SVD files?");


%% Finding "Kinematic Modes"
clear,clc

d = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Economo-Lab-Preprocessing';
addpath(d)
addpath(genpath(fullfile(d,'utils')))
addpath(genpath(fullfile(d,'zutils')))
addpath(genpath(fullfile(d,'DataLoadingScripts')))
addpath(genpath(fullfile(d,'ObjVis')))
addpath(genpath(fullfile(d,'funcs')))
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
params.tmax = 5;
params.dt = 1/100;
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
folderPath = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\video_data\TD15d\2024-11-26\cam1';
[frameCounts, mismatchFlags] = checkVideoFrameCounts(folderPath, obj);


%% Import the Facemap object and extract SVD features
% Load facemap processed files
SVD_feats_cam0_struct = load('C:\Research\Encoder_Modeling\Encoder_Analysis\Data\SVD_features\FaceMap_Processed\TD15d_11_26\Cam0_TD15d_2024-11-26_cam_0_date_2024_11_26_time_15_55_47_v001_proc.mat');
% SVD_feats_cam1_struct = load('C:\Research\Encoder_Modeling\Encoder_Analysis\Data\SVD_features\FaceMap_Processed\Cam1_TD13d_2024-11-12_cam_1_date_2024_11_12_time_17_49_00_v001_proc.mat');

% Extract features
SVD_Feats_cam0 = [SVD_feats_cam0_struct.motSVD_0(:,1:50), SVD_feats_cam0_struct.movSVD_0(:,1:50)];
% SVD_Feats_cam1 = [SVD_feats_cam1_struct.motSVD_0(:,1:50), SVD_feats_cam1_struct.movSVD_0(:,1:50)];
% SVD_features = [SVD_Feats_cam0, SVD_Feats_cam1];

SVD_features = SVD_Feats_cam0;

clear SVD_feats_cam0_struct
% clear SVD_feats_cam1_struct


%% Get each trial of SVD_features using the framcounts
cam_framerate = 400;
trialFeatures = cell(length(frameCounts), 1);
t = [];
startIdx = 1;

for i = 1:length(frameCounts)
    disp(i)
    endIdx = startIdx + frameCounts(i) - 1;

    % Handle off-by-one issue if endIdx exceeds SVD_features
    if endIdx > size(SVD_features, 1)
        warning('Frame count for trial %d exceeds available SVD data. Trimming last frame.', i);
        endIdx = size(SVD_features, 1);
    end

    % Check if slice is one longer than it should be
    actualLength = endIdx - startIdx + 1;
    expectedLength = frameCounts(i);
    if actualLength > expectedLength
        endIdx = endIdx - 1;  % Trim the last frame
        warning('Trial %d: Trimming last frame to match frameCounts.', i);
    end

    trialFeatures{i} = SVD_features(startIdx:endIdx, :);
    t = [t, size(trialFeatures{i}, 1) / cam_framerate];
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
    pos = getKeypointsFromVideo(obj(sessix),params(sessix));
end

% clearvars -except kin meta obj params


%% -- Extract All Trial Tongue Lengths -- %%
conds2use = [1];
condtrix = params(sessix).trialid{conds2use};                 % Get the trials from this condition
kinfeat = 'tongue_length';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1
sessix = 1;
kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
all_length = kin.dat(:,condtrix,kinix);

%% -- CHECK THAT THE CHOPPING PROCEDURE HAS THE RIGHT UNITS -- %%
%  -- The contacts are relative to the GC correct? -- %
%  -- Check that if this is true, chops are being handled correctly -- %

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
R1_Tongue = all_length((pre_gc_points-100+1):end, R1_Trials);  % -1s through 4s (500 points)
R4_Tongue = all_length((pre_gc_points-100+1):end, R4_Trials);

% Filter LP Contacts
R1_Contacts = all_contacts(R1_Trials);  % these contacts are relative to the gc
R4_Contacts = all_contacts(R4_Trials);  % these contacts are relative to the gc

%% Find trial FCs, Last Relevant Contacts (LRCs), and Trials2Remove
[trials2removeR1, FCs_R1, LRCs_R1] = filter_trials_by_licking(R1_Contacts, min_licks=3);
[trials2removeR4, FCs_R4, LRCs_R4] = filter_trials_by_licking(R4_Contacts, min_licks=5);

% This trials2remove are indices into R1 and R4_Contacts, not trial numbers

%% Get SVD features by trial
R1_Features = trialFeatures(R1_Trials);
R4_Features = trialFeatures(R4_Trials);

R1_Times = {obj.traj{1,1}(R1_Trials).frameTimes};
R4_Times = {obj.traj{1,1}(R4_Trials).frameTimes};

gc = obj.bp.ev.goCue(:);
R1_GC = gc(R1_Trials);
R4_GC = gc(R4_Trials);

vid_offset = 0.49;  % time axis offset btwn video and spike glx

% Find invalid trials where the GC is after the end of the video
[R1_Valid_Trials, ~, ~, ~] = ...
    filter_valid_trials(R1_Trials, R1_Features, R1_Times, R1_GC, vid_offset);

[R4_Valid_Trials, ~, ~, ~] = ...
    filter_valid_trials(R4_Trials, R4_Features, R4_Times, R4_GC, vid_offset);

% These are also trial indices into RN_Trials, not trial numbers

%% Filter the trials by valid from SVD analysis and removal from contacts
final_R1_trials = setdiff(R1_Valid_Trials, trials2removeR1);
final_R4_trials = setdiff(R4_Valid_Trials, trials2removeR4);

FCs_R1_clean = FCs_R1(final_R1_trials);
LRCs_R1_clean = LRCs_R1(final_R1_trials);

FCs_R4_clean = FCs_R4(final_R4_trials);
LRCs_R4_clean = LRCs_R4(final_R4_trials);

R1_Valid_Features = R1_Features(final_R1_trials);
R1_Valid_Times = R1_Times(final_R1_trials);
R1_Valid_GC = R1_GC(final_R1_trials);

R4_Valid_Features = R4_Features(final_R4_trials);
R4_Valid_Times = R4_Times(final_R4_trials);
R4_Valid_GC = R4_GC(final_R4_trials);

%% Now pass the valid trials to the feature extraction function
pregc_frames = (400 * 1) -1;
postgc_frames = (400 * 5);
R1_Trial_Features = extract_trial_features(R1_Valid_Features, R1_Valid_Times, R1_Valid_GC, pregc_frames, postgc_frames);
R4_Trial_Features = extract_trial_features(R4_Valid_Features, R4_Valid_Times, R4_Valid_GC, pregc_frames, postgc_frames);


%% Get into format for storage
% Assuming R1_Trial_Features and R4_Trial_Features are cell arrays of trial matrices
% Each matrix in the cell array has dimensions 2000 x 100 (timepoints x features)

% Initialize empty arrays to store the reshaped features
R1_SVD_Features_Uncut = [];
R4_SVD_Features_Uncut = [];

for i = 1:length(R1_Trial_Features)
    % Resample time axis from 400Hz to 100Hz (i.e., 1/4th the number of timepoints)
    resampled = resample(R1_Trial_Features{i}, 1, 4);  % downsample each trial
    R1_SVD_Features_Uncut = [R1_SVD_Features_Uncut; resampled];
end

for i = 1:length(R4_Trial_Features)
    resampled = resample(R4_Trial_Features{i}, 1, 4);
    R4_SVD_Features_Uncut = [R4_SVD_Features_Uncut; resampled];
end


% Convert cell array to 3D matrix
R1_SVD_Features_mat = cat(3, R1_Trial_Features{:});
R4_SVD_Features_mat = cat(3, R4_Trial_Features{:});

%% Get the cut features too

% Preallocate with cell because resampled lengths may vary
R1_SVD_Features_resampled = cell(1, size(R1_SVD_Features_mat, 3));
R4_SVD_Features_resampled = cell(1, size(R4_SVD_Features_mat, 3));

for i = 1:size(R1_SVD_Features_mat, 3)
    trial = R1_SVD_Features_mat(:, :, i);  % [T x D]
    resampled = resample(trial, 1, 4);     % downsample time axis
    R1_SVD_Features_resampled{i} = resampled;
end

for i = 1:size(R4_SVD_Features_mat, 3)
    trial = R4_SVD_Features_mat(:, :, i);
    resampled = resample(trial, 1, 4);
    R4_SVD_Features_resampled{i} = resampled;
end

% Convert back to matrix
R1_SVD_Features_mat_resampled = cat(3, R1_SVD_Features_resampled{:});
R4_SVD_Features_mat_resampled = cat(3, R4_SVD_Features_resampled{:});

R1_SVD_Features_Cut = chop_and_stack_neural_data(R1_SVD_Features_mat_resampled, LRCs_R1_clean, 100);
R4_SVD_Features_Cut = chop_and_stack_neural_data(R4_SVD_Features_mat_resampled, LRCs_R4_clean, 100);


% %% NORMALIZATION
% % Concatenate across time and trials: reshape to [T*N x D]
% R1_all = reshape(R1_SVD_Features_mat_resampled, [], size(R1_SVD_Features_mat_resampled, 2));
% R4_all = reshape(R4_SVD_Features_mat_resampled, [], size(R4_SVD_Features_mat_resampled, 2));
% 
% % Compute global z-score parameters (shared across R1 & R4)
% all_features = [R1_all; R4_all];  % [total_timepoints_across_all_trials x D]
% feature_means = mean(all_features, 1);
% feature_stds = std(all_features, 0, 1);
% feature_stds(feature_stds == 0) = 1;  % avoid divide-by-zero
% 
% 
% % Apply z-scoring
% R1_SVD_Features_mat_z = zscore_trials(R1_SVD_Features_mat_resampled, feature_means, feature_stds);
% R4_SVD_Features_mat_z = zscore_trials(R4_SVD_Features_mat_resampled, feature_means, feature_stds);
% 
% R1_SVD_Features_Cut = chop_and_stack_neural_data(R1_SVD_Features_mat_z, LRCs_R1_clean, 100);
% R4_SVD_Features_Cut = chop_and_stack_neural_data(R4_SVD_Features_mat_z, LRCs_R4_clean, 100);
% 
% 
% % All trials and timepoints from R1 and R4 stacked
% R1_all = reshape(R1_SVD_Features_mat_resampled, [], size(R1_SVD_Features_mat_resampled, 2));
% R4_all = reshape(R4_SVD_Features_mat_resampled, [], size(R4_SVD_Features_mat_resampled, 2));
% 
% % Compute global z-score stats
% all_features = [R1_all; R4_all];
% feature_means = mean(all_features, 1);
% feature_stds = std(all_features, 0, 1);
% feature_stds(feature_stds == 0) = 1;  % safeguard
% 
% % Normalize the uncut features
% R1_SVD_Features_Uncut = (R1_SVD_Features_Uncut - feature_means) ./ feature_stds;
% R4_SVD_Features_Uncut = (R4_SVD_Features_Uncut - feature_means) ./ feature_stds;


%% Get the keypoints by trial
R1_Keypoints_unfiltered = pos(101:end, :, R1_Trials);
R4_Keypoints_unfiltered = pos(101:end, :, R4_Trials);

R1_Keypoints = R1_Keypoints_unfiltered(:, :, final_R1_trials);
R4_Keypoints = R4_Keypoints_unfiltered(:, :, final_R4_trials);

% Reshape into matrix for storage
R1_Keypoints_Uncut = reshape(permute(R1_Keypoints, [1, 3, 2]), [], size(R1_Keypoints, 2));
R4_Keypoints_Uncut = reshape(permute(R4_Keypoints, [1, 3, 2]), [], size(R4_Keypoints, 2));

% Chop trials to LRCs for Cut storage
R1_Keypoints_Cut = chop_and_stack_neural_data(R1_Keypoints, LRCs_R1_clean, 100);
R4_Keypoints_Cut = chop_and_stack_neural_data(R4_Keypoints, LRCs_R4_clean, 100);

%% -- Get Region Specific Neural Data and Filter It-- %%
% Tudor code for separating probe 1 and 2
Ncells = size(obj.psth, 2);
probe1 = 1:numel(params.cluid{1, 1});
probe2 = size(params.cluid{1, 1},1)+1:Ncells;

probe1_trialdat = obj.trialdat(:,probe1, :);
probe2_trialdat = obj.trialdat(:,probe2, :);

% Limit to the desired trials
% This is done in multiple rouds since the final_RN_trials are indices into
% the RN_trials, not absolute trial numbers
probe1_R4 = probe1_trialdat(:,:,R4_Trials);  % -2 to 4s 100Hz
probe2_R4 = probe2_trialdat(:,:,R4_Trials);

probe1_R4 = probe1_R4(:,:,final_R4_trials);
probe2_R4 = probe2_R4(:,:,final_R4_trials);

probe1_R1 = probe1_trialdat(:,:,R1_Trials);
probe2_R1 = probe2_trialdat(:,:,R1_Trials);

probe1_R1 = probe1_R1(:,:,final_R1_trials);
probe2_R1 = probe2_R1(:,:,final_R1_trials);

%% Normalize the neural data to the baseline period
probe1_R4_norm = zscore_pregc(probe1_R4, pre_gc_points);  % -1 to 5s 100Hz
probe1_R1_norm = zscore_pregc(probe1_R1, pre_gc_points);
probe2_R4_norm = zscore_pregc(probe2_R4, pre_gc_points);
probe2_R1_norm = zscore_pregc(probe2_R1, pre_gc_points);

%% Get into uncut format for storage
% timepoints x neurons x trials
% puts into timepoines x trials x neurons and then reshapes
probe1_R4_Uncut = reshape(permute(probe1_R4_norm, [1, 3, 2]), [], size(probe1_R4_norm, 2));
probe1_R1_Uncut = reshape(permute(probe1_R1_norm, [1, 3, 2]), [], size(probe1_R1_norm, 2));
probe2_R4_Uncut = reshape(permute(probe2_R4_norm, [1, 3, 2]), [], size(probe2_R4_norm, 2));
probe2_R1_Uncut = reshape(permute(probe2_R1_norm, [1, 3, 2]), [], size(probe2_R1_norm, 2));

%% Chop up the neural data to relevant trial ends
% Handles LRCs relative to GC if neural data has 1s pre GC stored
probe1_R4_Cut = chop_and_stack_neural_data(probe1_R4_norm, LRCs_R4_clean, 100);
probe1_R1_Cut = chop_and_stack_neural_data(probe1_R1_norm, LRCs_R1_clean, 100);

probe2_R4_Cut = chop_and_stack_neural_data(probe2_R4_norm, LRCs_R4_clean, 100);
probe2_R1_Cut = chop_and_stack_neural_data(probe2_R1_norm, LRCs_R1_clean, 100);

%% Neural PCs
% Get -1s -> 5s data
probe1 = probe1_trialdat(101:end, :, :);
probe2 = probe2_trialdat(101:end, :, :);

[num_timepoints1, ~, num_trials1] = size(probe1);
[num_timepoints2, ~, num_trials2] = size(probe2);

% Reshape to (timepoints x trials) x features
probe1_PCA = reshape(permute(probe1, [1, 3, 2]), [], size(probe1, 2));
probe2_PCA = reshape(permute(probe2, [1, 3, 2]), [], size(probe2, 2));

% PCA on probe1
num_PCs = 12;
[coeff1, score1, latent1, tsquared1, explained1, mu1] = pca(probe1_PCA);
score1 = score1(:, 1:num_PCs);
score1_reshaped = reshape(score1, num_timepoints1, num_trials1, num_PCs);

% PCA on probe2
[coeff2, score2, latent2, tsquared2, explained2, mu2] = pca(probe2_PCA);
score2 = score2(:, 1:num_PCs);
score2_reshaped = reshape(score2, num_timepoints2, num_trials2, num_PCs);

% Separate PCs by trial type and filter by valid trials
Probe1_PCs_R1 = score1_reshaped(:, R1_Trials, :);
Probe1_PCs_R4 = score1_reshaped(:, R4_Trials, :);
Probe2_PCs_R1 = score2_reshaped(:, R1_Trials, :);
Probe2_PCs_R4 = score2_reshaped(:, R4_Trials, :);

Probe1_PCs_R1 = Probe1_PCs_R1(:, final_R1_trials, :);
Probe1_PCs_R4 = Probe1_PCs_R4(:, final_R4_trials, :);
Probe2_PCs_R1 = Probe2_PCs_R1(:, final_R1_trials, :);
Probe2_PCs_R4 = Probe2_PCs_R4(:, final_R4_trials, :);

% Reshape into matrices for Uncut storage
Probe1_PCs_R1_Uncut = reshape(Probe1_PCs_R1, [], size(Probe1_PCs_R1, 3));
Probe1_PCs_R4_Uncut = reshape(Probe1_PCs_R4, [], size(Probe1_PCs_R4, 3));
Probe2_PCs_R1_Uncut = reshape(Probe2_PCs_R1, [], size(Probe2_PCs_R1, 3));
Probe2_PCs_R4_Uncut = reshape(Probe2_PCs_R4, [], size(Probe2_PCs_R4, 3));

% Reshape to input into chopping procedure
Probe1_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R4, [1, 3, 2]), LRCs_R4_clean, 100);
Probe2_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R4, [1, 3, 2]), LRCs_R4_clean, 100);
Probe1_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R1, [1, 3, 2]), LRCs_R1_clean, 100);
Probe2_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R1, [1, 3, 2]), LRCs_R1_clean, 100);

%% Get the tongue length, FLCs, and LRCs times to save
SR = 100;
% Filter Tongue Length
R1_Tongue = all_length(pre_gc_points-100+1:end, R1_Trials);  % -1s through 4s (500 points)
R4_Tongue = all_length(pre_gc_points-100+1:end, R4_Trials);

R1_Tongue_Uncut = R1_Tongue(:,final_R1_trials);
R4_Tongue_Uncut = R4_Tongue(:,final_R4_trials);

FCs_Adj_R1 = ceil(FCs_R1_clean*SR + SR);
FCs_Adj_R4 = ceil(FCs_R4_clean*SR + SR);

LRCs_Adj_R1 = ceil(LRCs_R1_clean*SR + SR);
LRCs_Adj_R4 = ceil(LRCs_R4_clean*SR + SR);


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

% Save SVD Features
csvwrite(fullfile(outputFolder, "SVD_Feats_R1_Uncut.csv"), R1_SVD_Features_Uncut);
csvwrite(fullfile(outputFolder, "SVD_Feats_R4_Uncut.csv"), R4_SVD_Features_Uncut);
csvwrite(fullfile(outputFolder, "SVD_Feats_R1_Cut.csv"), R1_SVD_Features_Cut);
csvwrite(fullfile(outputFolder, "SVD_Feats_R4_Cut.csv"), R4_SVD_Features_Cut);

% Save key point features
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Uncut.csv"), R1_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Uncut.csv"), R4_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Cut.csv"), R1_Keypoints_Cut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Cut.csv"), R4_Keypoints_Cut);

% Save Neural FRs
csvwrite(fullfile(outputFolder, "Probe1_R1_Uncut.csv"), probe1_R1_Uncut);
csvwrite(fullfile(outputFolder, "Probe1_R4_Uncut.csv"), probe1_R4_Uncut);
csvwrite(fullfile(outputFolder, "Probe1_R1_Cut.csv"), probe1_R1_Cut);
csvwrite(fullfile(outputFolder, "Probe1_R4_Cut.csv"), probe1_R4_Cut);


csvwrite(fullfile(outputFolder, "Probe2_R1_Uncut.csv"), probe2_R1_Uncut);
csvwrite(fullfile(outputFolder, "Probe2_R4_Uncut.csv"), probe2_R4_Uncut);
csvwrite(fullfile(outputFolder, "Probe2_R1_Cut.csv"), probe2_R1_Cut);
csvwrite(fullfile(outputFolder, "Probe2_R4_Cut.csv"), probe2_R4_Cut);

% Save Neural PCs
csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Uncut.csv"), Probe1_PCs_R1_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Uncut.csv"), Probe1_PCs_R4_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Cut.csv"), Probe1_PCs_R1_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Cut.csv"), Probe1_PCs_R4_Cut);


csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Uncut.csv"), Probe2_PCs_R1_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Uncut.csv"), Probe2_PCs_R4_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Cut.csv"), Probe2_PCs_R1_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Cut.csv"), Probe2_PCs_R4_Cut);

% Save Tongue Length for visualizations
csvwrite(fullfile(outputFolder, "Tongue_R1.csv"), R1_Tongue_Uncut);
csvwrite(fullfile(outputFolder, "Tongue_R4.csv"), R4_Tongue_Uncut);

% Save FCs and LRCs
csvwrite(fullfile(outputFolder, "FCs_R1.csv"), FCs_Adj_R1);
csvwrite(fullfile(outputFolder, "FCs_R4.csv"), FCs_Adj_R4);

csvwrite(fullfile(outputFolder, "LRCs_R1.csv"), LRCs_Adj_R1);
csvwrite(fullfile(outputFolder, "LRCs_R4.csv"), LRCs_Adj_R4);

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
    data_pregc = data(1:200,:,:);
    % Concatenate trials along the second dimension (time dimension)
    concatenated_data = reshape(data_pregc, [], size(data_pregc, 2));
    
    % Compute mean and standard deviation for each neuron
    neuron_means = mean(concatenated_data, 1);
    neuron_stds = std(concatenated_data, 0, 1);
    
    % Z-score the full dataset
    zscored_data = (data - neuron_means ./ ...
                        neuron_stds);

    % chop the data to the gc-1s to end
    zscored_data = zscored_data(pre_gc_points-100+1:end, :, :);
end

% Function to apply z-score normalization to each trial
function normalized = zscore_trials(data, means, stds)
    % data: [T x D x N]
    sz = size(data);
    normalized = (data - reshape(means, 1, [], 1)) ./ reshape(stds, 1, [], 1);
end
