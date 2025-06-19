% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% May 2025
% Cleaning up preprocessing file for MC engagement analysis


%% Finding "Kinematic Modes"
clear,clc

% Check if running on desktop or laptop
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

if hostname == "DESKTOP-5JJC0TM"
    d = "C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Economo-Lab-Preprocessing";
else
    d = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Economo-Lab-Preprocessing';
end


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



% explore lowering this to aroiund 0.5
params.lowFR               = 1.0; % remove clusters with firing rates across all trials less than this val

% params.condition(1) = {'hit==1'};
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
params.smooth = 10;  % play 

% cluster qualities to use
params.quality = {'good','fair','excellent', 'ok'}; % accepts any cell array of strings - special character 'all' returns clusters of any quality

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
datapth = 'C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Data\processed sessions\r14';
meta = [];

meta = loadTD(meta,datapth);
params.probe = {meta.probe}; 

%% LOAD DATA
[obj,params] = loadSessionData(meta,params,params.behav_only);
% [obj,params] = loadSessionData(meta,params);

for sessix = 1:numel(meta)
    me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
end

%% Handle mismatch in trial number between neural and kinematic data

% Check for trial mismatch.
nTrials_neural = size(obj.traj{1}, 2);
nTrials_kinematic = size(me.data, 2);

% If there is a mismatch, cut extra me data and set obj.Ntrials to neural
% number so that later kinematic processing doesn't include blank trials
% This should fix the later problem of having to go into each subtrial and
% changing the number of trials.

if nTrials_kinematic - nTrials_neural > 0
    diff = nTrials_kinematic - nTrials_neural;
    obj.bp.Ntrials = nTrials_neural;
    me.data = me.data(:,1:(end-diff));
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
    pos = getKeypointsFromVideo(obj(sessix), me(sessix), params(sessix));
end

% clearvars -except kin meta obj params


%% -- Extract All Trial Tongue Lengths -- %%
conds2use = [1];
condtrix = params(sessix).trialid{conds2use};                 % Get the trials from this condition

diff = nTrials_kinematic - nTrials_neural;
condtrix = condtrix(1:(end-diff), :);

NTRIALS = nTrials_neural;

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


%% PSTH



%% -- Filter Tongue Length and LP Contacts by Trial Type -- %%
% Get trial IDs
R1_Trials = params.trialid{8};
R4_Trials = params.trialid{9};

R1_Trials = R1_Trials(R1_Trials <= NTRIALS);
R4_Trials = R4_Trials(R4_Trials <= NTRIALS);

R1_Trial_Track = R1_Trials;
R4_Trial_Track = R4_Trials;

% Filter Tongue Length
R1_Tongue = all_length((pre_gc_points-100+1):end, R1_Trials);  % -1s through 4s (500 points)
R4_Tongue = all_length((pre_gc_points-100+1):end, R4_Trials);

% Filter LP Contacts
R1_Contacts = all_contacts(R1_Trials);  % these contacts are relative to the gc
R4_Contacts = all_contacts(R4_Trials);  % these contacts are relative to the gc

%% Find trial FCs, Last Relevant Contacts (LRCs), and Trials2Remove
[trials2removeR1, FCs_R1_clean, SCs_R1_clean, Fourth_C_R1_clean, LRCs_R1_clean] = filter_trials_by_licking(R1_Contacts, min_licks=3);
[trials2removeR4, FCs_R4_clean, SCs_R4_clean, Fourth_C_R4_clean, LRCs_R4_clean] = filter_trials_by_licking(R4_Contacts, min_licks=5);

% This trials2remove are indices into R1 and R4_Contacts, not trial numbers

%% Filter the trials by valid from SVD analysis and removal from contacts

FCs_R1_clean(trials2removeR1) = [];
SCs_R1_clean(trials2removeR1) = [];
LRCs_R1_clean(trials2removeR1) = [];
Fourth_C_R1_clean(trials2removeR1) = [];

FCs_R4_clean(trials2removeR4) = [];
SCs_R4_clean(trials2removeR4) = [];
LRCs_R4_clean(trials2removeR4) = [];
Fourth_C_R4_clean(trials2removeR4) = [];

% Update the trial tracking lists
R1_Trial_Track(trials2removeR1) = [];
R4_Trial_Track(trials2removeR4) = [];

%% Get the keypoints by trial
R1_Keypoints = pos(101:end, :, R1_Trials);
R4_Keypoints= pos(101:end, :, R4_Trials);

R1_Keypoints(:,:,trials2removeR1) = [];
R4_Keypoints(:,:,trials2removeR4) = [];

% Reshape into matrix for storage
R1_Keypoints_Uncut = reshape(permute(R1_Keypoints, [1, 3, 2]), [], size(R1_Keypoints, 2));
R4_Keypoints_Uncut = reshape(permute(R4_Keypoints, [1, 3, 2]), [], size(R4_Keypoints, 2));

R1K = (R1_Keypoints_Uncut - mean(R1_Keypoints_Uncut, 'omitnan')) ./ std(R1_Keypoints_Uncut, 'omitnan');
R4K = (R4_Keypoints_Uncut - mean(R4_Keypoints_Uncut, 'omitnan')) ./ std(R4_Keypoints_Uncut, 'omitnan');

n_time = size(R1_Keypoints, 1);
n_keypoints = size(R1_Keypoints, 2);
n_trials = size(R1_Keypoints, 3);
% After normalization, reshape back
R1K_reshaped = reshape(R1K, n_time, n_trials, n_keypoints);
% Permute to original shape (time × keypoints × trials)
R1K_final = permute(R1K_reshaped, [1, 3, 2]);


n_time = size(R4_Keypoints, 1);
n_keypoints = size(R4_Keypoints, 2);
n_trials = size(R4_Keypoints, 3);
% After normalization, reshape back
R4K_reshaped = reshape(R4K, n_time, n_trials, n_keypoints);
% Permute to original shape (time × keypoints × trials)
R4K_final = permute(R4K_reshaped, [1, 3, 2]);

R1_Keypoints_Uncut = R1K;
R4_Keypoints_Uncut = R4K;

% Chop trials to LRCs for Cut storage
R1_Keypoints_Cut = chop_and_stack_neural_data(R1K_final, LRCs_R1_clean, 100);
R4_Keypoints_Cut = chop_and_stack_neural_data(R4K_final, LRCs_R4_clean, 100);

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

probe1_R4(:,:,trials2removeR4) =[];
probe2_R4(:,:,trials2removeR4) = [];

probe1_R1 = probe1_trialdat(:,:,R1_Trials);
probe2_R1 = probe2_trialdat(:,:,R1_Trials);

probe1_R1(:,:,trials2removeR1) = [];
probe2_R1(:,:,trials2removeR1) =[];

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

% Normalize the neural data prior to PCA
P1_PCA = (probe1_PCA - mean(probe1_PCA)) ./ std(probe1_PCA);
P2_PCA = (probe2_PCA - mean(probe2_PCA)) ./ std(probe2_PCA);

% PCA on probe1
num_PCs = 10;
[coeff1, score1, latent1, tsquared1, explained1, mu1] = pca(P1_PCA);
score1 = score1(:, 1:num_PCs);
score1_reshaped = reshape(score1, num_timepoints1, num_trials1, num_PCs);

% PCA on probe2
[coeff2, score2, latent2, tsquared2, explained2, mu2] = pca(P2_PCA);
score2 = score2(:, 1:num_PCs);
score2_reshaped = reshape(score2, num_timepoints2, num_trials2, num_PCs);

% Separate PCs by trial type and filter by valid trials
Probe1_PCs_R1 = score1_reshaped(:, R1_Trials, :);
Probe1_PCs_R4 = score1_reshaped(:, R4_Trials, :);
Probe2_PCs_R1 = score2_reshaped(:, R1_Trials, :);
Probe2_PCs_R4 = score2_reshaped(:, R4_Trials, :);

Probe1_PCs_R1(:, trials2removeR1, :) = [];
Probe2_PCs_R1(:, trials2removeR1, :) = [];
Probe1_PCs_R4(:, trials2removeR4, :) = [];
Probe2_PCs_R4(:, trials2removeR4, :) = [];

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
R1_Tongue_Uncut = all_length(pre_gc_points-100+1:end, R1_Trials);  % -1s through 4s (500 points)
R4_Tongue_Uncut = all_length(pre_gc_points-100+1:end, R4_Trials);

R1_Tongue_Uncut(:,trials2removeR1) = [];
R4_Tongue_Uncut(:,trials2removeR4) = [];

FCs_Adj_R1 = ceil(FCs_R1_clean*SR + SR);
FCs_Adj_R4 = ceil(FCs_R4_clean*SR + SR);

SCs_Adj_R1 = ceil(SCs_R1_clean*SR + SR);
SCs_Adj_R4 = ceil(SCs_R4_clean*SR + SR);

Fourth_C_Adj_R1 = ceil(Fourth_C_R1_clean*SR + SR);
Fourth_C_Adj_R4 = ceil(Fourth_C_R4_clean*SR + SR);

LRCs_Adj_R1 = ceil(LRCs_R1_clean*SR + SR);
LRCs_Adj_R4 = ceil(LRCs_R4_clean*SR + SR);

%% -- Get Jaw Data and Filter It -- %
% Get jaw data
kinfeat = 'jaw_ydisp_view1'; % Specify the kinematic feature USE Y IN REAL
% condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
jaw = kin(sessix).dat(pre_gc_points-100+1:end, condtrix, kinix); % Extract jaw length

kinfeat = 'jaw_yvel_view1';
% condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
jaw_vel = kin(sessix).dat(pre_gc_points-100+1:end, condtrix, kinix); % Extract jaw length

% Look at jaw for R4 trials then remove filtered trials
jaw_R4 = jaw(:, R4_Trials);
jaw_R4(:, trials2removeR4) = [];

jaw_vel_R4 = jaw_vel(:, R4_Trials);
jaw_vel_R4(:, trials2removeR4) = [];

jaw_R1 = jaw(:, R1_Trials);
jaw_R1(:, trials2removeR1) = [];

jaw_vel_R1 = jaw_vel(:, R1_Trials);
jaw_vel_R1(:, trials2removeR1) = [];

% Normalize the jaw data
[jaw_R4, jaw_vel_R4, jaw_R1, Jaw_vel_R1] = deal(normalize_trials(jaw_R4, "zscore"), normalize_trials(jaw_vel_R4, "zscore"), normalize_trials(jaw_R1, "zscore"), normalize_trials(jaw_vel_R1, "zscore"));


%% Create kin features
uncut_JawR1 = reshape(jaw_R1, [], 1);

uncut_JawR4 = reshape(jaw_R4, [], 1);

uncut_JawvelR1 = reshape(jaw_vel_R1, [], 1);

uncut_JawvelR4 = reshape(jaw_vel_R4, [], 1);

jawfeats_R1_Uncut = [uncut_JawR1, uncut_JawvelR1];
jawfeats_R4_Uncut = [uncut_JawR4, uncut_JawvelR4];

%% Chop up the jaw feats
% Reshape for our cutting function
jaw_R1 = reshape(jaw_R1, size(jaw_R1,1), 1, size(jaw_R1,2));  % becomes 600 x 1 x 152
jaw_R4 = reshape(jaw_R4, size(jaw_R4,1), 1, size(jaw_R4,2));

jaw_vel_R1 = reshape(jaw_vel_R1, size(jaw_vel_R1,1), 1, size(jaw_vel_R1,2));
jaw_vel_R4 = reshape(jaw_vel_R4, size(jaw_vel_R4,1), 1, size(jaw_vel_R4,2));

jaw_R1_cut = chop_and_stack_neural_data(jaw_R1, LRCs_R1_clean, 100);
jaw_vel_R1_cut = chop_and_stack_neural_data(jaw_vel_R1, LRCs_R1_clean, 100);

jaw_R4_cut = chop_and_stack_neural_data(jaw_R4, LRCs_R4_clean, 100);
jaw_vel_R4_cut = chop_and_stack_neural_data(jaw_vel_R4, LRCs_R4_clean, 100);

jawfeats_R1_Cut = [uncut_JawR1, uncut_JawvelR1];
jawfeats_R4_Cut = [uncut_JawR4, uncut_JawvelR4];

%% -- Construct the Output File -- %%
sessionName = meta.anm;
sessionDate = meta.date;

% Construct the output folder path

if hostname == "DESKTOP-5JJC0TM"
    outputFolder = fullfile(...
        'C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Preprocessed_Encoder\R14_Manual', ...
        [sessionName '_' sessionDate ]);
else
    outputFolder = fullfile( ...
        'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R14_Manual', ...
        [sessionName '_' sessionDate ]);
end

% Create the output folder if it does not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Save trial indices for each condition
csvwrite(fullfile(outputFolder, "R1_Trial_Track.csv"), R1_Trial_Track);
csvwrite(fullfile(outputFolder, "R4_Trial_Track.csv"), R4_Trial_Track);

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

% Save Jaw Feats
csvwrite(fullfile(outputFolder, "JawFeats_R1_Uncut.csv"), jawfeats_R1_Uncut);
csvwrite(fullfile(outputFolder, "JawFeats_R4_Uncut.csv"), jawfeats_R4_Uncut);

csvwrite(fullfile(outputFolder, "JawFeats_R1_Cut.csv"), jawfeats_R1_Cut);
csvwrite(fullfile(outputFolder, "JawFeats_R4_Cut.csv"), jawfeats_R4_Cut);

% Save Tongue Length for visualizations
csvwrite(fullfile(outputFolder, "Tongue_R1.csv"), R1_Tongue_Uncut);
csvwrite(fullfile(outputFolder, "Tongue_R4.csv"), R4_Tongue_Uncut);

% Save FCs and LRCs
csvwrite(fullfile(outputFolder, "FCs_R1.csv"), FCs_Adj_R1);
csvwrite(fullfile(outputFolder, "FCs_R4.csv"), FCs_Adj_R4);

csvwrite(fullfile(outputFolder, "SCs_R1.csv"), SCs_Adj_R1);
csvwrite(fullfile(outputFolder, "SCs_R4.csv"), SCs_Adj_R4);

csvwrite(fullfile(outputFolder, "Fourth_C_R1.csv"), Fourth_C_Adj_R1);
csvwrite(fullfile(outputFolder, "Fourth_C_R4.csv"), Fourth_C_Adj_R4);

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
