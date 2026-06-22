% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% May 2025
% Cleaning up preprocessing file for MC engagement analysis

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



% explore lowering this to aroiund 0.5
params.lowFR               = 1; % remove clusters with firing rates across all trials less than this val

% params.condition(1) = {'hit==1'};
params.condition(1) = {'hit==1 | hit==0' };    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'rewardedLick==1 & stimLocation==1'};
params.condition(end+1) = {'rewardedLick==1 & stimLocation==0'};

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
datapth = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\processed sessions\VTA_Stim';
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

%% -- Filter Tongue Length and LP Contacts by Trial Type -- %%
% Get trial IDs
Stim_Trials = params.trialid{2};
NoStim_Trials = params.trialid{3};
%%
Stim_Trials = Stim_Trials(Stim_Trials < NTRIALS);
NoStim_Trials = NoStim_Trials(NoStim_Trials < NTRIALS);

% Filter Tongue Length
Stim_Tongue = all_length((pre_gc_points-100+1):end, Stim_Trials);  % -1s through 4s (500 points)
NoStim_Tongue = all_length((pre_gc_points-100+1):end, NoStim_Trials);

% Filter LP Contacts
Stim_Contacts = all_contacts(Stim_Trials);  % these contacts are relative to the gc
NoStim_Contacts = all_contacts(NoStim_Trials);  % these contacts are relative to the gc

%% Find trial FCs, Last Relevant Contacts (LRCs), and Trials2Remove
[trials2removeStim, FCs_Stim_clean, SCs_Stim_clean, Fourth_C_Stim_clean, Sixth_C_Stim_clean, LRCs_Stim_clean] = filter_trials_by_licking(Stim_Contacts, min_licks=4);
[trials2removeNoStim, FCs_NoStim_clean, SCs_NoStim_clean, Fourth_C_NoStim_clean, Sixth_C_NoStim_clean, LRCs_NoStim_clean] = filter_trials_by_licking(NoStim_Contacts, min_licks=4);

% This trials2remove are indices into R1 and R4_Contacts, not trial numbers

%% Filter the trials by valid from SVD analysis and removal from contacts

FCs_Stim_clean(trials2removeStim) = [];
SCs_Stim_clean(trials2removeStim) = [];
LRCs_Stim_clean(trials2removeStim) = [];
Fourth_C_Stim_clean(trials2removeStim) = [];
Sixth_C_Stim_clean(trials2removeStim) = [];

FCs_NoStim_clean(trials2removeNoStim) = [];
SCs_NoStim_clean(trials2removeNoStim) = [];
LRCs_NoStim_clean(trials2removeNoStim) = [];
Fourth_C_NoStim_clean(trials2removeNoStim) = [];
Sixth_C_NoStim_clean(trials2removeNoStim) = [];


%% Get the keypoints by trial
Stim_Keypoints = pos(101:end, :, Stim_Trials);
NoStim_Keypoints= pos(101:end, :, NoStim_Trials);

Stim_Keypoints(:,:,trials2removeStim) = [];
NoStim_Keypoints(:,:,trials2removeNoStim) = [];

% Reshape into matrix for storage
Stim_Keypoints_Uncut = reshape(permute(Stim_Keypoints, [1, 3, 2]), [], size(Stim_Keypoints, 2));
NoStim_Keypoints_Uncut = reshape(permute(NoStim_Keypoints, [1, 3, 2]), [], size(NoStim_Keypoints, 2));

StimK = (Stim_Keypoints_Uncut - mean(Stim_Keypoints_Uncut, 'omitnan')) ./ std(Stim_Keypoints_Uncut, 'omitnan');
NoStimK = (NoStim_Keypoints_Uncut - mean(NoStim_Keypoints_Uncut, 'omitnan')) ./ std(NoStim_Keypoints_Uncut, 'omitnan');

n_time = size(Stim_Keypoints, 1);
n_keypoints = size(Stim_Keypoints, 2);
n_trials = size(Stim_Keypoints, 3);
% After normalization, reshape back
StimK_reshaped = reshape(StimK, n_time, n_trials, n_keypoints);
% Permute to original shape (time × keypoints × trials)
StimK_final = permute(StimK_reshaped, [1, 3, 2]);


n_time = size(NoStim_Keypoints, 1);
n_keypoints = size(NoStim_Keypoints, 2);
n_trials = size(NoStim_Keypoints, 3);
% After normalization, reshape back
NoStimK_reshaped = reshape(NoStimK, n_time, n_trials, n_keypoints);
% Permute to original shape (time × keypoints × trials)
NoStimK_final = permute(NoStimK_reshaped, [1, 3, 2]);

Stim_Keypoints_Uncut = StimK;
NoStim_Keypoints_Uncut = NoStimK;

% Chop trials to LRCs for Cut storage
Stim_Keypoints_Cut = chop_and_stack_neural_data(StimK_final, LRCs_Stim_clean, 100);
NoStim_Keypoints_Cut = chop_and_stack_neural_data(NoStimK_final, LRCs_NoStim_clean, 100);

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
probe1_NoStim = probe1_trialdat(:,:,NoStim_Trials);  % -2 to 4s 100Hz
probe2_NoStim = probe2_trialdat(:,:,NoStim_Trials);

probe1_NoStim(:,:,trials2removeNoStim) =[];
probe2_NoStim(:,:,trials2removeNoStim) = [];

probe1_Stim = probe1_trialdat(:,:,Stim_Trials);
probe2_Stim = probe2_trialdat(:,:,Stim_Trials);

probe1_Stim(:,:,trials2removeStim) = [];
probe2_Stim(:,:,trials2removeStim) =[];

%% Normalize the neural data to the baseline period
probe1_NoStim_norm = zscore_pregc(probe1_NoStim, pre_gc_points);  % -1 to 5s 100Hz
probe1_Stim_norm = zscore_pregc(probe1_Stim, pre_gc_points);
probe2_NoStim_norm = zscore_pregc(probe2_NoStim, pre_gc_points);
probe2_Stim_norm = zscore_pregc(probe2_Stim, pre_gc_points);

%% Get into uncut format for storage
% timepoints x neurons x trials
% puts into timepoines x trials x neurons and then reshapes
probe1_NoStim_Uncut = reshape(permute(probe1_NoStim_norm, [1, 3, 2]), [], size(probe1_NoStim_norm, 2));
probe1_Stim_Uncut = reshape(permute(probe1_Stim_norm, [1, 3, 2]), [], size(probe1_Stim_norm, 2));
probe2_NoStim_Uncut = reshape(permute(probe2_NoStim_norm, [1, 3, 2]), [], size(probe2_NoStim_norm, 2));
probe2_Stim_Uncut = reshape(permute(probe2_Stim_norm, [1, 3, 2]), [], size(probe2_Stim_norm, 2));

%% Chop up the neural data to relevant trial ends
% Handles LRCs relative to GC if neural data has 1s pre GC stored
probe1_NoStim_Cut = chop_and_stack_neural_data(probe1_NoStim_norm, LRCs_NoStim_clean, 100);
probe1_Stim_Cut = chop_and_stack_neural_data(probe1_Stim_norm, LRCs_Stim_clean, 100);

probe2_NoStim_Cut = chop_and_stack_neural_data(probe2_NoStim_norm, LRCs_NoStim_clean, 100);
probe2_Stim_Cut = chop_and_stack_neural_data(probe2_Stim_norm, LRCs_Stim_clean, 100);

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
Probe1_PCs_Stim = score1_reshaped(:, Stim_Trials, :);
Probe1_PCs_NoStim = score1_reshaped(:, NoStim_Trials, :);
Probe2_PCs_Stim = score2_reshaped(:, Stim_Trials, :);
Probe2_PCs_NoStim = score2_reshaped(:, NoStim_Trials, :);

Probe1_PCs_Stim(:, trials2removeStim, :) = [];
Probe2_PCs_Stim(:, trials2removeStim, :) = [];
Probe1_PCs_NoStim(:, trials2removeNoStim, :) = [];
Probe2_PCs_NoStim(:, trials2removeNoStim, :) = [];

% Reshape into matrices for Uncut storage
Probe1_PCs_Stim_Uncut = reshape(Probe1_PCs_Stim, [], size(Probe1_PCs_Stim, 3));
Probe1_PCs_NoStim_Uncut = reshape(Probe1_PCs_NoStim, [], size(Probe1_PCs_NoStim, 3));
Probe2_PCs_Stim_Uncut = reshape(Probe2_PCs_Stim, [], size(Probe2_PCs_Stim, 3));
Probe2_PCs_NoStim_Uncut = reshape(Probe2_PCs_NoStim, [], size(Probe2_PCs_NoStim, 3));

% Reshape to input into chopping procedure
Probe1_PCs_NoStim_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_NoStim, [1, 3, 2]), LRCs_NoStim_clean, 100);
Probe2_PCs_NoStim_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_NoStim, [1, 3, 2]), LRCs_NoStim_clean, 100);
Probe1_PCs_Stim_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_Stim, [1, 3, 2]), LRCs_Stim_clean, 100);
Probe2_PCs_Stim_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_Stim, [1, 3, 2]), LRCs_Stim_clean, 100);

%% Get the tongue length, FLCs, and LRCs times to save
SR = 100;
% Filter Tongue Length
Stim_Tongue_Uncut = all_length(pre_gc_points-100+1:end, Stim_Trials);  % -1s through 4s (500 points)
NoStim_Tongue_Uncut = all_length(pre_gc_points-100+1:end, NoStim_Trials);

Stim_Tongue_Uncut(:,trials2removeStim) = [];
NoStim_Tongue_Uncut(:,trials2removeNoStim) = [];

FCs_Adj_Stim = ceil(FCs_Stim_clean*SR + SR);
FCs_Adj_NoStim = ceil(FCs_NoStim_clean*SR + SR);

SCs_Adj_Stim = ceil(SCs_Stim_clean*SR + SR);
SCs_Adj_NoStim = ceil(SCs_NoStim_clean*SR + SR);

Fourth_C_Adj_Stim = ceil(Fourth_C_Stim_clean*SR + SR);
Fourth_C_Adj_NoStim = ceil(Fourth_C_NoStim_clean*SR + SR);

Sixth_C_Adj_Stim = ceil(Sixth_C_Stim_clean*SR + SR);
Sixth_C_Adj_NoStim = ceil(Sixth_C_NoStim_clean*SR + SR);

LRCs_Adj_Stim = ceil(LRCs_Stim_clean*SR + SR);
LRCs_Adj_NoStim = ceil(LRCs_NoStim_clean*SR + SR);

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
jaw_NoStim = jaw(:, NoStim_Trials);
jaw_NoStim(:, trials2removeNoStim) = [];

jaw_vel_NoStim = jaw_vel(:, NoStim_Trials);
jaw_vel_NoStim(:, trials2removeNoStim) = [];

jaw_Stim = jaw(:, Stim_Trials);
jaw_Stim(:, trials2removeStim) = [];

jaw_vel_Stim = jaw_vel(:, Stim_Trials);
jaw_vel_Stim(:, trials2removeStim) = [];

% Normalize the jaw data
[jaw_NoStim, jaw_vel_NoStim, jaw_Stim, Jaw_vel_R1] = deal(normalize_trials(jaw_NoStim, "zscore"), normalize_trials(jaw_vel_NoStim, "zscore"), normalize_trials(jaw_Stim, "zscore"), normalize_trials(jaw_vel_Stim, "zscore"));


%% Create kin features
uncut_JawStim = reshape(jaw_Stim, [], 1);

uncut_JawNoStim = reshape(jaw_NoStim, [], 1);

uncut_JawvelStim = reshape(jaw_vel_Stim, [], 1);

uncut_JawvelNoStim = reshape(jaw_vel_NoStim, [], 1);

jawfeats_Stim_Uncut = [uncut_JawStim, uncut_JawvelStim];
jawfeats_NoStim_Uncut = [uncut_JawNoStim, uncut_JawvelNoStim];

%% Chop up the jaw feats
% Reshape for our cutting function
jaw_Stim = reshape(jaw_Stim, size(jaw_Stim,1), 1, size(jaw_Stim,2));  % becomes 600 x 1 x 152
jaw_NoStim = reshape(jaw_NoStim, size(jaw_NoStim,1), 1, size(jaw_NoStim,2));

jaw_vel_Stim = reshape(jaw_vel_Stim, size(jaw_vel_Stim,1), 1, size(jaw_vel_Stim,2));
jaw_vel_NoStim = reshape(jaw_vel_NoStim, size(jaw_vel_NoStim,1), 1, size(jaw_vel_NoStim,2));

jaw_Stim_cut = chop_and_stack_neural_data(jaw_Stim, LRCs_Stim_clean, 100);
jaw_vel_Stim_cut = chop_and_stack_neural_data(jaw_vel_Stim, LRCs_Stim_clean, 100);

jaw_NoStim_cut = chop_and_stack_neural_data(jaw_NoStim, LRCs_NoStim_clean, 100);
jaw_vel_NoStim_cut = chop_and_stack_neural_data(jaw_vel_NoStim, LRCs_NoStim_clean, 100);

jawfeats_Stim_Cut = [uncut_JawStim, uncut_JawvelStim];
jawfeats_NoStim_Cut = [uncut_JawNoStim, uncut_JawvelNoStim];

%% -- Construct the Output File -- %%
sessionName = meta.anm;
sessionDate = meta.date;

% Construct the output folder path
outputFolder = fullfile( ...
    'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\VTA_Stim', ...
    [sessionName '_' sessionDate ]);

% Create the output folder if it does not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Save key point features
csvwrite(fullfile(outputFolder, "Keypoint_Feats_Stim_Uncut.csv"), Stim_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_NoStim_Uncut.csv"), NoStim_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_Stim_Cut.csv"), Stim_Keypoints_Cut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_NoStim_Cut.csv"), NoStim_Keypoints_Cut);

% Save Neural FRs
csvwrite(fullfile(outputFolder, "Probe1_Stim_Uncut.csv"), probe1_Stim_Uncut);
csvwrite(fullfile(outputFolder, "Probe1_NoStim_Uncut.csv"), probe1_NoStim_Uncut);
csvwrite(fullfile(outputFolder, "Probe1_Stim_Cut.csv"), probe1_Stim_Cut);
csvwrite(fullfile(outputFolder, "Probe1_NoStim_Cut.csv"), probe1_NoStim_Cut);


csvwrite(fullfile(outputFolder, "Probe2_Stim_Uncut.csv"), probe2_Stim_Uncut);
csvwrite(fullfile(outputFolder, "Probe2_NoStim_Uncut.csv"), probe2_NoStim_Uncut);
csvwrite(fullfile(outputFolder, "Probe2_Stim_Cut.csv"), probe2_Stim_Cut);
csvwrite(fullfile(outputFolder, "Probe2_NoStim_Cut.csv"), probe2_NoStim_Cut);

% Save Neural PCs
csvwrite(fullfile(outputFolder, "PCA_Probe1_Stim_Uncut.csv"), Probe1_PCs_Stim_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_NoStim_Uncut.csv"), Probe1_PCs_NoStim_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_Stim_Cut.csv"), Probe1_PCs_Stim_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_NoStim_Cut.csv"), Probe1_PCs_NoStim_Cut);


csvwrite(fullfile(outputFolder, "PCA_Probe2_Stim_Uncut.csv"), Probe2_PCs_Stim_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_NoStim_Uncut.csv"), Probe2_PCs_NoStim_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_Stim_Cut.csv"), Probe2_PCs_Stim_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_NoStim_Cut.csv"), Probe2_PCs_NoStim_Cut);

% Save Jaw Feats
csvwrite(fullfile(outputFolder, "JawFeats_Stim_Uncut.csv"), jawfeats_Stim_Uncut);
csvwrite(fullfile(outputFolder, "JawFeats_NoStim_Uncut.csv"), jawfeats_NoStim_Uncut);

csvwrite(fullfile(outputFolder, "JawFeats_Stim_Cut.csv"), jawfeats_Stim_Cut);
csvwrite(fullfile(outputFolder, "JawFeats_NoStim_Cut.csv"), jawfeats_NoStim_Cut);

% Save Tongue Length for visualizations
csvwrite(fullfile(outputFolder, "Tongue_Stim.csv"), Stim_Tongue_Uncut);
csvwrite(fullfile(outputFolder, "Tongue_NoStim.csv"), NoStim_Tongue_Uncut);

% Save FCs and LRCs
csvwrite(fullfile(outputFolder, "FCs_Stim.csv"), FCs_Adj_Stim);
csvwrite(fullfile(outputFolder, "FCs_NoStim.csv"), FCs_Adj_NoStim);

csvwrite(fullfile(outputFolder, "SCs_Stim.csv"), SCs_Adj_Stim);
csvwrite(fullfile(outputFolder, "SCs_NoStim.csv"), SCs_Adj_NoStim);

csvwrite(fullfile(outputFolder, "LRCs_Stim.csv"), LRCs_Adj_Stim);
csvwrite(fullfile(outputFolder, "LRCs_NoStim.csv"), LRCs_Adj_NoStim);

csvwrite(fullfile(outputFolder, "Sixth_C_Stim.csv"), Sixth_C_Adj_Stim);
csvwrite(fullfile(outputFolder, "Sixth_C_NoStim.csv"), Sixth_C_Adj_NoStim);

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
