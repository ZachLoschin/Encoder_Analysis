% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% May 2025
% Cleaning up preprocessing file for MC engagement analysis

%% Finding "Kinematic Modes"
clear, clc

d = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Economo-Lab-Preprocessing';
addpath(d)
addpath(genpath(fullfile(d,'utils')))
addpath(genpath(fullfile(d,'zutils')))
addpath(genpath(fullfile(d,'DataLoadingScripts')))
addpath(genpath(fullfile(d,'ObjVis')))
addpath(genpath(fullfile(d,'funcs')))

%% PARAMETERS
params.alignEvent          = 'goCue'; % 'fourthLick' 'goCue'  'moveOnset'  'firstLick' 'thirdLick' 'lastLick' 'reward'

params.behav_only          = 0;
params.timeWarp            = 0;  
params.nLicks              = 20; 

params.lowFR               = 1; 

params.condition(1) = {'hit==1 | hit==0' };    
params.condition(end+1) = {'rewardedLick==1 & stimLocation==1'};
params.condition(end+1) = {'rewardedLick==1 & stimLocation==0'};

params.tmin = -2;
params.tmax = 5;
params.dt = 1/100;
pre_gc_points = -params.tmin / params.dt;

params.smooth = 10;  

params.quality = {'good','fair','excellent', 'ok'};

params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_nostril','bottom_nostril'}};
params.feat_varToExplain = 99;  
params.N_varToExplain = 80;     
params.advance_movement = 0;

params.fcut = 10;          
params.cond = 5;         
params.method = 'xcorr';   
params.fa = false;         
params.bctype = 'reflect'; 

%% SPECIFY DATA TO LOAD
datapth = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\processed sessions\VTA_Stim';
meta = [];

meta = loadTD(meta,datapth);
params.probe = {meta.probe}; 

%% LOAD DATA
[obj,params] = loadSessionData(meta,params,params.behav_only);

for sessix = 1:numel(meta)
    me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
end

%% Handle mismatch in trial number between neural and kinematic data
nTrials_neural = size(obj.traj{1}, 2);
nTrials_kinematic = size(me.data, 2);

if nTrials_kinematic - nTrials_neural > 0
    diff = nTrials_kinematic - nTrials_neural;
    obj.bp.Ntrials = nTrials_neural;
    me.data = me.data(:,1:(end-diff));
end

%% Get kinematic data
nSessions = numel(meta);
for sessix = 1:nSessions
    disp(['----Getting kinematic data for session ', num2str(sessix),' out of ',num2str(nSessions),'----']);
    kin(sessix) = getKinematics(obj(sessix), me(sessix), params(sessix));
    pos = getKeypointsFromVideo(obj(sessix), me(sessix), params(sessix));
end

%% -- Extract All Trial Tongue Lengths -- %%
conds2use = [1];
condtrix = params(sessix).trialid{conds2use};                 
diff = nTrials_kinematic - nTrials_neural;
condtrix = condtrix(1:(end-diff), :);

NTRIALS = nTrials_neural;
kinfeat = 'tongue_length';    
sessix = 1;
kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
all_length = kin.dat(:,condtrix,kinix);

%% -- Extract All Trial Lick Port Contacts -- %%
all_contacts = obj.bp.ev.lickL;

for i = 1:obj.bp.Ntrials
    contacts = obj.bp.ev.lickL{i, 1};
    gc = obj.bp.ev.goCue(i);
    contacts = contacts - gc;
    all_contacts{i} = contacts;
end

%% -- Filter Tongue Length and LP Contacts by Trial Type --
Stim_Trials = params.trialid{2};
NoStim_Trials = params.trialid{3};
Stim_Trials = Stim_Trials(Stim_Trials < NTRIALS);
NoStim_Trials = NoStim_Trials(NoStim_Trials < NTRIALS);

Stim_Tongue = all_length((pre_gc_points-100+1):end, Stim_Trials);  
NoStim_Tongue = all_length((pre_gc_points-100+1):end, NoStim_Trials);

Stim_Contacts = all_contacts(Stim_Trials);  
NoStim_Contacts = all_contacts(NoStim_Trials);  

[trials2removeStim, FCs_Stim_clean, SCs_Stim_clean, Fourth_C_Stim_clean, Sixth_C_Stim_clean, LRCs_Stim_clean] = filter_trials_by_licking(Stim_Contacts, min_licks=4);
[trials2removeNoStim, FCs_NoStim_clean, SCs_NoStim_clean, Fourth_C_NoStim_clean, Sixth_C_NoStim_clean, LRCs_NoStim_clean] = filter_trials_by_licking(NoStim_Contacts, min_licks=4);

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

Stim_Keypoints_Uncut = reshape(permute(Stim_Keypoints, [1, 3, 2]), [], size(Stim_Keypoints, 2));
NoStim_Keypoints_Uncut = reshape(permute(NoStim_Keypoints, [1, 3, 2]), [], size(NoStim_Keypoints, 2));

StimK = (Stim_Keypoints_Uncut - mean(Stim_Keypoints_Uncut, 'omitnan')) ./ std(Stim_Keypoints_Uncut, 'omitnan');
NoStimK = (NoStim_Keypoints_Uncut - mean(NoStim_Keypoints_Uncut, 'omitnan')) ./ std(NoStim_Keypoints_Uncut, 'omitnan');

n_time = size(Stim_Keypoints, 1);
n_keypoints = size(Stim_Keypoints, 2);
n_trials = size(Stim_Keypoints, 3);
StimK_reshaped = reshape(StimK, n_time, n_trials, n_keypoints);
StimK_final = permute(StimK_reshaped, [1, 3, 2]);

n_time = size(NoStim_Keypoints, 1);
n_keypoints = size(NoStim_Keypoints, 2);
n_trials = size(NoStim_Keypoints, 3);
NoStimK_reshaped = reshape(NoStimK, n_time, n_trials, n_keypoints);
NoStimK_final = permute(NoStimK_reshaped, [1, 3, 2]);

Stim_Keypoints_Uncut = StimK;
NoStim_Keypoints_Uncut = NoStimK;

Stim_Keypoints_Cut = chop_and_stack_neural_data(StimK_final, LRCs_Stim_clean, 100);
NoStim_Keypoints_Cut = chop_and_stack_neural_data(NoStimK_final, LRCs_NoStim_clean, 100);

%% -- Get Region Specific Neural Data and Filter It-- %%
Ncells = size(obj.psth, 2);
nProbes = numel(params.cluid);
disp(['Number of probes detected: ', num2str(nProbes)]);

probe1 = 1:numel(params.cluid{1,1});
if nProbes > 1
    probe2 = numel(params.cluid{1,1})+1:Ncells;
end

probe1_trialdat = obj.trialdat(:,probe1, :);
if nProbes > 1
    probe2_trialdat = obj.trialdat(:,probe2, :);
end

probe1_NoStim = probe1_trialdat(:,:,NoStim_Trials);
probe1_Stim = probe1_trialdat(:,:,Stim_Trials);
probe1_NoStim(:,:,trials2removeNoStim) = [];
probe1_Stim(:,:,trials2removeStim) = [];

if nProbes > 1
    probe2_NoStim = probe2_trialdat(:,:,NoStim_Trials);
    probe2_Stim = probe2_trialdat(:,:,Stim_Trials);
    probe2_NoStim(:,:,trials2removeNoStim) = [];
    probe2_Stim(:,:,trials2removeStim) = [];
end

%% Normalize neural data
probe1_NoStim_norm = zscore_pregc(probe1_NoStim, pre_gc_points);
probe1_Stim_norm = zscore_pregc(probe1_Stim, pre_gc_points);

if nProbes > 1
    probe2_NoStim_norm = zscore_pregc(probe2_NoStim, pre_gc_points);
    probe2_Stim_norm = zscore_pregc(probe2_Stim, pre_gc_points);
end

%% PCA per probe
num_PCs = 10;

% Probe 1 PCA
[num_timepoints1, ~, num_trials1] = size(probe1_trialdat);
probe1_PCA = reshape(permute(probe1_trialdat, [1,3,2]), [], size(probe1_trialdat,2));
P1_PCA = (probe1_PCA - mean(probe1_PCA)) ./ std(probe1_PCA);
[~, score1, ~, ~, ~, ~] = pca(P1_PCA);
score1 = score1(:,1:num_PCs);
score1_reshaped = reshape(score1, num_timepoints1, num_trials1, num_PCs);

if nProbes > 1
    [num_timepoints2, ~, num_trials2] = size(probe2_trialdat);
    probe2_PCA = reshape(permute(probe2_trialdat, [1,3,2]), [], size(probe2_trialdat,2));
    P2_PCA = (probe2_PCA - mean(probe2_PCA)) ./ std(probe2_PCA);
    [~, score2, ~, ~, ~, ~] = pca(P2_PCA);
    score2 = score2(:,1:num_PCs);
    score2_reshaped = reshape(score2, num_timepoints2, num_trials2, num_PCs);
end

%% -- CSV Saving Section --
sessionName = meta.anm;
sessionDate = meta.date;
outputFolder = fullfile('C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\VTA_Stim',[sessionName '_' sessionDate ]);
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end

% Save keypoint features
csvwrite(fullfile(outputFolder, "Keypoint_Feats_Stim_Uncut.csv"), Stim_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_NoStim_Uncut.csv"), NoStim_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_Stim_Cut.csv"), Stim_Keypoints_Cut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_NoStim_Cut.csv"), NoStim_Keypoints_Cut);

% Save Neural FRs for Probe 1
csvwrite(fullfile(outputFolder, "Probe1_Stim_Uncut.csv"), probe1_Stim_norm);
csvwrite(fullfile(outputFolder, "Probe1_NoStim_Uncut.csv"), probe1_NoStim_norm);

if nProbes > 1
    % Save Neural FRs for Probe 2
    csvwrite(fullfile(outputFolder, "Probe2_Stim_Uncut.csv"), probe2_Stim_norm);
    csvwrite(fullfile(outputFolder, "Probe2_NoStim_Uncut.csv"), probe2_NoStim_norm);
end
