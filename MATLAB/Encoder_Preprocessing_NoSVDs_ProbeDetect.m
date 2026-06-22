% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% May 2025
% Cleaning up preprocessing file for MC engagement analysis

%% Finding "Kinematic Modes"
clear, clc

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
params.alignEvent = 'goCue';
params.behav_only = 0;
params.timeWarp = 0;
params.nLicks = 20;
params.lowFR = 1.0;

params.condition(1) = {'hit==1 | hit==0'};
params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 1'};
params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 1'};
params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 1'};
params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 4'};
params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 4'};
params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 4'};
params.condition(end+1) = {'hit==1 & rewardedLick == 1'};
params.condition(end+1) = {'hit==1 & rewardedLick == 4'};
params.condition(end+1) = {'hit==1'};

params.tmin = -2;
params.tmax = 5;
params.dt = 1/100;
pre_gc_points = -params.tmin / params.dt;
params.smooth = 10;
params.quality = {'good','fair','excellent','ok'};

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
if hostname == "DESKTOP-5JJC0TM"
    datapth = 'C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Data\processed sessions\r1';
else
    datapth = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\processed sessions\r1';
end

meta = loadTD([],datapth);
params.probe = {meta.probe};
params.probe = {[1]};

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

%% Kinematic data extraction
nSessions = numel(meta);
for sessix = 1:numel(meta)
    disp(['----Getting kinematic data for session ' num2str(sessix) ' out of ' num2str(nSessions) '----']);
    kin(sessix) = getKinematics(obj(sessix), me(sessix), params(sessix));
    pos = getKeypointsFromVideo(obj(sessix), me(sessix), params(sessix));
end

%% Extract all trial tongue lengths
conds2use = [1];
condtrix = params(sessix).trialid{conds2use};
diff = nTrials_kinematic - nTrials_neural;
condtrix = condtrix(1:(end-diff), :);

NTRIALS = nTrials_neural;
kinfeat = 'tongue_length';
sessix = 1;
kinix = find(strcmp(kin(sessix).featLeg,kinfeat));
all_length = kin.dat(:,condtrix,kinix);

%% Lick contacts
all_contacts = obj.bp.ev.lickL;
for i = 1:obj.bp.Ntrials
    contacts = obj.bp.ev.lickL{i, 1};
    gc = obj.bp.ev.goCue(i);
    all_contacts{i} = contacts - gc;
end

%% Filter tongue length and LP contacts
R1_Trials = params.trialid{8};
R4_Trials = params.trialid{9};
R1_Trials = R1_Trials(R1_Trials <= NTRIALS);
R4_Trials = R4_Trials(R4_Trials <= NTRIALS);

R1_Trial_Track = R1_Trials;
R4_Trial_Track = R4_Trials;

R1_Tongue = all_length((pre_gc_points-100+1):end, R1_Trials);
R4_Tongue = all_length((pre_gc_points-100+1):end, R4_Trials);
R1_Contacts = all_contacts(R1_Trials);
R4_Contacts = all_contacts(R4_Trials);

%% Filter by licking
[trials2removeR1, FCs_R1_clean, SCs_R1_clean, Fourth_C_R1_clean, ~, LRCs_R1_clean] = filter_trials_by_licking(R1_Contacts, min_licks=3);
[trials2removeR4, FCs_R4_clean, SCs_R4_clean, Fourth_C_R4_clean, ~, LRCs_R4_clean] = filter_trials_by_licking(R4_Contacts, min_licks=5);

FCs_R1_clean(trials2removeR1) = [];
SCs_R1_clean(trials2removeR1) = [];
LRCs_R1_clean(trials2removeR1) = [];
Fourth_C_R1_clean(trials2removeR1) = [];

FCs_R4_clean(trials2removeR4) = [];
SCs_R4_clean(trials2removeR4) = [];
LRCs_R4_clean(trials2removeR4) = [];
Fourth_C_R4_clean(trials2removeR4) = [];

R1_Trial_Track(trials2removeR1) = [];
R4_Trial_Track(trials2removeR4) = [];

%% Keypoints
R1_Keypoints = pos(101:end, :, R1_Trials);
R4_Keypoints = pos(101:end, :, R4_Trials);
R1_Keypoints(:,:,trials2removeR1) = [];
R4_Keypoints(:,:,trials2removeR4) = [];

R1_Keypoints_Uncut = reshape(permute(R1_Keypoints, [1, 3, 2]), [], size(R1_Keypoints, 2));
R4_Keypoints_Uncut = reshape(permute(R4_Keypoints, [1, 3, 2]), [], size(R4_Keypoints, 2));
R1K = zscore(R1_Keypoints_Uncut);
R4K = zscore(R4_Keypoints_Uncut);

n_time = size(R1_Keypoints, 1);
n_keypoints = size(R1_Keypoints, 2);
n_trials = size(R1_Keypoints, 3);
R1K_final = permute(reshape(R1K, n_time, n_trials, n_keypoints), [1, 3, 2]);

n_time = size(R4_Keypoints, 1);
n_keypoints = size(R4_Keypoints, 2);
n_trials = size(R4_Keypoints, 3);
R4K_final = permute(reshape(R4K, n_time, n_trials, n_keypoints), [1, 3, 2]);

R1_Keypoints_Uncut = R1K;
R4_Keypoints_Uncut = R4K;
R1_Keypoints_Cut = chop_and_stack_neural_data(R1K_final, LRCs_R1_clean, 100);
R4_Keypoints_Cut = chop_and_stack_neural_data(R4K_final, LRCs_R4_clean, 100);

%% Neural Data Processing
Ncells = size(obj.psth, 2);

if iscell(params.cluid) && numel(params.cluid) > 1
    % Two probes
    probe1 = 1:numel(params.cluid{1, 1});
    probe2 = numel(params.cluid{1, 1}) + 1 : Ncells;
else
    % One probe
    probe1 = 1:numel(params.cluid);
    probe2 = []; % No second probe
end

% Build trialdat slices based on detected probes
probe1_trialdat = obj.trialdat(:, probe1, :);
if ~isempty(probe2)
    probe2_trialdat = obj.trialdat(:, probe2, :);
end

% Display the number of probes detected
if ~isempty(probe2)
    nProbes = 2;
else
    nProbes = 1;
end
disp(['Number of probes: ', num2str(nProbes)]);

% Process probe1
probe1_R4 = probe1_trialdat(:,:,R4_Trials);
probe1_R4(:,:,trials2removeR4) = [];
probe1_R1 = probe1_trialdat(:,:,R1_Trials);
probe1_R1(:,:,trials2removeR1) = [];
probe1_R4_norm = zscore_pregc(probe1_R4, pre_gc_points);
probe1_R1_norm = zscore_pregc(probe1_R1, pre_gc_points);
probe1_R4_Uncut = reshape(permute(probe1_R4_norm, [1, 3, 2]), [], size(probe1_R4_norm, 2));
probe1_R1_Uncut = reshape(permute(probe1_R1_norm, [1, 3, 2]), [], size(probe1_R1_norm, 2));
probe1_R4_Cut = chop_and_stack_neural_data(probe1_R4_norm, LRCs_R4_clean, 100);
probe1_R1_Cut = chop_and_stack_neural_data(probe1_R1_norm, LRCs_R1_clean, 100);

if ~isempty(probe2)
    % Process probe2 only if exists
    probe2_R4 = probe2_trialdat(:,:,R4_Trials);
    probe2_R4(:,:,trials2removeR4) = [];
    probe2_R1 = probe2_trialdat(:,:,R1_Trials);
    probe2_R1(:,:,trials2removeR1) = [];
    probe2_R4_norm = zscore_pregc(probe2_R4, pre_gc_points);
    probe2_R1_norm = zscore_pregc(probe2_R1, pre_gc_points);
    probe2_R4_Uncut = reshape(permute(probe2_R4_norm, [1, 3, 2]), [], size(probe2_R4_norm, 2));
    probe2_R1_Uncut = reshape(permute(probe2_R1_norm, [1, 3, 2]), [], size(probe2_R1_norm, 2));
    probe2_R4_Cut = chop_and_stack_neural_data(probe2_R4_norm, LRCs_R4_clean, 100);
    probe2_R1_Cut = chop_and_stack_neural_data(probe2_R1_norm, LRCs_R1_clean, 100);
end

%% PCA for probe1
probe1_segment = probe1_trialdat(101:end, :, :);
[num_timepoints1, ~, num_trials1] = size(probe1_segment);
probe1_PCA = reshape(permute(probe1_segment, [1, 3, 2]), [], size(probe1_segment, 2));
[coeff1, score1] = pca(zscore(probe1_PCA));
score1 = score1(:, 1:10);
score1_reshaped = reshape(score1, num_timepoints1, num_trials1, 10);

Probe1_PCs_R1 = score1_reshaped(:, R1_Trials, :);
Probe1_PCs_R4 = score1_reshaped(:, R4_Trials, :);
Probe1_PCs_R1(:, trials2removeR1, :) = [];
Probe1_PCs_R4(:, trials2removeR4, :) = [];
Probe1_PCs_R1_Uncut = reshape(Probe1_PCs_R1, [], size(Probe1_PCs_R1, 3));
Probe1_PCs_R4_Uncut = reshape(Probe1_PCs_R4, [], size(Probe1_PCs_R4, 3));
Probe1_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R4, [1, 3, 2]), LRCs_R4_clean, 100);
Probe1_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R1, [1, 3, 2]), LRCs_R1_clean, 100);

if ~isempty(probe2)
    % PCA for probe2
    probe2_segment = probe2_trialdat(101:end, :, :);
    [num_timepoints2, ~, num_trials2] = size(probe2_segment);
    probe2_PCA = reshape(permute(probe2_segment, [1, 3, 2]), [], size(probe2_segment, 2));
    [coeff2, score2] = pca(zscore(probe2_PCA));
    score2 = score2(:, 1:10);
    score2_reshaped = reshape(score2, num_timepoints2, num_trials2, 10);

    Probe2_PCs_R1 = score2_reshaped(:, R1_Trials, :);
    Probe2_PCs_R4 = score2_reshaped(:, R4_Trials, :);
    Probe2_PCs_R1(:, trials2removeR1, :) = [];
    Probe2_PCs_R4(:, trials2removeR4, :) = [];
    Probe2_PCs_R1_Uncut = reshape(Probe2_PCs_R1, [], size(Probe2_PCs_R1, 3));
    Probe2_PCs_R4_Uncut = reshape(Probe2_PCs_R4, [], size(Probe2_PCs_R4, 3));
    Probe2_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R4, [1, 3, 2]), LRCs_R4_clean, 100);
    Probe2_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R1, [1, 3, 2]), LRCs_R1_clean, 100);
end

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

%% Save Data
sessionName = meta.anm;
sessionDate = meta.date;

if hostname == "DESKTOP-5JJC0TM"
    outputFolder = fullfile('C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Preprocessed_Encoder\r1_sep', [sessionName '_' sessionDate ]);
else
    outputFolder = fullfile('C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\r1_sep', [sessionName '_' sessionDate ]);
end

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% Save Key Point Features
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Uncut.csv"), R1_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Uncut.csv"), R4_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Cut.csv"), R1_Keypoints_Cut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Cut.csv"), R4_Keypoints_Cut);

%% Save Neural FRs for Probe 1
csvwrite(fullfile(outputFolder, "Probe1_R1_Uncut.csv"), probe1_R1_Uncut);
csvwrite(fullfile(outputFolder, "Probe1_R4_Uncut.csv"), probe1_R4_Uncut);
csvwrite(fullfile(outputFolder, "Probe1_R1_Cut.csv"), probe1_R1_Cut);
csvwrite(fullfile(outputFolder, "Probe1_R4_Cut.csv"), probe1_R4_Cut);

%% Save Neural FRs for Probe 2 (only if nProbes > 1)
if nProbes > 1
    csvwrite(fullfile(outputFolder, "Probe2_R1_Uncut.csv"), probe2_R1_Uncut);
    csvwrite(fullfile(outputFolder, "Probe2_R4_Uncut.csv"), probe2_R4_Uncut);
    csvwrite(fullfile(outputFolder, "Probe2_R1_Cut.csv"), probe2_R1_Cut);
    csvwrite(fullfile(outputFolder, "Probe2_R4_Cut.csv"), probe2_R4_Cut);
end

%% Save Neural PCs for Probe 1
csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Uncut.csv"), Probe1_PCs_R1_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Uncut.csv"), Probe1_PCs_R4_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Cut.csv"), Probe1_PCs_R1_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Cut.csv"), Probe1_PCs_R4_Cut);

%% Save Neural PCs for Probe 2 (only if nProbes > 1)
if nProbes > 1
    csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Uncut.csv"), Probe2_PCs_R1_Uncut);
    csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Uncut.csv"), Probe2_PCs_R4_Uncut);
    csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Cut.csv"), Probe2_PCs_R1_Cut);
    csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Cut.csv"), Probe2_PCs_R4_Cut);
end

%% Save Jaw Features
csvwrite(fullfile(outputFolder, "JawFeats_R1_Uncut.csv"), jawfeats_R1_Uncut);
csvwrite(fullfile(outputFolder, "JawFeats_R4_Uncut.csv"), jawfeats_R4_Uncut);
csvwrite(fullfile(outputFolder, "JawFeats_R1_Cut.csv"), jawfeats_R1_Cut);
csvwrite(fullfile(outputFolder, "JawFeats_R4_Cut.csv"), jawfeats_R4_Cut);

%% Save Tongue Length for visualizations
csvwrite(fullfile(outputFolder, "Tongue_R1.csv"), R1_Tongue_Uncut);
csvwrite(fullfile(outputFolder, "Tongue_R4.csv"), R4_Tongue_Uncut);

%% Save FCs and LRCs
csvwrite(fullfile(outputFolder, "FCs_R1.csv"), FCs_Adj_R1);
csvwrite(fullfile(outputFolder, "FCs_R4.csv"), FCs_Adj_R4);
csvwrite(fullfile(outputFolder, "SCs_R1.csv"), SCs_Adj_R1);
csvwrite(fullfile(outputFolder, "SCs_R4.csv"), SCs_Adj_R4);
csvwrite(fullfile(outputFolder, "LRCs_R1.csv"), LRCs_Adj_R1);
csvwrite(fullfile(outputFolder, "LRCs_R4.csv"), LRCs_Adj_R4);
% csvwrite(fullfile(outputFolder, "Sixth_C_R1.csv"), Sixth_C_Adj_R1);
% csvwrite(fullfile(outputFolder, "Sixth_C_R4.csv"), Sixth_C_Adj_R4);

%% Save metadata as a .txt file for record-keeping
metadataFile = fullfile(outputFolder, 'metadata.txt');
fid = fopen(metadataFile, 'w');
fprintf(fid, 'Processing Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Script Name: %s\n', mfilename('fullpath'));
fprintf(fid, 'Session ID: %s\n', sessionName);
fprintf(fid, 'Session Date: %s\n', sessionDate);
fclose(fid);


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
