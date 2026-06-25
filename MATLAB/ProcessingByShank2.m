% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% May 2025
% Cleaning up preprocessing file for MC engagement analysis


%% Finding "Kinematic Modes"
clear,clc, fclose('all');
close all;

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
params.alignEvent          = 'goCue';
params.behav_only          = 0;
params.timeWarp            = 0;
params.nLicks              = 20;
params.lowFR               = 0.1;

params.condition(1)        = {'hit==1 | hit==0' };
params.condition(end+1)    = {'hit==1 & trialTypes == 1& rewardedLick == 1'};
params.condition(end+1)    = {'hit==1 & trialTypes == 2& rewardedLick == 1'};
params.condition(end+1)    = {'hit==1 & trialTypes == 3& rewardedLick == 1'};
params.condition(end+1)    = {'hit==1 & trialTypes == 1& rewardedLick == 4'};
params.condition(end+1)    = {'hit==1 & trialTypes == 2& rewardedLick == 4'};
params.condition(end+1)    = {'hit==1 & trialTypes == 3& rewardedLick == 4'};
params.condition(end+1)    = {'hit==1 & rewardedLick == 1'};
params.condition(end+1)    = {'hit==1 & rewardedLick == 4'};
params.condition(end+1)    = {'hit==1' };

params.tmin = -2;
params.tmax = 5;
params.dt = 1/200;
SR = 1/params.dt;
pre_gc_points = -params.tmin / params.dt;

params.smooth = 10;
params.quality = {'good','fair','excellent', 'ok'};

params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_nostril','bottom_nostril'}};
params.feat_varToExplain = 99;
params.N_varToExplain = 80;
params.advance_movement = 0;

params.fcut    = 10;
params.cond    = 5;
params.method  = 'xcorr';
params.fa      = false;
params.bctype  = 'reflect';

%% SPECIFY DATA TO LOAD

if hostname == "DESKTOP-5JJC0TM"
    datapth = 'C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Data\processed sessions\r14';
else
    datapth = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\processed sessions\r14';
end

meta = [];
anm    = "TDsa10";
date   = "2026-05-21";
shanks = [1, 2];       % shanks to include from probe 1 (MC probe)
probes = [1, 2];    % which probes to load and process
nProbes = numel(probes);

meta = loadTD(meta, datapth, anm, date, probes);
params.probe = {meta.probe};

%% LOAD DATA
[obj, params] = loadSessionData(meta, params, params.behav_only);

for sessix = 1:numel(meta)
    me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
end

%% Handle mismatch in trial number between neural and kinematic data
nTrials_neural    = size(obj.traj{1}, 2);
nTrials_kinematic = size(me.data, 2);

if nTrials_kinematic - nTrials_neural > 0
    diff = nTrials_kinematic - nTrials_neural;
    obj.bp.Ntrials = nTrials_neural;
    me.data = me.data(:, 1:(end-diff));
end

%% Add in probe/shank data
cm      = load('chanMap_NPtype24_first96_allShanks.mat');
kcoords = cm.kcoords;
xcoords = cm.xcoords;
ycoords = cm.ycoords;
chanMap = cm.chanMap;

nProbes_obj = numel(obj.clu);
for pr = 1:nProbes_obj
    nCells = numel(obj.clu{pr});
    for cc = 1:nCells
        chanNum = obj.clu{pr}(cc).channel;
        if chanNum > 384
            chanNum = 384;
        end
        obj.clu{pr}(cc).shank = kcoords(chanNum);
    end
end

%% Get kinematic data
nSessions = numel(meta);
for sessix = 1:numel(meta)
    message = strcat('----Getting kinematic data for session',{' '},num2str(sessix), {' '},'out of',{' '},num2str(nSessions),'----');
    disp(message)
    kin(sessix) = getKinematics(obj(sessix), me(sessix), params(sessix));
    pos = getKeypointsFromVideo(obj(sessix), me(sessix), params(sessix));
end

%% Extract All Trial Tongue Lengths
conds2use = [1];
condtrix  = params(sessix).trialid{conds2use};
condtrix  = condtrix(1:(end-(nTrials_kinematic-nTrials_neural)), :);

NTRIALS = nTrials_neural;

kinfeat  = 'tongue_length';
sessix   = 1;
kinix    = find(strcmp(kin(sessix).featLeg, kinfeat));
all_length = kin.dat(:, condtrix, kinix);

%% Extract All Trial Lick Port Contacts
all_contacts = obj.bp.ev.lickL;
for i = 1:obj.bp.Ntrials
    contacts = obj.bp.ev.lickL{i, 1};
    gc = obj.bp.ev.goCue(i);
    all_contacts{i} = contacts - gc;
end

%% Filter by Trial Type
R1_Trials = params.trialid{8};
R4_Trials = params.trialid{9};
R1_Trials = R1_Trials(R1_Trials <= NTRIALS);
R4_Trials = R4_Trials(R4_Trials <= NTRIALS);

R1_Trial_Track = R1_Trials;
R4_Trial_Track = R4_Trials;

R1_Tongue = all_length((pre_gc_points-SR+1):end, R1_Trials);
R4_Tongue = all_length((pre_gc_points-SR+1):end, R4_Trials);

R1_Contacts = all_contacts(R1_Trials);
R4_Contacts = all_contacts(R4_Trials);

%% Find FCs, LRCs, and Trials to Remove
[trials2removeR1, FCs_R1_clean, SCs_R1_clean, Fourth_C_R1_clean, Sixth_C_R1, LRCs_R1_clean] = filter_trials_by_licking(R1_Contacts, SR, min_licks=3);
[trials2removeR4, FCs_R4_clean, SCs_R4_clean, Fourth_C_R4_clean, Sixth_C_R4, LRCs_R4_clean] = filter_trials_by_licking(R4_Contacts, SR, min_licks=5);

FCs_R1_clean(trials2removeR1)    = [];
SCs_R1_clean(trials2removeR1)    = [];
LRCs_R1_clean(trials2removeR1)   = [];
Fourth_C_R1_clean(trials2removeR1) = [];

FCs_R4_clean(trials2removeR4)    = [];
SCs_R4_clean(trials2removeR4)    = [];
LRCs_R4_clean(trials2removeR4)   = [];
Fourth_C_R4_clean(trials2removeR4) = [];

R1_Trial_Track(trials2removeR1) = [];
R4_Trial_Track(trials2removeR4) = [];

%% Build per-probe cell arrays of trial data
% Compute cluster index ranges for each probe
Ncells = zeros(1, nProbes);
for p = 1:nProbes
    Ncells(p) = numel(params.cluid{1, p});
end

clu_ranges = cell(1, nProbes);
cumN = 0;
for p = 1:nProbes
    clu_ranges{p} = cumN + (1:Ncells(p));
    cumN = cumN + Ncells(p);
end

% For probe 1 (MC), apply shank filter
shanks_r = zeros(Ncells(1), 1);
for i = 1:Ncells(1)
    cluIdx     = params.cluid{1, 1}(i);
    shanks_r(i) = obj.clu{1, 1}(cluIdx).shank + 1;
end
neuronsToInclude = [];
for i = 1:length(shanks)
    neuronsToInclude = [neuronsToInclude; find(shanks_r == shanks(i))];
end
clu_ranges{1} = clu_ranges{1}(neuronsToInclude);

% Extract trial data per probe, remove lick-filtered trials
probe_R1 = cell(1, nProbes);
probe_R4 = cell(1, nProbes);
for p = 1:nProbes
    tmp_R4 = obj.trialdat(:, clu_ranges{p}, R4_Trials);
    tmp_R4(:, :, trials2removeR4) = [];
    probe_R4{p} = tmp_R4;

    tmp_R1 = obj.trialdat(:, clu_ranges{p}, R1_Trials);
    tmp_R1(:, :, trials2removeR1) = [];
    probe_R1{p} = tmp_R1;
end

%% Visual Assessment of Trial FR (before setting exclusion thresholds)
figure('Name', 'Trial FR Assessment');

probe1_R4mean_vis = mean(probe_R4{1}, 2);
data_R4_vis       = squeeze(probe1_R4mean_vis);
data_R4_vis       = data_R4_vis(400:800, :);
trial_FR_R4       = mean(data_R4_vis, 1);

subplot(2, 2, 1);
imagesc(data_R4_vis');
colorbar;
xlabel('Time (samples)'); ylabel('Trial');
title('R4 Mean FR: Probe 1');
box off; set(gca, 'TickLength', [0 0]);

subplot(2, 2, 2);
plot(trial_FR_R4, 1:numel(trial_FR_R4), 'k.', 'MarkerSize', 10);
hold on;
xline(mean(trial_FR_R4), 'b--', 'LineWidth', 1.5);
xlabel('Mean FR (sp/s)'); ylabel('Trial index');
title('R4 per-trial mean FR');
set(gca, 'YDir', 'reverse');
box off;

probe1_R1mean_vis = mean(probe_R1{1}, 2);
data_R1_vis       = squeeze(probe1_R1mean_vis);
data_R1_vis       = data_R1_vis(400:800, :);
trial_FR_R1       = mean(data_R1_vis, 1);

subplot(2, 2, 3);
imagesc(data_R1_vis');
colorbar;
xlabel('Time (samples)'); ylabel('Trial');
title('R1 Mean FR: Probe 1');
box off; set(gca, 'TickLength', [0 0]);

subplot(2, 2, 4);
plot(trial_FR_R1, 1:numel(trial_FR_R1), 'k.', 'MarkerSize', 10);
hold on;
xline(mean(trial_FR_R1), 'b--', 'LineWidth', 1.5);
xlabel('Mean FR (sp/s)'); ylabel('Trial index');
title('R1 per-trial mean FR');
set(gca, 'YDir', 'reverse');
box off;

sgtitle('Trial FR Assessment — set thresholds based on these', 'FontSize', 12, 'FontWeight', 'bold');

%% Trial Exclusion Parameters
FR_threshold     = 0.0;   % absolute FR threshold; set 0.0 to disable
min_trial_idx_R1 = 1;     % keep R1 trials with index >= this; 1 = no cap
max_trial_idx_R1 = Inf;   % keep R1 trials with index <= this; Inf = no cap
min_trial_idx_R4 = 1;     % keep R4 trials with index >= this; 1 = no cap
max_trial_idx_R4 = Inf;   % keep R4 trials with index <= this; Inf = no cap

%% Omit Trials: Low FR and/or Trial Index Cap — R4
probe1_R4mean = mean(probe_R4{1}, 2);
data          = squeeze(probe1_R4mean);
data          = data(400:800, :);
trial_mean_FR = mean(data, 1);

low_FR_trials_R4   = find(trial_mean_FR < FR_threshold);
idx_R4             = 1:size(probe_R4{1}, 3);
out_of_window_R4   = find(idx_R4 < min_trial_idx_R4 | idx_R4 > max_trial_idx_R4);
bad_trials_R4      = union(low_FR_trials_R4, out_of_window_R4);

fprintf('R4 excluded: %d FR-based, %d index-based, %d total / %d\n', ...
    numel(low_FR_trials_R4), numel(out_of_window_R4), numel(bad_trials_R4), size(probe_R4{1}, 3));
disp(bad_trials_R4);

figure;
imagesc(data');
colorbar;
xlabel('Time (samples)'); ylabel('Trial');
title('R4 Mean Firing Rate: Probe 1');
box off; set(gca, 'TickLength', [0 0]);

figure;
imagesc(data');
colorbar; hold on;
for i = 1:numel(bad_trials_R4)
    yline(bad_trials_R4(i), 'r-', 'LineWidth', 1.5);
end
xlabel('Time (samples)'); ylabel('Trial');
title('R4 Mean Firing Rate: Probe 1 (excluded in red)');
box off; set(gca, 'TickLength', [0 0]);

%% Omit Trials: Low FR and/or Trial Index Cap — R1
probe1_R1mean = mean(probe_R1{1}, 2);
data          = squeeze(probe1_R1mean);
data          = data(400:800, :);
trial_mean_FR = mean(data, 1);

low_FR_trials_R1   = find(trial_mean_FR < FR_threshold);
idx_R1             = 1:size(probe_R1{1}, 3);
out_of_window_R1   = find(idx_R1 < min_trial_idx_R1 | idx_R1 > max_trial_idx_R1);
bad_trials_R1      = union(low_FR_trials_R1, out_of_window_R1);

fprintf('R1 excluded: %d FR-based, %d index-based, %d total / %d\n', ...
    numel(low_FR_trials_R1), numel(out_of_window_R1), numel(bad_trials_R1), size(probe_R1{1}, 3));

figure;
imagesc(data');
colorbar;
xlabel('Time (samples)'); ylabel('Trial');
title('R1 Mean Firing Rate: Probe 1');
box off; set(gca, 'TickLength', [0 0]);

figure;
imagesc(data');
colorbar; hold on;
for i = 1:numel(bad_trials_R1)
    yline(bad_trials_R1(i), 'r-', 'LineWidth', 1.5);
end
xlabel('Time (samples)'); ylabel('Trial');
title('R1 Mean Firing Rate: Probe 1 (excluded in red)');
box off; set(gca, 'TickLength', [0 0]);

%% Remove bad trials from all probes
for p = 1:nProbes
    probe_R4{p}(:, :, bad_trials_R4) = [];
    probe_R1{p}(:, :, bad_trials_R1) = [];
end

FCs_R1_clean(bad_trials_R1)      = [];
SCs_R1_clean(bad_trials_R1)      = [];
LRCs_R1_clean(bad_trials_R1)     = [];
Fourth_C_R1_clean(bad_trials_R1) = [];

FCs_R4_clean(bad_trials_R4)      = [];
SCs_R4_clean(bad_trials_R4)      = [];
LRCs_R4_clean(bad_trials_R4)     = [];
Fourth_C_R4_clean(bad_trials_R4) = [];

R1_Trial_Track(bad_trials_R1) = [];
R4_Trial_Track(bad_trials_R4) = [];

%% Get the keypoints by trial
R1_Keypoints = pos(SR+1:end, :, R1_Trials);
R4_Keypoints = pos(SR+1:end, :, R4_Trials);

R1_Keypoints(:, :, trials2removeR1) = [];
R4_Keypoints(:, :, trials2removeR4) = [];
R1_Keypoints(:, :, bad_trials_R1)   = [];
R4_Keypoints(:, :, bad_trials_R4)   = [];

R1_Keypoints_Uncut = reshape(permute(R1_Keypoints, [1, 3, 2]), [], size(R1_Keypoints, 2));
R4_Keypoints_Uncut = reshape(permute(R4_Keypoints, [1, 3, 2]), [], size(R4_Keypoints, 2));

R1K = (R1_Keypoints_Uncut - mean(R1_Keypoints_Uncut, 'omitnan')) ./ std(R1_Keypoints_Uncut, 'omitnan');
R4K = (R4_Keypoints_Uncut - mean(R4_Keypoints_Uncut, 'omitnan')) ./ std(R4_Keypoints_Uncut, 'omitnan');

n_time       = size(R1_Keypoints, 1);
n_keypoints  = size(R1_Keypoints, 2);
n_trials     = size(R1_Keypoints, 3);
R1K_reshaped = reshape(R1K, n_time, n_trials, n_keypoints);
R1K_final    = permute(R1K_reshaped, [1, 3, 2]);

n_time       = size(R4_Keypoints, 1);
n_keypoints  = size(R4_Keypoints, 2);
n_trials     = size(R4_Keypoints, 3);
R4K_reshaped = reshape(R4K, n_time, n_trials, n_keypoints);
R4K_final    = permute(R4K_reshaped, [1, 3, 2]);

R1_Keypoints_Uncut = R1K;
R4_Keypoints_Uncut = R4K;

R1_Keypoints_Cut = chop_and_stack_neural_data(R1K_final, LRCs_R1_clean, SR);
R4_Keypoints_Cut = chop_and_stack_neural_data(R4K_final, LRCs_R4_clean, SR);

%% Normalize neural data to baseline
probe_R4_norm = cell(1, nProbes);
probe_R1_norm = cell(1, nProbes);
for p = 1:nProbes
    probe_R4_norm{p} = zscore_pregc(probe_R4{p}, pre_gc_points, SR);
    probe_R1_norm{p} = zscore_pregc(probe_R1{p}, pre_gc_points, SR);
end

%% Get into Uncut format for storage
probe_R4_Uncut = cell(1, nProbes);
probe_R1_Uncut = cell(1, nProbes);
for p = 1:nProbes
    probe_R4_Uncut{p} = reshape(permute(probe_R4_norm{p}, [1, 3, 2]), [], size(probe_R4_norm{p}, 2));
    probe_R1_Uncut{p} = reshape(permute(probe_R1_norm{p}, [1, 3, 2]), [], size(probe_R1_norm{p}, 2));
end

%% Chop neural data to trial ends
probe_R4_Cut = cell(1, nProbes);
probe_R1_Cut = cell(1, nProbes);
for p = 1:nProbes
    probe_R4_Cut{p} = chop_and_stack_neural_data(probe_R4_norm{p}, LRCs_R4_clean, SR);
    probe_R1_Cut{p} = chop_and_stack_neural_data(probe_R1_norm{p}, LRCs_R1_clean, SR);
end

%% Neural PCs
probe_cat = cell(1, nProbes);
for p = 1:nProbes
    tmp = cat(3, probe_R4{p}, probe_R1{p});
    probe_cat{p} = tmp(SR+1:end, :, :);
end

num_PCs = 10;
Probe_PCs_R4     = cell(1, nProbes);
Probe_PCs_R1     = cell(1, nProbes);
Probe_PCs_R4_Uncut = cell(1, nProbes);
Probe_PCs_R1_Uncut = cell(1, nProbes);
Probe_PCs_R4_Cut   = cell(1, nProbes);
Probe_PCs_R1_Cut   = cell(1, nProbes);

for p = 1:nProbes
    dat = probe_cat{p};
    [num_tp, ~, num_tr] = size(dat);

    dat_PCA = reshape(permute(dat, [1, 3, 2]), [], size(dat, 2));
    dat_PCA = (dat_PCA - mean(dat_PCA)) ./ std(dat_PCA);

    [~, score, ~] = pca(dat_PCA);
    score = score(:, 1:num_PCs);
    score_reshaped = reshape(score, num_tp, num_tr, num_PCs);

    nR4 = size(probe_R4{p}, 3);
    PCs_R4 = score_reshaped(:, 1:nR4, :);
    PCs_R1 = score_reshaped(:, nR4+1:end, :);

    Probe_PCs_R4{p} = PCs_R4;
    Probe_PCs_R1{p} = PCs_R1;

    Probe_PCs_R4_Uncut{p} = reshape(PCs_R4, [], size(PCs_R4, 3));
    Probe_PCs_R1_Uncut{p} = reshape(PCs_R1, [], size(PCs_R1, 3));

    Probe_PCs_R4_Cut{p} = chop_and_stack_neural_data(permute(PCs_R4, [1, 3, 2]), LRCs_R4_clean, SR);
    Probe_PCs_R1_Cut{p} = chop_and_stack_neural_data(permute(PCs_R1, [1, 3, 2]), LRCs_R1_clean, SR);
end

%% Tongue length for saving
R1_Tongue_Uncut = all_length(pre_gc_points-SR+1:end, R1_Trials);
R4_Tongue_Uncut = all_length(pre_gc_points-SR+1:end, R4_Trials);
R1_Tongue_Uncut(:, trials2removeR1) = [];
R4_Tongue_Uncut(:, trials2removeR4) = [];
R1_Tongue_Uncut(:, bad_trials_R1)   = [];
R4_Tongue_Uncut(:, bad_trials_R4)   = [];

FCs_Adj_R1    = ceil(FCs_R1_clean*SR + SR);
FCs_Adj_R4    = ceil(FCs_R4_clean*SR + SR);
SCs_Adj_R1    = ceil(SCs_R1_clean*SR + SR);
SCs_Adj_R4    = ceil(SCs_R4_clean*SR + SR);
Fourth_C_Adj_R1 = ceil(Fourth_C_R1_clean*SR + SR);
Fourth_C_Adj_R4 = ceil(Fourth_C_R4_clean*SR + SR);
LRCs_Adj_R1   = ceil(LRCs_R1_clean*SR + SR);
LRCs_Adj_R4   = ceil(LRCs_R4_clean*SR + SR);

%% Construct Output Folder
sessionName = string(meta.anm);
sessionDate = string(meta.date);
baseFolder  = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\TDsa9';
outputFolder = fullfile(baseFolder, sessionName + "_" + sessionDate);

if ~exist(char(outputFolder), 'dir')
    mkdir(char(outputFolder));
end

%% Save
csvwrite(fullfile(outputFolder, "R1_Trial_Track.csv"), R1_Trial_Track);
csvwrite(fullfile(outputFolder, "R4_Trial_Track.csv"), R4_Trial_Track);

csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Uncut.csv"), R1_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Uncut.csv"), R4_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Cut.csv"),   R1_Keypoints_Cut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Cut.csv"),   R4_Keypoints_Cut);

csvwrite(fullfile(outputFolder, "Tongue_R1.csv"), R1_Tongue_Uncut);
csvwrite(fullfile(outputFolder, "Tongue_R4.csv"), R4_Tongue_Uncut);

csvwrite(fullfile(outputFolder, "FCs_R1.csv"),  FCs_Adj_R1);
csvwrite(fullfile(outputFolder, "FCs_R4.csv"),  FCs_Adj_R4);
csvwrite(fullfile(outputFolder, "LRCs_R1.csv"), LRCs_Adj_R1);
csvwrite(fullfile(outputFolder, "LRCs_R4.csv"), LRCs_Adj_R4);

% Save per-probe PCs (only for probes in the probes array)
for p = 1:nProbes
    probeNum = probes(p);
    csvwrite(fullfile(outputFolder, sprintf("PCA_Probe%d_R1_Uncut.csv", probeNum)), Probe_PCs_R1_Uncut{p});
    csvwrite(fullfile(outputFolder, sprintf("PCA_Probe%d_R4_Uncut.csv", probeNum)), Probe_PCs_R4_Uncut{p});
    csvwrite(fullfile(outputFolder, sprintf("PCA_Probe%d_R1_Cut.csv",   probeNum)), Probe_PCs_R1_Cut{p});
    csvwrite(fullfile(outputFolder, sprintf("PCA_Probe%d_R4_Cut.csv",   probeNum)), Probe_PCs_R4_Cut{p});
end

% Save metadata
metadataFile = fullfile(outputFolder, 'metadata.txt');
fid = fopen(metadataFile, 'w');
fprintf(fid, 'Processing Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Script Name: %s\n', mfilename('fullpath'));
fprintf(fid, 'Session ID: %s\n', sessionName);
fprintf(fid, 'Session Date: %s\n', sessionDate);
fprintf(fid, 'Probes processed: %s\n', num2str(probes));
fprintf(fid, 'Shanks (probe 1): %s\n', num2str(shanks));
fclose("all");

%% Local Functions

function normalized_data = normalize_trials(data, method)
    data(isnan(data)) = 0;
    concatenated = data(:);
    switch lower(method)
        case 'zscore'
            normalized_concatenated = (concatenated - mean(concatenated)) / std(concatenated);
        case 'range'
            normalized_concatenated = normalize(concatenated, 'range');
        otherwise
            error("Invalid method. Choose 'zscore' or 'range'.");
    end
    normalized_data = reshape(normalized_concatenated, size(data));
end

function zscored_data = zscore_pregc(data, pre_gc_points, SR)
    data_pregc = data(1:pre_gc_points, :, :);
    concatenated_data = reshape(data_pregc, [], size(data_pregc, 2));
    neuron_means = mean(concatenated_data, 1);
    neuron_stds  = std(concatenated_data, 0, 1);
    zscored_data = (data - neuron_means) ./ neuron_stds;
    S = SR * 2;
    zscored_data = zscored_data(pre_gc_points-S+1:end, :, :);
end

function normalized = zscore_trials(data, means, stds)
    normalized = (data - reshape(means, 1, [], 1)) ./ reshape(stds, 1, [], 1);
end