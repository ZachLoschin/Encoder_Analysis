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
params.alignEvent          = 'goCue'; % 'fourthLick' 'goCue'  'moveOnset'  'firstLick' 'thirdLick' 'lastLick' 'reward'

% time warping only operates on neural data for now.
params.behav_only          = 0;
params.timeWarp            = 0;  % piecewise linear time warping - each lick duration on each trial gets warped to median lick duration for that lick across trials
params.nLicks              = 20; % number of post go cue licks to calculate median lick duration for and warp individual trials to


% Check for trials that data loss occurs from recording. This could
% affect the PC calculation and the HMM-GLM fits.

% explore lowering this to aroiund 0.5
params.lowFR               = 0.1; % remove clusters with firing rates across all trials less than this val

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
params.dt = 1/200;
SR = 1/params.dt;
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

if hostname == "DESKTOP-5JJC0TM"
    datapth = 'C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Data\processed sessions\r14';
else
    datapth = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\processed sessions\r14';
end

meta = [];
anm="TDsa12";
date="2026-06-08";
shanks=[1];
probes=[1,2,3];
meta = loadTD(meta, datapth, anm, date, probes);
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

%% Add in probe number data
    % Channel map
    cm = load('chanMap_NPtype24_first96_allShanks.mat');
    
    % Shank number for each channel
    kcoords = cm.kcoords;          % 384×1 vector
    % Sanity check plot of channel map organization
    xcoords = cm.xcoords;   % x-location (in microns) of each channel
    ycoords = cm.ycoords;   % y-location/depth (in microns) of each channel
        
    chanMap = cm.chanMap;
    
    nChans = length(xcoords);
    % for cc = 1:nChans
    %     scatter(xcoords(cc),ycoords(cc)); hold on
    %     text(xcoords(cc)+3,ycoords(cc),num2str(chanMap(cc)))
    % 
    % end
    % title('Channel Organization - chanMap-NPtype24-first96-allShanks')
    % xlabel('microns (x direction)')
    % ylabel('microns (y direction)')
    
    % Assign a shank number to each cell in obj.clu
    nProbes = numel(obj.clu);
    for pr = 1:nProbes                       % For each probe in the obj...
        nCells = numel(obj.clu{pr});
        for cc = 1:nCells                       % For each cell on the probe...
            chanNum = obj.clu{pr}(cc).channel;  % Get channel number of cell
            if chanNum > 384
                chanNum = 384;
            end
            shankNum = kcoords(chanNum);        % Get shank number of channel
            obj.clu{pr}(cc).shank = shankNum;   % Save to field 'shank' in obj.clu
        end
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
R1_Trials = params.trialid{8};
R4_Trials = params.trialid{9};

R1_Trials = R1_Trials(R1_Trials <= NTRIALS);
R4_Trials = R4_Trials(R4_Trials <= NTRIALS);

R1_Trial_Track = R1_Trials;
R4_Trial_Track = R4_Trials;

% Filter Tongue Length
R1_Tongue = all_length((pre_gc_points-SR+1):end, R1_Trials);  % -1s through 4s (500 points)
R4_Tongue = all_length((pre_gc_points-SR+1):end, R4_Trials);

% Filter LP Contacts
R1_Contacts = all_contacts(R1_Trials);  % these contacts are relative to the gc
R4_Contacts = all_contacts(R4_Trials);  % these contacts are relative to the gc

%% Find trial FCs, Last Relevant Contacts (LRCs), and Trials2Remove

[trials2removeR1, FCs_R1_clean, SCs_R1_clean, Fourth_C_R1_clean, Sixth_C_R1, LRCs_R1_clean] = filter_trials_by_licking(R1_Contacts, SR, min_licks=3);
[trials2removeR4, FCs_R4_clean, SCs_R4_clean, Fourth_C_R4_clean, Sixth_C_R4, LRCs_R4_clean] = filter_trials_by_licking(R4_Contacts, SR, min_licks=5);

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

%% -- Get Region Specific Neural Data and Filter It By Shank-- %%
Ncells_P1 = numel(params.cluid{1, 1});
Ncells_P2 = numel(params.cluid{1, 2});
Ncells_P3 = numel(params.cluid{1, 3});

clu_1 = 1:Ncells_P1;
clu_2 = Ncells_P1+1:Ncells_P1+Ncells_P2;
clu_3 = (Ncells_P1+Ncells_P2+1):Ncells_P1+Ncells_P2+Ncells_P3;

shanks_r = zeros(Ncells_P1, 1);
probeID = 1;

% Get shank number for each neuron on probe 1 (MC)
for i=1:Ncells_P1
    cluIdx = params.cluid{1, probeID}(i);
    shanks_r(i) = obj.clu{1, probeID}(cluIdx).shank + 1;  % shanks 1-4
end

neuronsToInclude = [];
% Include only neurons from desired shanks
for i=1:length(shanks)
    shankID = shanks(i);
    idx = find(shanks_r == shankID);
    neuronsToInclude = [neuronsToInclude; idx];
end

clu_1 = clu_1(neuronsToInclude);

probe1_trialdat = obj.trialdat(:,clu_1, :);
probe2_trialdat = obj.trialdat(:,clu_2, :);
probe3_trialdat = obj.trialdat(:,clu_3, :);

% Limit to the desired trials
% This is done in multiple rouds since the final_RN_trials are indices into
% the RN_trials, not absolute trial numbers
probe1_R4 = probe1_trialdat(:,:,R4_Trials);  % -2 to 4s 100Hz
probe2_R4 = probe2_trialdat(:,:,R4_Trials);
probe3_R4 = probe3_trialdat(:,:,R4_Trials);

probe1_R4(:,:,trials2removeR4) =[];
probe2_R4(:,:,trials2removeR4) = [];
probe3_R4(:,:,trials2removeR4) = [];

probe1_R1 = probe1_trialdat(:,:,R1_Trials);
probe2_R1 = probe2_trialdat(:,:,R1_Trials);
probe3_R1 = probe3_trialdat(:,:,R1_Trials);

probe1_R1(:,:,trials2removeR1) = [];
probe2_R1(:,:,trials2removeR1) =[];
probe3_R1(:,:,trials2removeR1) =[];

%% Visual Assessment of Trial FR (before setting exclusion thresholds)

figure('Name', 'Trial FR Assessment');

% --- R4 ---
probe1_R4mean_vis = mean(probe1_R4, 2);
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

% --- R1 ---
probe1_R1mean_vis = mean(probe1_R1, 2);
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
FR_threshold      = 0.0;    % absolute FR threshold; set 0.0 to disable
min_trial_idx_R1  = 1;      % keep R1 trials with index >= this; 1 = no cap
max_trial_idx_R1  = Inf;    % keep R1 trials with index <= this; Inf = no cap
min_trial_idx_R4  = 1;      % keep R4 trials with index >= this; 1 = no cap
max_trial_idx_R4  = Inf;    % keep R4 trials with index <= this; Inf = no cap

%% Omit Trials with Significantly Lower FR than Mean R4
probe1_R4mean    = mean(probe1_R4, 2);
data             = squeeze(probe1_R4mean);
data             = data(400:800, :);
trial_mean_FR    = mean(data, 1);
grand_mean       = mean(trial_mean_FR);
grand_std        = std(trial_mean_FR);

low_FR_trials_R4   = find(trial_mean_FR < FR_threshold);
idx_R4             = 1:size(probe1_R4, 3);
out_of_window_R4   = find(idx_R4 < min_trial_idx_R4 | idx_R4 > max_trial_idx_R4);
bad_trials_R4      = union(low_FR_trials_R4, out_of_window_R4);

fprintf('R4 excluded: %d FR-based, %d index-based, %d total / %d\n', ...
    numel(low_FR_trials_R4), numel(out_of_window_R4), numel(bad_trials_R4), size(probe1_R4, 3));
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

%% Omit Trials with Significantly Lower FR than Mean R1
probe1_R1mean    = mean(probe1_R1, 2);
data             = squeeze(probe1_R1mean);
data             = data(400:800, :);
trial_mean_FR    = mean(data, 1);
grand_mean       = mean(trial_mean_FR);
grand_std        = std(trial_mean_FR);

low_FR_trials_R1   = find(trial_mean_FR < FR_threshold);
idx_R1             = 1:size(probe1_R1, 3);
out_of_window_R1   = find(idx_R1 < min_trial_idx_R1 | idx_R1 > max_trial_idx_R1);
bad_trials_R1      = union(low_FR_trials_R1, out_of_window_R1);

fprintf('R1 excluded: %d FR-based, %d index-based, %d total / %d\n', ...
    numel(low_FR_trials_R1), numel(out_of_window_R1), numel(bad_trials_R1), size(probe1_R1, 3));

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

%% Remove bad trials from neural and kinematic data
probe1_R4(:,:,bad_trials_R4) = [];
probe2_R4(:,:,bad_trials_R4) = [];
probe3_R4(:,:,bad_trials_R4) = [];
probe1_R1(:,:,bad_trials_R1) = [];
probe2_R1(:,:,bad_trials_R1) = [];
probe3_R1(:,:,bad_trials_R1) = [];

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
R4_Keypoints= pos(SR+1:end, :, R4_Trials);

R1_Keypoints(:,:,trials2removeR1) = [];
R4_Keypoints(:,:,trials2removeR4) = [];

R1_Keypoints(:,:,bad_trials_R1) = [];
R4_Keypoints(:,:,bad_trials_R4) = [];

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
R1_Keypoints_Cut = chop_and_stack_neural_data(R1K_final, LRCs_R1_clean, SR);
R4_Keypoints_Cut = chop_and_stack_neural_data(R4K_final, LRCs_R4_clean, SR);

%% Normalize the neural data to the baseline period
probe1_R4_norm = zscore_pregc(probe1_R4, pre_gc_points, SR);  % -1 to 5s 100Hz
probe1_R1_norm = zscore_pregc(probe1_R1, pre_gc_points, SR);

probe2_R4_norm = zscore_pregc(probe2_R4, pre_gc_points, SR);
probe2_R1_norm = zscore_pregc(probe2_R1, pre_gc_points, SR);

probe3_R4_norm = zscore_pregc(probe3_R4, pre_gc_points, SR);
probe3_R1_norm = zscore_pregc(probe3_R1, pre_gc_points, SR);

%% Get into uncut format for storage
% timepoints x neurons x trials
% puts into timepoines x trials x neurons and then reshapes
probe1_R4_Uncut = reshape(permute(probe1_R4_norm, [1, 3, 2]), [], size(probe1_R4_norm, 2));
probe1_R1_Uncut = reshape(permute(probe1_R1_norm, [1, 3, 2]), [], size(probe1_R1_norm, 2));

probe2_R4_Uncut = reshape(permute(probe2_R4_norm, [1, 3, 2]), [], size(probe2_R4_norm, 2));
probe2_R1_Uncut = reshape(permute(probe2_R1_norm, [1, 3, 2]), [], size(probe2_R1_norm, 2));

probe3_R4_Uncut = reshape(permute(probe3_R4_norm, [1, 3, 2]), [], size(probe3_R4_norm, 2));
probe3_R1_Uncut = reshape(permute(probe3_R1_norm, [1, 3, 2]), [], size(probe3_R1_norm, 2));

%% Chop up the neural data to relevant trial ends
% Handles LRCs relative to GC if neural data has 1s pre GC stored
probe1_R4_Cut = chop_and_stack_neural_data(probe1_R4_norm, LRCs_R4_clean, SR);
probe1_R1_Cut = chop_and_stack_neural_data(probe1_R1_norm, LRCs_R1_clean, SR);

probe2_R4_Cut = chop_and_stack_neural_data(probe2_R4_norm, LRCs_R4_clean, SR);
probe2_R1_Cut = chop_and_stack_neural_data(probe2_R1_norm, LRCs_R1_clean, SR);

probe3_R4_Cut = chop_and_stack_neural_data(probe3_R4_norm, LRCs_R4_clean, SR);
probe3_R1_Cut = chop_and_stack_neural_data(probe3_R1_norm, LRCs_R1_clean, SR);

% %% Plotting Checks for Data Quality PSTH
% n_timepoints = size(obj.psth, 1);
% time = -2 + (0:n_timepoints-1) * (1/200);
% 
% % --- Select how many neurons and which condition ---
% n_to_plot    = 20;  % how many neurons to plot
% cond_to_plot = 1;   % which condition
% 
% neurons_to_plot = 1:n_to_plot;
% colors = lines(n_to_plot);
% 
% n_cols = 4;
% n_rows = ceil(n_to_plot / n_cols);
% 
% figure;
% for i = 1:n_to_plot
%     subplot(n_rows, n_cols, i); hold on;
%     plot(time, obj.psth(:, neurons_to_plot(i), cond_to_plot), ...
%         'Color', colors(i,:), 'LineWidth', 1.0);
%     xline(0, 'k--', 'LineWidth', 0.8);
%     xlim([-2 5]);
%     title(sprintf('Neuron %d', neurons_to_plot(i)));
%     xlabel('Time (s)'); ylabel('Hz');
% end
% 
% sgtitle(sprintf('PSTH — Condition %d', cond_to_plot));

%% Neural PCs
% Get -1s -> 5s data
% probe1 = probe1_trialdat(SR+1:end, :, :);
% probe2 = probe2_trialdat(SR+1:end, :, :);
% probe3 = probe3_trialdat(SR+1:end, :, :);

% Concatenate the already trial sorted data for PCA
probe1 = cat(3, probe1_R4, probe1_R1);
probe2 = cat(3, probe2_R4, probe2_R1);
probe3 = cat(3, probe3_R4, probe3_R1);

probe1 = probe1(SR+1:end, :, :);
probe2 = probe2(SR+1:end, :, :);
probe3 = probe3(SR+1:end, :, :);


[num_timepoints1, ~, num_trials1] = size(probe1);
[num_timepoints2, ~, num_trials2] = size(probe2);
[num_timepoints3, ~, num_trials3] = size(probe3);

% Reshape to (timepoints x trials) x features
probe1_PCA = reshape(permute(probe1, [1, 3, 2]), [], size(probe1, 2));
probe2_PCA = reshape(permute(probe2, [1, 3, 2]), [], size(probe2, 2));
probe3_PCA = reshape(permute(probe3, [1, 3, 2]), [], size(probe3, 2));

% Normalize the neural data prior to PCA
P1_PCA = (probe1_PCA - mean(probe1_PCA)) ./ std(probe1_PCA);
P2_PCA = (probe2_PCA - mean(probe2_PCA)) ./ std(probe2_PCA);
P3_PCA = (probe3_PCA - mean(probe3_PCA)) ./ std(probe3_PCA);

% PCA on probe1
num_PCs = 10;
[coeff1, score1, latent1, tsquared1, explained1, mu1] = pca(P1_PCA);
score1 = score1(:, 1:num_PCs);
score1_reshaped = reshape(score1, num_timepoints1, num_trials1, num_PCs);

% PCA on probe2
% a = input("LIMITED PCS FOR SPECIFIC 13D SESSION");
num_PCs = 10;
[coeff2, score2, latent2, tsquared2, explained2, mu2] = pca(P2_PCA);
score2 = score2(:, 1:num_PCs);
score2_reshaped = reshape(score2, num_timepoints2, num_trials2, num_PCs);

% PCA on probe3
% a = input("LIMITED PCS FOR SPECIFIC 13D SESSION");
num_PCs = 10;
[coeff3, score3, latent3, tsquared3, explained3, mu3] = pca(P3_PCA);
score3 = score3(:, 1:num_PCs);
score3_reshaped = reshape(score3, num_timepoints3, num_trials3, num_PCs);

% Separate PCs by trial type and filter by valid trials
% Probe1_PCs_R1 = score1_reshaped(:, R1_Trials, :);
% Probe1_PCs_R4 = score1_reshaped(:, R4_Trials, :);
% 
% Probe2_PCs_R1 = score2_reshaped(:, R1_Trials, :);
% Probe2_PCs_R4 = score2_reshaped(:, R4_Trials, :);
% 
% Probe3_PCs_R1 = score3_reshaped(:, R1_Trials, :);
% Probe3_PCs_R4 = score3_reshaped(:, R4_Trials, :);

nR4 = size(probe1_R4, 3);
nR1 = size(probe1_R1, 3);

Probe1_PCs_R4 = score1_reshaped(:, 1:nR4, :);
Probe1_PCs_R1 = score1_reshaped(:, nR4+1:end, :);

Probe2_PCs_R4 = score2_reshaped(:, 1:nR4, :);
Probe2_PCs_R1 = score2_reshaped(:, nR4+1:end, :);

Probe3_PCs_R4 = score3_reshaped(:, 1:nR4, :);
Probe3_PCs_R1 = score3_reshaped(:, nR4+1:end, :);

% Probe1_PCs_R1(:, trials2removeR1, :) = [];
% Probe2_PCs_R1(:, trials2removeR1, :) = [];
% Probe3_PCs_R1(:, trials2removeR1, :) = [];
% 
% Probe1_PCs_R4(:, trials2removeR4, :) = [];
% Probe2_PCs_R4(:, trials2removeR4, :) = [];
% Probe3_PCs_R4(:, trials2removeR4, :) = [];

% Reshape into matrices for Uncut storage
Probe1_PCs_R1_Uncut = reshape(Probe1_PCs_R1, [], size(Probe1_PCs_R1, 3));
Probe1_PCs_R4_Uncut = reshape(Probe1_PCs_R4, [], size(Probe1_PCs_R4, 3));

Probe2_PCs_R1_Uncut = reshape(Probe2_PCs_R1, [], size(Probe2_PCs_R1, 3));
Probe2_PCs_R4_Uncut = reshape(Probe2_PCs_R4, [], size(Probe2_PCs_R4, 3));

Probe3_PCs_R1_Uncut = reshape(Probe3_PCs_R1, [], size(Probe3_PCs_R1, 3));
Probe3_PCs_R4_Uncut = reshape(Probe3_PCs_R4, [], size(Probe3_PCs_R4, 3));

% Reshape to input into chopping procedure
Probe1_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R4, [1, 3, 2]), LRCs_R4_clean, SR);
Probe2_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R4, [1, 3, 2]), LRCs_R4_clean, SR);
Probe3_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe3_PCs_R4, [1, 3, 2]), LRCs_R4_clean, SR);

Probe1_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R1, [1, 3, 2]), LRCs_R1_clean, SR);
Probe2_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R1, [1, 3, 2]), LRCs_R1_clean, SR);
Probe3_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe3_PCs_R1, [1, 3, 2]), LRCs_R1_clean, SR);


%% Compare Z-scored vs Mean-subtracted PCA

% % --- Mean-subtract only (no scaling) ---
% P1_PCA_meansub = probe1_PCA - mean(probe1_PCA, 1);
% P2_PCA_meansub = probe2_PCA - mean(probe2_PCA, 1);
% P3_PCA_meansub = probe3_PCA - mean(probe3_PCA, 1);
% 
% [coeff1_ms, score1_ms, ~] = pca(P1_PCA_meansub, 'Centered', false);  % already centered
% [coeff2_ms, score2_ms, ~] = pca(P2_PCA_meansub, 'Centered', false);
% [coeff3_ms, score3_ms, ~] = pca(P3_PCA_meansub, 'Centered', false);
% 
% score1_ms = score1_ms(:, 1:num_PCs);
% score2_ms = score2_ms(:, 1:num_PCs);
% score3_ms = score3_ms(:, 1:num_PCs);
% 
% score1_ms_reshaped = reshape(score1_ms, num_timepoints1, num_trials1, num_PCs);
% score2_ms_reshaped = reshape(score2_ms, num_timepoints2, num_trials2, num_PCs);
% score3_ms_reshaped = reshape(score3_ms, num_timepoints3, num_trials3, num_PCs);
% 
% % Separate by trial type
% P1_ms_R1 = score1_ms_reshaped(:, R1_Trials, :);
% P1_ms_R4 = score1_ms_reshaped(:, R4_Trials, :);
% P2_ms_R1 = score2_ms_reshaped(:, R1_Trials, :);
% P2_ms_R4 = score2_ms_reshaped(:, R4_Trials, :);
% P3_ms_R1 = score3_ms_reshaped(:, R1_Trials, :);
% P3_ms_R4 = score3_ms_reshaped(:, R4_Trials, :);
% 
% P1_ms_R1(:, trials2removeR1, :) = [];
% P1_ms_R4(:, trials2removeR4, :) = [];
% P2_ms_R1(:, trials2removeR1, :) = [];
% P2_ms_R4(:, trials2removeR4, :) = [];
% P3_ms_R1(:, trials2removeR1, :) = [];
% P3_ms_R4(:, trials2removeR4, :) = [];
% 
% % Add pregc-zscored PCA
% 
% % Use the already-computed probe1_R1_norm / probe1_R4_norm (pregc zscored, -1 to 5s)
% % These are [T x N x trials], need to reshape to (T*trials) x N for PCA
% 
% % Concatenate R1 and R4 together so PCA space is shared
% probe1_all_norm = cat(3, probe1_R1_norm, probe1_R4_norm);
% probe2_all_norm = cat(3, probe2_R1_norm, probe2_R4_norm);
% probe3_all_norm = cat(3, probe3_R1_norm, probe3_R4_norm);
% 
% [T_norm, N1, ~] = size(probe1_all_norm);
% [~,      N2, ~] = size(probe2_all_norm);
% [~,      N3, ~] = size(probe3_all_norm);
% n_trials_R1_p1  = size(probe1_R1_norm, 3);
% n_trials_R4_p1  = size(probe1_R4_norm, 3);
% n_trials_R1_p2  = size(probe2_R1_norm, 3);
% n_trials_R4_p2  = size(probe2_R4_norm, 3);
% n_trials_R1_p3  = size(probe3_R1_norm, 3);
% n_trials_R4_p3  = size(probe3_R4_norm, 3);
% 
% % Reshape to (T*trials) x N
% probe1_norm_PCA = reshape(permute(probe1_all_norm, [1,3,2]), [], N1);
% probe2_norm_PCA = reshape(permute(probe2_all_norm, [1,3,2]), [], N2);
% probe3_norm_PCA = reshape(permute(probe3_all_norm, [1,3,2]), [], N3);
% 
% % PCA (mean subtract internally via default 'Centered', true)
% [~, score1_norm, ~] = pca(probe1_norm_PCA);
% [~, score2_norm, ~] = pca(probe2_norm_PCA);
% [~, score3_norm, ~] = pca(probe3_norm_PCA);
% 
% score1_norm = score1_norm(:, 1:num_PCs);
% score2_norm = score2_norm(:, 1:num_PCs);
% score3_norm = score3_norm(:, 1:num_PCs);
% 
% % Reshape back to time x trials x PCs
% n_trials_total_p1 = n_trials_R1_p1 + n_trials_R4_p1;
% n_trials_total_p2 = n_trials_R1_p2 + n_trials_R4_p2;
% n_trials_total_p3 = n_trials_R1_p3 + n_trials_R4_p3;
% 
% score1_norm_reshaped = reshape(score1_norm, T_norm, n_trials_total_p1, num_PCs);
% score2_norm_reshaped = reshape(score2_norm, T_norm, n_trials_total_p2, num_PCs);
% score3_norm_reshaped = reshape(score3_norm, T_norm, n_trials_total_p3, num_PCs);
% 
% % Split back into R1 and R4
% P1_pregc_R1 = score1_norm_reshaped(:, 1:n_trials_R1_p1, :);
% P1_pregc_R4 = score1_norm_reshaped(:, n_trials_R1_p1+1:end, :);
% 
% P2_pregc_R1 = score2_norm_reshaped(:, 1:n_trials_R1_p2, :);
% P2_pregc_R4 = score2_norm_reshaped(:, n_trials_R1_p2+1:end, :);
% 
% P3_pregc_R1 = score3_norm_reshaped(:, 1:n_trials_R1_p3, :);
% P3_pregc_R4 = score3_norm_reshaped(:, n_trials_R1_p3+1:end, :);
% 
% --- Plotting parameters ---
% n_pcs_to_plot = 1;
% n_single_trials = 5;  % how many single trials to show
% 
% single_trial_alpha = 0.25;
% colors = struct('R1', [0.2 0.5 0.9], 'R4', [0.9 0.3 0.2]);

% %% Plot Z-scored PCs — 3 probes
% n_pcs_to_plot = 10;
% n_single_trials = 5;  % how many single trials to show
% 
% single_trial_alpha = 0.25;
% colors = struct('R1', [0.2 0.5 0.9], 'R4', [0.9 0.3 0.2]);
% 
% t_axis_full = linspace(-1, 5, num_timepoints1);
% 
% probe_configs = {
%     'Probe 1', Probe1_PCs_R1, Probe1_PCs_R4;
%     'Probe 2', Probe2_PCs_R1, Probe2_PCs_R4;
%     'Probe 3', Probe3_PCs_R1, Probe3_PCs_R4;
% };
% 
% for pc = 1:n_pcs_to_plot
% 
%     figure('Position', [100 100 1400 400]);
%     sgtitle(sprintf('PC %d — Z-scored (Global)', pc), 'FontSize', 14);
% 
%     ax_handles = gobjects(1, 3);
% 
%     for pr = 1
%         probe_label = probe_configs{pr, 1};
%         R1_dat      = probe_configs{pr, 2};
%         R4_dat      = probe_configs{pr, 3};
% 
%         ax = subplot(1, 3, pr);
%         ax_handles(pr) = ax;
% 
%         % Single trials
%         trial_idx = randperm(size(R1_dat, 2), min(n_single_trials, size(R1_dat, 2)));
%         for tr = trial_idx
%             plot(t_axis_full, R1_dat(:,tr,pc), 'Color', [colors.R1, single_trial_alpha]); hold on;
%         end
%         trial_idx = randperm(size(R4_dat, 2), min(n_single_trials, size(R4_dat, 2)));
%         for tr = trial_idx
%             plot(t_axis_full, R4_dat(:,tr,pc), 'Color', [colors.R4, single_trial_alpha]); hold on;
%         end
% 
%         % Means
%         plot(t_axis_full, mean(R1_dat(:,:,pc), 2), 'Color', colors.R1, 'LineWidth', 2.5);
%         plot(t_axis_full, mean(R4_dat(:,:,pc), 2), 'Color', colors.R4, 'LineWidth', 2.5);
% 
%         xline(0, 'k--');
%         ylabel('PC score');
%         xlabel('Time from GC (s)');
%         title(probe_label);
%         legend('','','R1 mean','R4 mean', 'Location', 'northwest');
%     end
% 
%     % Link y-axes across probes for same-scale comparison
%     linkaxes(ax_handles, 'y');
% 
% end


%% Get the tongue length, FLCs, and LRCs times to save
% Filter Tongue Length
R1_Tongue_Uncut = all_length(pre_gc_points-SR+1:end, R1_Trials);  % -1s through 4s (500 points)
R4_Tongue_Uncut = all_length(pre_gc_points-SR+1:end, R4_Trials);

R1_Tongue_Uncut(:,trials2removeR1) = [];
R4_Tongue_Uncut(:,trials2removeR4) = [];

R1_Tongue_Uncut(:,bad_trials_R1) = [];
R4_Tongue_Uncut(:,bad_trials_R4) = [];

FCs_Adj_R1 = ceil(FCs_R1_clean*SR + SR);
FCs_Adj_R4 = ceil(FCs_R4_clean*SR + SR);

SCs_Adj_R1 = ceil(SCs_R1_clean*SR + SR);
SCs_Adj_R4 = ceil(SCs_R4_clean*SR + SR);

Fourth_C_Adj_R1 = ceil(Fourth_C_R1_clean*SR + SR);
Fourth_C_Adj_R4 = ceil(Fourth_C_R4_clean*SR + SR);

LRCs_Adj_R1 = ceil(LRCs_R1_clean*SR + SR);
LRCs_Adj_R4 = ceil(LRCs_R4_clean*SR + SR);

% %% -- Get Jaw Data and Filter It -- %
% % Get jaw data
% kinfeat = 'jaw_ydisp_view1'; % Specify the kinematic feature USE Y IN REAL
% % condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
% kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
% jaw = kin(sessix).dat(pre_gc_points-SR+1:end, condtrix, kinix); % Extract jaw length
% 
% kinfeat = 'jaw_yvel_view1';
% % condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
% kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
% jaw_vel = kin(sessix).dat(pre_gc_points-SR+1:end, condtrix, kinix); % Extract jaw length
% 
% % Look at jaw for R4 trials then remove filtered trials
% jaw_R4 = jaw(:, R4_Trials);
% jaw_R4(:, trials2removeR4) = [];
% 
% jaw_vel_R4 = jaw_vel(:, R4_Trials);
% jaw_vel_R4(:, trials2removeR4) = [];
% 
% jaw_R1 = jaw(:, R1_Trials);
% jaw_R1(:, trials2removeR1) = [];
% 
% jaw_vel_R1 = jaw_vel(:, R1_Trials);
% jaw_vel_R1(:, trials2removeR1) = [];
% 
% % Normalize the jaw data
% [jaw_R4, jaw_vel_R4, jaw_R1, Jaw_vel_R1] = deal(normalize_trials(jaw_R4, "zscore"), normalize_trials(jaw_vel_R4, "zscore"), normalize_trials(jaw_R1, "zscore"), normalize_trials(jaw_vel_R1, "zscore"));


% %% Create kin features
% uncut_JawR1 = reshape(jaw_R1, [], 1);
% 
% uncut_JawR4 = reshape(jaw_R4, [], 1);
% 
% uncut_JawvelR1 = reshape(jaw_vel_R1, [], 1);
% 
% uncut_JawvelR4 = reshape(jaw_vel_R4, [], 1);
% 
% jawfeats_R1_Uncut = [uncut_JawR1, uncut_JawvelR1];
% jawfeats_R4_Uncut = [uncut_JawR4, uncut_JawvelR4];
% 
% %% Chop up the jaw feats
% % Reshape for our cutting function
% jaw_R1 = reshape(jaw_R1, size(jaw_R1,1), 1, size(jaw_R1,2));  % becomes 600 x 1 x 152
% jaw_R4 = reshape(jaw_R4, size(jaw_R4,1), 1, size(jaw_R4,2));
% 
% jaw_vel_R1 = reshape(jaw_vel_R1, size(jaw_vel_R1,1), 1, size(jaw_vel_R1,2));
% jaw_vel_R4 = reshape(jaw_vel_R4, size(jaw_vel_R4,1), 1, size(jaw_vel_R4,2));
% 
% jaw_R1_cut = chop_and_stack_neural_data(jaw_R1, LRCs_R1_clean, SR);
% jaw_vel_R1_cut = chop_and_stack_neural_data(jaw_vel_R1, LRCs_R1_clean, SR);
% 
% jaw_R4_cut = chop_and_stack_neural_data(jaw_R4, LRCs_R4_clean, SR);
% jaw_vel_R4_cut = chop_and_stack_neural_data(jaw_vel_R4, LRCs_R4_clean, SR);
% 
% jawfeats_R1_Cut = [uncut_JawR1, uncut_JawvelR1];
% jawfeats_R4_Cut = [uncut_JawR4, uncut_JawvelR4];

%% -- Construct the Output File -- %%
sessionName = string(meta.anm);
sessionDate = string(meta.date);
baseFolder = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\TDsa12';
outputFolder = fullfile(baseFolder, sessionName + "_" + sessionDate);

if ~exist(char(outputFolder), 'dir')
    mkdir(char(outputFolder));
end

%%

% Save trial indices for each condition
csvwrite(fullfile(outputFolder, "R1_Trial_Track.csv"), R1_Trial_Track);
csvwrite(fullfile(outputFolder, "R4_Trial_Track.csv"), R4_Trial_Track);

% Save key point features
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Uncut.csv"), R1_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Uncut.csv"), R4_Keypoints_Uncut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Cut.csv"), R1_Keypoints_Cut);
csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Cut.csv"), R4_Keypoints_Cut);

% % Save Neural FRs
% csvwrite(fullfile(outputFolder, "Probe1_R1_Uncut.csv"), probe1_R1_Uncut);
% csvwrite(fullfile(outputFolder, "Probe1_R4_Uncut.csv"), probe1_R4_Uncut);
% csvwrite(fullfile(outputFolder, "Probe1_R1_Cut.csv"), probe1_R1_Cut);
% csvwrite(fullfile(outputFolder, "Probe1_R4_Cut.csv"), probe1_R4_Cut);
% 
% 
% csvwrite(fullfile(outputFolder, "Probe2_R1_Uncut.csv"), probe2_R1_Uncut);
% csvwrite(fullfile(outputFolder, "Probe2_R4_Uncut.csv"), probe2_R4_Uncut);
% csvwrite(fullfile(outputFolder, "Probe2_R1_Cut.csv"), probe2_R1_Cut);
% csvwrite(fullfile(outputFolder, "Probe2_R4_Cut.csv"), probe2_R4_Cut);
% 
% csvwrite(fullfile(outputFolder, "Probe3_R1_Uncut.csv"), probe3_R1_Uncut);
% csvwrite(fullfile(outputFolder, "Probe3_R4_Uncut.csv"), probe3_R4_Uncut);
% csvwrite(fullfile(outputFolder, "Probe3_R1_Cut.csv"), probe3_R1_Cut);
% csvwrite(fullfile(outputFolder, "Probe3_R4_Cut.csv"), probe3_R4_Cut);

% Save Neural PCs
csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Uncut.csv"), Probe1_PCs_R1_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Uncut.csv"), Probe1_PCs_R4_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Cut.csv"), Probe1_PCs_R1_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Cut.csv"), Probe1_PCs_R4_Cut);


csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Uncut.csv"), Probe2_PCs_R1_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Uncut.csv"), Probe2_PCs_R4_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Cut.csv"), Probe2_PCs_R1_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Cut.csv"), Probe2_PCs_R4_Cut);

csvwrite(fullfile(outputFolder, "PCA_Probe3_R1_Uncut.csv"), Probe3_PCs_R1_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe3_R4_Uncut.csv"), Probe3_PCs_R4_Uncut);
csvwrite(fullfile(outputFolder, "PCA_Probe3_R1_Cut.csv"), Probe3_PCs_R1_Cut);
csvwrite(fullfile(outputFolder, "PCA_Probe3_R4_Cut.csv"), Probe3_PCs_R4_Cut);

% Save Jaw Feats
% csvwrite(fullfile(outputFolder, "JawFeats_R1_Uncut.csv"), jawfeats_R1_Uncut);
% csvwrite(fullfile(outputFolder, "JawFeats_R4_Uncut.csv"), jawfeats_R4_Uncut);
% 
% csvwrite(fullfile(outputFolder, "JawFeats_R1_Cut.csv"), jawfeats_R1_Cut);
% csvwrite(fullfile(outputFolder, "JawFeats_R4_Cut.csv"), jawfeats_R4_Cut);

% Save Tongue Length for visualizations
csvwrite(fullfile(outputFolder, "Tongue_R1.csv"), R1_Tongue_Uncut);
csvwrite(fullfile(outputFolder, "Tongue_R4.csv"), R4_Tongue_Uncut);

% Save FCs and LRCs
csvwrite(fullfile(outputFolder, "FCs_R1.csv"), FCs_Adj_R1);
csvwrite(fullfile(outputFolder, "FCs_R4.csv"), FCs_Adj_R4);

% csvwrite(fullfile(outputFolder, "SCs_R1.csv"), SCs_Adj_R1);
% csvwrite(fullfile(outputFolder, "SCs_R4.csv"), SCs_Adj_R4);
% 
% csvwrite(fullfile(outputFolder, "Fourth_C_R1.csv"), Fourth_C_Adj_R1);
% csvwrite(fullfile(outputFolder, "Fourth_C_R4.csv"), Fourth_C_Adj_R4);

csvwrite(fullfile(outputFolder, "LRCs_R1.csv"), LRCs_Adj_R1);
csvwrite(fullfile(outputFolder, "LRCs_R4.csv"), LRCs_Adj_R4);

% Save metadata as a .txt file for record-keepingela
metadataFile = fullfile(outputFolder, 'metadata.txt');
fid = fopen(metadataFile, 'w');
fprintf(fid, 'Processing Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Script Name: %s\n', mfilename('fullpath'));
fprintf(fid, 'Session ID: %s\n', sessionName);
fprintf(fid, 'Session Date: %s\n', sessionDate);
fclose("all")

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

% Changed 4/20/2026
function zscored_data = zscore_pregc(data, pre_gc_points, SR)
    data_pregc = data(1:pre_gc_points,:,:);
    % Concatenate trials along the second dimension (time dimension)
    concatenated_data = reshape(data_pregc, [], size(data_pregc, 2));
    
    % Compute mean and standard deviation for each neuron
    neuron_means = mean(concatenated_data, 1);
    neuron_stds = std(concatenated_data, 0, 1);
    
    % Z-score the full dataset
    % zscored_data = (data - neuron_means ./ ...
    %                     neuron_stds);

    zscored_data = (data - neuron_means) ./ neuron_stds;

    % chop the data to the gc-1s to end
    S = SR*2;
    zscored_data = zscored_data(pre_gc_points-S+1:end, :, :);
end

% Function to apply z-score normalization to each trial
function normalized = zscore_trials(data, means, stds)
    % data: [T x D x N]
    sz = size(data);
    normalized = (data - reshape(means, 1, [], 1)) ./ reshape(stds, 1, [], 1);
end
