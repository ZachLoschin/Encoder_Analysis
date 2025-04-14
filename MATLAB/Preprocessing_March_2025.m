% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% March 2025
% Cleaning up preprocessing file for MC engagement analysis

%% Finding "Kinematic Modes"
clear,clc

d = 'C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\Disengagement_Analysis_2025';
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
params.tmin = -5;
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
datapth = 'C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\Disengagement_Analysis_2025\processed sessions\r14';
meta = [];
meta = loadTD(meta,datapth);
params.probe = {meta.probe}; 

%% LOAD DATA
[obj,params] = loadSessionData(meta,params,params.behav_only);
% [obj,params] = loadSessionData(meta,params);

for sessix = 1:numel(meta)
    me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
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

% Subtract 1 second from each number then filter greater than 0 to get only
% after GC

% This is unecessary

% for i = 1:obj.bp.Ntrials
%     contacts = all_contacts{i}; % Extract the vector directly
%     contacts = contacts - 1; % Subtract 1 from each element
%     contacts = contacts(contacts > 0); % Keep only positive elements
%     all_contacts{i} = contacts; % Store back into the cell array
% end


%% -- Filter Tongue Length and LP Contacts by Trial Type -- %%
% Get trial IDs
R1_Trials = params.trialid{8};
R4_Trials = params.trialid{9};

% Filter Tongue Length
% Remove first second of data because its before gc
R1_Tongue = all_length(pre_gc_points-100+1:end, R1_Trials);  % 100 before gc to end of trial (600 points)
R4_Tongue = all_length(pre_gc_points-100+1:end, R4_Trials);  % 100 before gc to end of trial (600 points)

% Filter LP Contacts
R1_Contacts = all_contacts(R1_Trials);  % these contacts are relative to the gc
R4_Contacts = all_contacts(R4_Trials);  % these contacts are relative to the gc

%% - Filter and Chop by Good Trial Criteria -- %%
% Chop after last relevant lick
% Filter if there are not at least 8 licks

R4_Contacts;
R4_Tongue;

num_R4trials = length(R4_Contacts);
trials2remove_R4 = [];
first_lick_R4 = [];
trial_end_R4 = [];
time_max = 480;


for tr = 1:num_R4trials
    % Get specific trial contacts and take all after GC
    trial_contacts = R4_Contacts{tr};
    trial_contacts = trial_contacts(trial_contacts>0);
    trial_contacts = trial_contacts(trial_contacts<time_max/100);

    % Remove trials with low # of LP contacts after GC
    if length(trial_contacts) < 6
        trials2remove_R4 = [trials2remove_R4, tr];
        continue
    end

    % Find last relevant lick
    median_ILI = median(diff(trial_contacts));
    threshold = 2*median_ILI;

    offense = diff(trial_contacts) > threshold;

    if any(offense)
        id = find(offense==1);
        id = id(1);
        if id < 6  % Remove trials that dont have 8 consecutive good licks
            trials2remove_R4 = [trials2remove_R4, tr];
            continue
        end
        stop = (trial_contacts(offense(id)) * 100) + 110;
    else
        stop = (trial_contacts(end) * 100) + 110;
    end
    
    if stop < 200  % last check to ensure longer trials
        trials2remove_R4 = [trials2remove_R4, tr];
        continue
    end
    disp(stop)
    first_lick_R4 = [first_lick_R4, trial_contacts(1)];
    trial_end_R4 = [trial_end_R4, stop];
end


R4_Tongue(:, trials2remove_R4) = [];
R4_Contacts(trials2remove_R4) = [];






% R1's turn
num_R1trials = length(R1_Contacts);
trials2remove_R1 = [];
first_lick_R1 = [];
trial_end_R1 = [];

for tr = 1:num_R1trials
    % Get specific trial contacts and take all after GC
    trial_contacts = R1_Contacts{tr};
    trial_contacts = trial_contacts(trial_contacts>0);
    trial_contacts = trial_contacts(trial_contacts<time_max/100);

    % Remove trials with low # of LP contacts after GC
    if length(trial_contacts) < 3
        disp("i")
        trials2remove_R1 = [trials2remove_R1, tr];
        continue
    end

    % Find last relevant lick
    median_ILI = median(diff(trial_contacts));
    threshold = 2*median_ILI;

    offense = diff(trial_contacts) > threshold;

    if any(offense)
        id = find(offense==1);
        id = id(1);
        if id < 3  % Remove trials that dont have 8 consecutive good licks
            disp("o")
            trials2remove_R1 = [trials2remove_R1, tr];
            continue
        end
        stop = (trial_contacts(offense(id)) * 100) + 110;
    else
        stop = (trial_contacts(end) * 100) + 110;
    end
    
    if stop < 200  % last check to ensure longer trials
        disp("q")
        trials2remove_R1 = [trials2remove_R1, tr];
        continue
    end

    first_lick_R1 = [first_lick_R1, trial_contacts(1)];
    trial_end_R1 = [trial_end_R1, stop];
end


R1_Tongue(:, trials2remove_R1) = [];
R1_Contacts(trials2remove_R1) = [];


%% -- Get Jaw Data and Filter It -- %
% Get jaw data
kinfeat = 'jaw_ydisp_view1'; % Specify the kinematic feature USE Y IN REAL
condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
jaw = kin(sessix).dat(pre_gc_points-100+1:end, condtrix, kinix); % Extract jaw length

kinfeat = 'jaw_yvel_view1';
condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
jaw_vel = kin(sessix).dat(pre_gc_points-100+1:end, condtrix, kinix); % Extract jaw length

% try out tongue_length here too.


% dont use these below
% kinfeat = 'jaw_ydisp_view2';
kinfeat = 'tongue_length';
condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
jaw_v2 = kin(sessix).dat(pre_gc_points-100+1:end, condtrix, kinix); % Extract jaw length

kinfeat = 'jaw_yvel_view2';
condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
jaw_vel_v2 = kin(sessix).dat(pre_gc_points-100+1:end, condtrix, kinix); % Extract jaw length

kinfeat = 'motion_energy';
condtrix = params(sessix).trialid{conds2use}; % Get the trials from this condition
kinix = find(strcmp(kin(sessix).featLeg, kinfeat)); % Find index of the kinematic feature
ME = kin(sessix).dat(pre_gc_points-100+1:end, condtrix, kinix); % Extract jaw length

% Look at jaw for R4 trials then remove filtered trials
jaw_R4 = jaw(:, R4_Trials);
jaw_R4(:, trials2remove_R4) = [];
% 
jaw_R4_v2 = jaw_v2(:, R4_Trials);
jaw_R4_v2(:, trials2remove_R4) = [];

ME_R4 = ME(:, R4_Trials);
ME_R4(:, trials2remove_R4) = [];

jaw_vel_R4 = jaw_vel(:, R4_Trials);
jaw_vel_R4(:, trials2remove_R4) = [];

jaw_vel_R4_v2 = jaw_vel(:, R4_Trials);
jaw_vel_R4_v2(:, trials2remove_R4) = [];

jaw_R1 = jaw(:, R1_Trials);
jaw_R1(:, trials2remove_R1) = [];

jaw_R1_v2 = jaw_v2(:, R1_Trials);
jaw_R1_v2(:, trials2remove_R1) = [];

ME_R1 = ME(:, R1_Trials);
ME_R1(:, trials2remove_R1) = [];

jaw_vel_R1 = jaw_vel(:, R1_Trials);
jaw_vel_R1(:, trials2remove_R1) = [];

jaw_vel_R1_v2 = jaw_vel(:, R1_Trials);
jaw_vel_R1_v2(:, trials2remove_R1) = [];

%% Normalize the kinematics data across all trials
% Concatenate trials, normalize, and reshape back for each variable
[jaw_R4, jaw_R4_v2, jaw_vel_R4, jaw_vel_R4_v2, ...
 jaw_R1, jaw_R1_v2, jaw_vel_R1, jaw_vel_R1_v2, ...
 ME_R1, ME_R4] = deal(normalize_trials(jaw_R4, "zscore"), normalize_trials(jaw_R4_v2, "zscore"), ...
                     normalize_trials(jaw_vel_R4, "zscore"), normalize_trials(jaw_vel_R4_v2, "zscore"), ...
                     normalize_trials(jaw_R1, "zscore"), normalize_trials(jaw_R1_v2, "zscore"), ...
                     normalize_trials(jaw_vel_R1, "zscore"), normalize_trials(jaw_vel_R1_v2, "zscore"), ...
                     normalize_trials(ME_R1, "zscore"), normalize_trials(ME_R4, "zscore"));



%% Create kin features
uncut_JawR1 = reshape(jaw_R1, [], 1);
uncut_JawR1_v2 = reshape(jaw_R1_v2, [], 1);

uncut_JawR4 = reshape(jaw_R4, [], 1);
uncut_JawR4_v2 = reshape(jaw_R4_v2, [], 1);

uncut_JawvelR1 = reshape(jaw_vel_R1, [], 1);
uncut_JawvelR1_v2 = reshape(jaw_vel_R1_v2, [], 1);

uncut_JawvelR4 = reshape(jaw_vel_R4, [], 1);
uncut_JawvelR4_v2 = reshape(jaw_vel_R4_v2, [], 1);

uncut_ME_R1 = reshape(ME_R1, [], 1);
uncut_ME_R4 = reshape(ME_R4, [], 1);

jawfeats_R1 = [uncut_JawR1, uncut_JawvelR1, uncut_JawR1_v2, uncut_JawvelR1_v2, uncut_ME_R1];
jawfeats_R4 = [uncut_JawR4, uncut_JawvelR4, uncut_JawR4_v2, uncut_JawvelR4_v2, uncut_ME_R4];

%% -- Get Region Specific Neural Data and Filter It-- %%
% Tudor code for separating probe 1 and 2
Ncells = size(obj.psth, 2);
probe1 = 1:numel(params.cluid{1, 1});
probe2 = size(params.cluid{1, 1},1)+1:Ncells;

probe1_trialdat = obj.trialdat(:,probe1, :);
probe2_trialdat = obj.trialdat(:,probe2, :);

% Limit to the desired trials
probe1_R4 = probe1_trialdat(:,:,R4_Trials);
probe2_R4 = probe2_trialdat(:,:,R4_Trials);

probe1_R1 = probe1_trialdat(:,:,R1_Trials);
probe2_R1 = probe2_trialdat(:,:,R1_Trials);

% Remove filtered trials
probe1_R4(:,:,trials2remove_R4) = [];
probe2_R4(:,:,trials2remove_R4) = [];

probe1_R1(:,:,trials2remove_R1) = [];
probe2_R1(:,:,trials2remove_R1) = [];


%% Normalize the neural data to the baseline period

probe1_R4 = zscore_pregc(probe1_R4, pre_gc_points);
probe1_R1 = zscore_pregc(probe1_R1, pre_gc_points);
probe2_R4 = zscore_pregc(probe2_R4, pre_gc_points);
probe2_R1 = zscore_pregc(probe2_R1, pre_gc_points);



%% -- Chop Jaw and Tongue Data to Trial Ends -- %%
% Round to indices
trial_end_R4 = floor(trial_end_R4);
[~,num_trials_R4] = size(trial_end_R4);

trial_end_R1 = floor(trial_end_R1);
[~,num_trials_R1] = size(trial_end_R1);


% Remove nans and normalize jaw data
jaw_R4(isnan(jaw_R4)) = 0;
jaw_R4 = normalize(jaw_R4, 'zscore');

jaw_vel_R4(isnan(jaw_vel_R4)) = 0;
jaw_vel_R4 = normalize(jaw_vel_R4, 'zscore');

jaw_R1(isnan(jaw_R1)) = 0;
jaw_R1 = normalize(jaw_R1, 'zscore');

jaw_vel_R1(isnan(jaw_vel_R1)) = 0;
jaw_vel_R1 = normalize(jaw_vel_R1, 'zscore');


R4_Tongue(isnan(R4_Tongue)) = 0;
R4_Tongue = normalize(R4_Tongue, 'zscore');

R1_Tongue(isnan(R1_Tongue)) = 0;
R1_Tongue = normalize(R1_Tongue, 'zscore');


% Chop the kinematic data
chopped_jaw_R4 = [];
chopped_jawvel_R4 = [];
chopped_tongue_R4 = [];

chopped_jaw_R1 = [];
chopped_jawvel_R1 = [];
chopped_tongue_R1 = [];

for idx = 1:num_trials_R4
    % Jaw chopped
    jaw_trial = jaw_R4(1:trial_end_R4(idx), idx);
    chopped_jaw_R4 = [chopped_jaw_R4; jaw_trial];

    jaw_vel_trial = jaw_vel_R4(1:trial_end_R4(idx), idx);
    chopped_jawvel_R4 = [chopped_jawvel_R4; jaw_vel_trial];

    % Tongue chopped
    tongue_trial = R4_Tongue(1:trial_end_R4(idx), idx);
    chopped_tongue_R4 = [chopped_tongue_R4; tongue_trial];
end

for idx = 1:num_trials_R1
    % Jaw chopped
    jaw_trial = jaw_R1(1:trial_end_R1(idx), idx);
    chopped_jaw_R1 = [chopped_jaw_R1; jaw_trial];

    jaw_vel_trial = jaw_vel_R1(1:trial_end_R1(idx), idx);
    chopped_jawvel_R1 = [chopped_jawvel_R1; jaw_vel_trial];

    % Tongue chopped
    tongue_trial = R1_Tongue(1:trial_end_R1(idx), idx);
    chopped_tongue_R1 = [chopped_tongue_R1; tongue_trial];
end

%% -- Chop the Probe 2 Neural Data to Trial Ends -- %%

[~,num_neurons,num_trials_R4] = size(probe2_R4);
% Calculate total number of rows required
total_rows = sum(trial_end_R4);

% Preallocate the matrix for the chopped neural data
final_neural_data_R4 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R4
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe2_R4(1:trial_end_R4(idx), n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_R4(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row + trial_end_R4(idx);
end






[~,num_neurons,num_trials_R1] = size(probe2_R1);
% Calculate total number of rows required
total_rows = sum(trial_end_R1);

% Preallocate the matrix for the chopped neural data
final_neural_data_R1 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R1
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe2_R1(1:trial_end_R1(idx), n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_R1(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row + trial_end_R1(idx);
end




%% -- Chop the probe 1 Neural Data to Trial Ends -- %%

[~,num_neurons,num_trials_R4] = size(probe1_R4);
% Calculate total number of rows required
total_rows = sum(trial_end_R4)

% Preallocate the matrix for the chopped neural data
final_neural_data_probe1_R4 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R4
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe1_R4(1:trial_end_R4(idx), n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_probe1_R4(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row + trial_end_R4(idx);
end





[~,num_neurons,num_trials_R1] = size(probe1_R1);
% Calculate total number of rows required
total_rows = sum(trial_end_R1)

% Preallocate the matrix for the chopped neural data
final_neural_data_probe1_R1 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R1
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe1_R1(1:trial_end_R1(idx), n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_probe1_R1(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row + trial_end_R1(idx);
end

%% -- Save probe1 Neural Data Full Trial -- %%

[~,num_neurons,num_trials_R4] = size(probe1_R4);
% Calculate total number of rows required
total_rows = sum(trial_end_R4)

% Preallocate the matrix for the chopped neural data
final_neural_data_probe1_uncut_R4 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R4
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe1_R4(:, n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_probe1_uncut_R4(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row +600;
end


[~,num_neurons,num_trials_R1] = size(probe1_R1);
% Calculate total number of rows required
total_rows = sum(trial_end_R1)

% Preallocate the matrix for the chopped neural data
final_neural_data_probe1_uncut_R1 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R1
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe1_R1(:, n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_probe1_uncut_R1(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row +600;
end












%% -- Save probe2 Neural Data Full Trial -- %%

[~,num_neurons,num_trials_R4] = size(probe2_R4);
% Calculate total number of rows required
total_rows = sum(trial_end_R4)

% Preallocate the matrix for the chopped neural data
final_neural_data_probe2_uncut_R4 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R4
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe2_R4(:, n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_probe2_uncut_R4(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row +600;
end


[~,num_neurons,num_trials_R1] = size(probe2_R1);
% Calculate total number of rows required
total_rows = sum(trial_end_R1)

% Preallocate the matrix for the chopped neural data
final_neural_data_probe2_uncut_R1 = NaN(total_rows, num_neurons);

% Initialize the row index
current_row = 1;

% Iterate through trials and neurons to fill the preallocated matrix
for idx = 1:num_trials_R1
    for n_idx = 1:num_neurons
        % Save neural data for each neuron
        neural_trial = probe2_R1(:, n_idx, idx);
        
        % Calculate the end row index for the current trial
        end_row = current_row + length(neural_trial) - 1;
        
        % Write the chopped data into the matrix
        final_neural_data_probe2_uncut_R1(current_row:end_row, n_idx) = neural_trial;
    end
    
    % Update the current row index for the next trial
    current_row = current_row +600;
end

%% -- Construct the Output File -- %%
sessionName = meta.anm;
sessionDate = meta.date;

% Construct the output folder path
outputFolder = fullfile( ...
    'C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\Disengagement_Analysis_2025\preprocessed_data', ...
    [sessionName '_' sessionDate]);

% Create the output folder if it does not exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Save R4 Files


csvwrite(fullfile(outputFolder, "Probe1_R4.csv"), final_neural_data_probe1_R4);
% csvwrite(fullfile(outputFolder, "Probe1_R4_PC.csv"), reduced_data);
csvwrite(fullfile(outputFolder, "Probe2_R4.csv"), final_neural_data_R4);
csvwrite(fullfile(outputFolder, "Jaw_R4.csv"), chopped_jaw_R4);
csvwrite(fullfile(outputFolder, "Tongue_R4.csv"), chopped_tongue_R4);
csvwrite(fullfile(outputFolder, "First_Contact_R4.csv"), first_lick_R4);
csvwrite(fullfile(outputFolder, "Trial_End_R4.csv"), trial_end_R4);
csvwrite(fullfile(outputFolder, "Jaw_feats_R4.csv"), jawfeats_R4);

csvwrite(fullfile(outputFolder, "Probe1_uncut_R4.csv"), final_neural_data_probe1_uncut_R4);
csvwrite(fullfile(outputFolder, "Probe2_uncut_R4.csv"), final_neural_data_probe2_uncut_R4);
csvwrite(fullfile(outputFolder, "Tongue_uncut_R4.csv"), R4_Tongue);
csvwrite(fullfile(outputFolder, "Jaw_uncut_R4.csv"), jaw_R4);

% Save R1 Files
csvwrite(fullfile(outputFolder, "Probe1_R1.csv"), final_neural_data_probe1_R1);
csvwrite(fullfile(outputFolder, "Probe2_R1.csv"), final_neural_data_R1);
csvwrite(fullfile(outputFolder, "Jaw_R1.csv"), chopped_jaw_R1);
csvwrite(fullfile(outputFolder, "Tongue_R1.csv"), chopped_tongue_R1);
csvwrite(fullfile(outputFolder, "First_Contact_R1.csv"), first_lick_R1);
csvwrite(fullfile(outputFolder, "Trial_End_R1.csv"), trial_end_R1);
csvwrite(fullfile(outputFolder, "Jaw_feats_R1.csv"), jawfeats_R1);

csvwrite(fullfile(outputFolder, "Probe1_uncut_R1.csv"), final_neural_data_probe1_uncut_R1);
csvwrite(fullfile(outputFolder, "Probe2_uncut_R1.csv"), final_neural_data_probe2_uncut_R1);
csvwrite(fullfile(outputFolder, "Tongue_uncut_R1.csv"), R1_Tongue);
csvwrite(fullfile(outputFolder, "Jaw_uncut_R1.csv"), jaw_R1);

[d, p1_units] = size(final_neural_data_probe1_R1);
[d, p2_units] = size(final_neural_data_R1);

% Optional: Save metadata as a .txt file for record-keeping
metadataFile = fullfile(outputFolder, 'metadata.txt');
fid = fopen(metadataFile, 'w');
fprintf(fid, 'Processing Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Script Name: %s\n', mfilename('fullpath'));
fprintf(fid, 'Session ID: %s\n', sessionName);
fprintf(fid, 'Session Date: %s\n', sessionDate);
fprintf(fid, '0-5s after go cue, 1/100 sampling, trial needs 8 consecutive good licks, cut at last relevant lick \n')
fprintf(fid, 'Probe 1 # Clusters: %d\n', p1_units)
fprintf(fid, 'Probe 2 # Clusters: %d\n', p2_units)
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
    data_pregc = data(1:400,:,:);
    % Concatenate trials along the second dimension (time dimension)
    concatenated_data = reshape(data_pregc, [], size(data_pregc, 2));
    
    % Compute mean and standard deviation for each neuron
    neuron_means = mean(concatenated_data, 1);
    neuron_stds = std(concatenated_data, 0, 1);
    
    % Z-score the full dataset
    zscored_data = (data - neuron_means ./ ...
                        neuron_stds);

    % chop the data to the gc to end
    zscored_data = zscored_data(pre_gc_points-100+1:end, :, :);
end
