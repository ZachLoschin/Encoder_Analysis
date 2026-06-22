% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% May 2025
% Cleaned + Looped preprocessing for MC engagement analysis

%% Setup paths
clear, clc

[~, hostname] = system('hostname');
hostname = strtrim(hostname);

if hostname == "DESKTOP-5JJC0TM"
    d           = "C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Economo-Lab-Preprocessing";
    datapth     = 'C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Data\processed sessions\r14';
    outRoot     = 'C:\Users\zlosc\Documents\GitHub\Encoder_Analysis\Preprocessed_Encoder\R14_2026';
else
    d           = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Economo-Lab-Preprocessing';
    datapth     = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Data\processed sessions\r14';
    outRoot     = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R14_2026';
end

addpath(d)
addpath(genpath(fullfile(d,'utils')))
addpath(genpath(fullfile(d,'zutils')))
addpath(genpath(fullfile(d,'DataLoadingScripts')))
addpath(genpath(fullfile(d,'ObjVis')))
addpath(genpath(fullfile(d,'funcs')))

%% PARAMETERS  (defined once, reused every session)
params_template.alignEvent       = 'goCue';
params_template.behav_only       = 0;
params_template.timeWarp         = 0;
params_template.nLicks           = 20;
params_template.lowFR            = 1.0;

params_template.condition(1)     = {'hit==1 | hit==0'};
params_template.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 1'};
params_template.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 1'};
params_template.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 1'};
params_template.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 4'};
params_template.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 4'};
params_template.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 4'};
params_template.condition(end+1) = {'hit==1 & rewardedLick == 1'};
params_template.condition(end+1) = {'hit==1 & rewardedLick == 4'};
params_template.condition(end+1) = {'hit==1'};

params_template.tmin             = -2;
params_template.tmax             = 5;
params_template.dt               = 1/200;
params_template.smooth           = 10;
params_template.quality          = {'good','fair','excellent','ok'};

params_template.traj_features    = { ...
    {'tongue','left_tongue','right_tongue','jaw','trident','nose'}, ...
    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_nostril','bottom_nostril'}};
params_template.feat_varToExplain = 99;
params_template.N_varToExplain   = 80;
params_template.advance_movement = 0;
params_template.fcut             = 10;
params_template.cond             = 5;
params_template.method           = 'xcorr';
params_template.fa               = false;
params_template.bctype           = 'reflect';

%% Discover sessions from data files
anmName     = 'TDsa9';   % <-- change per animal
dataObjPath = fullfile(datapth, 'DataObjects', anmName);

filePattern  = fullfile(dataObjPath, ['data_structure_' anmName '_*.mat']);
sessionFiles = dir(filePattern);

if isempty(sessionFiles)
    error('No session files found in %s', dataObjPath);
end

sessionDates = cell(numel(sessionFiles), 1);
for i = 1:numel(sessionFiles)
    [~, fname, ~]  = fileparts(sessionFiles(i).name);  % strip .mat
    parts          = split(fname, '_');                 % {'data','structure','TDsa2','2026-02-04'}
    sessionDates{i} = parts{end};                      % '2026-02-04'
end
fprintf('Found %d session(s) for %s.\n', numel(sessionDates), anmName);

%% ============================================================
%  MAIN SESSION LOOP
%% ============================================================
for sessIdx = 1:numel(sessionDates)

    sessionDate = sessionDates{sessIdx};
    folderName  = [anmName '_' sessionDate];  % used only for output folder naming
    
    try

        fprintf('\n=== Processing session %d / %d : %s  %s ===\n', ...
            sessIdx, numel(sessionDates), anmName, sessionDate);
    
        % --- Build meta for this session --------------------------------
        meta = loadTD_session(anmName, sessionDate, datapth);
    
        % --- Params (fresh copy each session) --------------------------
        params        = params_template;
        params.probe  = {meta.probe};
    
        SR            = 1/params.dt;
        pre_gc_points = -params.tmin / params.dt;
    
        % --- Load data --------------------------------------------------
        try
            [obj, params] = loadSessionData(meta, params, params.behav_only);
            me = loadMotionEnergy(obj, meta, params, dataObjPath);
        catch ME_err
            warning('Failed to load data for %s %s: %s — skipping.', anmName, sessionDate, ME_err.message);
            continue
        end
    
        % --- Handle trial-count mismatch --------------------------------
        nTrials_neural    = size(obj.traj{1}, 2);
        nTrials_kinematic = size(me.data, 2);
    
        if nTrials_kinematic > nTrials_neural
            trial_diff     = nTrials_kinematic - nTrials_neural;
            obj.bp.Ntrials = nTrials_neural;
            me.data        = me.data(:, 1:(end-trial_diff));
        else
            trial_diff = 0;
        end
        NTRIALS = nTrials_neural;
    
        % --- Kinematics -------------------------------------------------
        kin = getKinematics(obj, me, params);
        pos = getKeypointsFromVideo(obj, me, params);
    
        % --- Tongue length (all trials) ---------------------------------
        kinfeat   = 'tongue_length';
        conds2use = 1;
        condtrix  = params.trialid{conds2use};
        condtrix  = condtrix(1:(end-trial_diff), :);
    
        kinix      = find(strcmp(kin.featLeg, kinfeat));
        all_length = kin.dat(:, condtrix, kinix);
    
        % --- Lick-port contacts (GC-relative) ---------------------------
        all_contacts = obj.bp.ev.lickL;
        for i = 1:obj.bp.Ntrials
            contacts        = obj.bp.ev.lickL{i,1};
            gc              = obj.bp.ev.goCue(i);
            all_contacts{i} = contacts - gc;
        end
    
        % --- Split by reward lick (R1 / R4) -----------------------------
        R1_Trials = params.trialid{8};
        R4_Trials = params.trialid{9};
        R1_Trials = R1_Trials(R1_Trials <= NTRIALS);
        R4_Trials = R4_Trials(R4_Trials <= NTRIALS);
    
        R1_Trial_Track = R1_Trials;
        R4_Trial_Track = R4_Trials;
    
        R1_Contacts = all_contacts(R1_Trials);
        R4_Contacts = all_contacts(R4_Trials);
    
        % --- Find / filter by lick events -------------------------------
        [trials2removeR1, FCs_R1_clean, SCs_R1_clean, Fourth_C_R1_clean, ~, LRCs_R1_clean] = ...
            filter_trials_by_licking(R1_Contacts, SR, min_licks=3);
        [trials2removeR4, FCs_R4_clean, SCs_R4_clean, Fourth_C_R4_clean, ~, LRCs_R4_clean] = ...
            filter_trials_by_licking(R4_Contacts, SR, min_licks=5);
    
        FCs_R1_clean(trials2removeR1)      = [];
        SCs_R1_clean(trials2removeR1)      = [];
        LRCs_R1_clean(trials2removeR1)     = [];
        Fourth_C_R1_clean(trials2removeR1) = [];
    
        FCs_R4_clean(trials2removeR4)      = [];
        SCs_R4_clean(trials2removeR4)      = [];
        LRCs_R4_clean(trials2removeR4)     = [];
        Fourth_C_R4_clean(trials2removeR4) = [];
    
        R1_Trial_Track(trials2removeR1) = [];
        R4_Trial_Track(trials2removeR4) = [];
    
        % --- Keypoints --------------------------------------------------
        R1_Keypoints = pos(SR+1:end, :, R1_Trials);
        R4_Keypoints = pos(SR+1:end, :, R4_Trials);
        R1_Keypoints(:,:,trials2removeR1) = [];
        R4_Keypoints(:,:,trials2removeR4) = [];
    
        R1_Keypoints_Uncut = reshape(permute(R1_Keypoints,[1,3,2]), [], size(R1_Keypoints,2));
        R4_Keypoints_Uncut = reshape(permute(R4_Keypoints,[1,3,2]), [], size(R4_Keypoints,2));
    
        R1K = (R1_Keypoints_Uncut - mean(R1_Keypoints_Uncut,'omitnan')) ./ std(R1_Keypoints_Uncut,'omitnan');
        R4K = (R4_Keypoints_Uncut - mean(R4_Keypoints_Uncut,'omitnan')) ./ std(R4_Keypoints_Uncut,'omitnan');
    
        [nt1,nk1,ntr1] = size(R1_Keypoints);
        R1K_final = permute(reshape(R1K, nt1, ntr1, nk1), [1,3,2]);
    
        [nt4,nk4,ntr4] = size(R4_Keypoints);
        R4K_final = permute(reshape(R4K, nt4, ntr4, nk4), [1,3,2]);
    
        R1_Keypoints_Uncut = R1K;
        R4_Keypoints_Uncut = R4K;
    
        R1_Keypoints_Cut = chop_and_stack_neural_data(R1K_final, LRCs_R1_clean, SR);
        R4_Keypoints_Cut = chop_and_stack_neural_data(R4K_final, LRCs_R4_clean, SR);
    
        % --- Region-specific neural data --------------------------------
        Ncells_P1 = numel(params.cluid{1,1});
        Ncells_P2 = numel(params.cluid{1,2});
        Ncells_P3 = numel(params.cluid{1,3});
    
        clu_1 = 1:Ncells_P1;
        clu_2 = Ncells_P1+1 : Ncells_P1+Ncells_P2;
        clu_3 = Ncells_P1+Ncells_P2+1 : Ncells_P1+Ncells_P2+Ncells_P3;
    
        probe1_trialdat = obj.trialdat(:, clu_1, :);
        probe2_trialdat = obj.trialdat(:, clu_2, :);
        probe3_trialdat = obj.trialdat(:, clu_3, :);
    
        probe1_R4 = probe1_trialdat(:,:,R4_Trials); probe1_R4(:,:,trials2removeR4) = [];
        probe2_R4 = probe2_trialdat(:,:,R4_Trials); probe2_R4(:,:,trials2removeR4) = [];
        probe3_R4 = probe3_trialdat(:,:,R4_Trials); probe3_R4(:,:,trials2removeR4) = [];
    
        probe1_R1 = probe1_trialdat(:,:,R1_Trials); probe1_R1(:,:,trials2removeR1) = [];
        probe2_R1 = probe2_trialdat(:,:,R1_Trials); probe2_R1(:,:,trials2removeR1) = [];
        probe3_R1 = probe3_trialdat(:,:,R1_Trials); probe3_R1(:,:,trials2removeR1) = [];
    
        % Normalize
        probe1_R4_norm = zscore_pregc(probe1_R4, pre_gc_points, SR);
        probe1_R1_norm = zscore_pregc(probe1_R1, pre_gc_points, SR);
        probe2_R4_norm = zscore_pregc(probe2_R4, pre_gc_points, SR);
        probe2_R1_norm = zscore_pregc(probe2_R1, pre_gc_points, SR);
        probe3_R4_norm = zscore_pregc(probe3_R4, pre_gc_points, SR);
        probe3_R1_norm = zscore_pregc(probe3_R1, pre_gc_points, SR);
    
        % Uncut reshape
        probe1_R4_Uncut = reshape(permute(probe1_R4_norm,[1,3,2]), [], size(probe1_R4_norm,2));
        probe1_R1_Uncut = reshape(permute(probe1_R1_norm,[1,3,2]), [], size(probe1_R1_norm,2));
        probe2_R4_Uncut = reshape(permute(probe2_R4_norm,[1,3,2]), [], size(probe2_R4_norm,2));
        probe2_R1_Uncut = reshape(permute(probe2_R1_norm,[1,3,2]), [], size(probe2_R1_norm,2));
        probe3_R4_Uncut = reshape(permute(probe3_R4_norm,[1,3,2]), [], size(probe3_R4_norm,2));
        probe3_R1_Uncut = reshape(permute(probe3_R1_norm,[1,3,2]), [], size(probe3_R1_norm,2));
    
        % Cut
        probe1_R4_Cut = chop_and_stack_neural_data(probe1_R4_norm, LRCs_R4_clean, SR);
        probe1_R1_Cut = chop_and_stack_neural_data(probe1_R1_norm, LRCs_R1_clean, SR);
        probe2_R4_Cut = chop_and_stack_neural_data(probe2_R4_norm, LRCs_R4_clean, SR);
        probe2_R1_Cut = chop_and_stack_neural_data(probe2_R1_norm, LRCs_R1_clean, SR);
        probe3_R4_Cut = chop_and_stack_neural_data(probe3_R4_norm, LRCs_R4_clean, SR);
        probe3_R1_Cut = chop_and_stack_neural_data(probe3_R1_norm, LRCs_R1_clean, SR);
    
        % --- Neural PCs -------------------------------------------------
        probe1 = probe1_trialdat(SR+1:end, :, :);
        probe2 = probe2_trialdat(SR+1:end, :, :);
        probe3 = probe3_trialdat(SR+1:end, :, :);
    
        [ntp1,~,ntr1_all] = size(probe1);
        [ntp2,~,ntr2_all] = size(probe2);
        [ntp3,~,ntr3_all] = size(probe3);
    
        probe1_PCA = reshape(permute(probe1,[1,3,2]), [], size(probe1,2));
        probe2_PCA = reshape(permute(probe2,[1,3,2]), [], size(probe2,2));
        probe3_PCA = reshape(permute(probe3,[1,3,2]), [], size(probe3,2));
    
        P1_PCA = (probe1_PCA - mean(probe1_PCA)) ./ std(probe1_PCA);
        P2_PCA = (probe2_PCA - mean(probe2_PCA)) ./ std(probe2_PCA);
        P3_PCA = (probe3_PCA - mean(probe3_PCA)) ./ std(probe3_PCA);
    
        num_PCs = 10;
        [~, score1, ~, ~, explained1] = pca(P1_PCA);
        [~, score2, ~, ~, explained2] = pca(P2_PCA);
        [~, score3, ~, ~, explained3] = pca(P3_PCA);
    
        score1_reshaped = reshape(score1(:,1:num_PCs), ntp1, ntr1_all, num_PCs);
        score2_reshaped = reshape(score2(:,1:num_PCs), ntp2, ntr2_all, num_PCs);
        score3_reshaped = reshape(score3(:,1:num_PCs), ntp3, ntr3_all, num_PCs);
    
        Probe1_PCs_R1 = score1_reshaped(:,R1_Trials,:); Probe1_PCs_R1(:,trials2removeR1,:) = [];
        Probe1_PCs_R4 = score1_reshaped(:,R4_Trials,:); Probe1_PCs_R4(:,trials2removeR4,:) = [];
        Probe2_PCs_R1 = score2_reshaped(:,R1_Trials,:); Probe2_PCs_R1(:,trials2removeR1,:) = [];
        Probe2_PCs_R4 = score2_reshaped(:,R4_Trials,:); Probe2_PCs_R4(:,trials2removeR4,:) = [];
        Probe3_PCs_R1 = score3_reshaped(:,R1_Trials,:); Probe3_PCs_R1(:,trials2removeR1,:) = [];
        Probe3_PCs_R4 = score3_reshaped(:,R4_Trials,:); Probe3_PCs_R4(:,trials2removeR4,:) = [];
    
        Probe1_PCs_R1_Uncut = reshape(Probe1_PCs_R1, [], size(Probe1_PCs_R1,3));
        Probe1_PCs_R4_Uncut = reshape(Probe1_PCs_R4, [], size(Probe1_PCs_R4,3));
        Probe2_PCs_R1_Uncut = reshape(Probe2_PCs_R1, [], size(Probe2_PCs_R1,3));
        Probe2_PCs_R4_Uncut = reshape(Probe2_PCs_R4, [], size(Probe2_PCs_R4,3));
        Probe3_PCs_R1_Uncut = reshape(Probe3_PCs_R1, [], size(Probe3_PCs_R1,3));
        Probe3_PCs_R4_Uncut = reshape(Probe3_PCs_R4, [], size(Probe3_PCs_R4,3));
    
        Probe1_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R4,[1,3,2]), LRCs_R4_clean, SR);
        Probe2_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R4,[1,3,2]), LRCs_R4_clean, SR);
        Probe3_PCs_R4_Cut = chop_and_stack_neural_data(permute(Probe3_PCs_R4,[1,3,2]), LRCs_R4_clean, SR);
        Probe1_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe1_PCs_R1,[1,3,2]), LRCs_R1_clean, SR);
        Probe2_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe2_PCs_R1,[1,3,2]), LRCs_R1_clean, SR);
        Probe3_PCs_R1_Cut = chop_and_stack_neural_data(permute(Probe3_PCs_R1,[1,3,2]), LRCs_R1_clean, SR);
    
        % --- Tongue uncut (clean trials only) ---------------------------
        R1_Tongue_Uncut = all_length(pre_gc_points-SR+1:end, R1_Trials);
        R4_Tongue_Uncut = all_length(pre_gc_points-SR+1:end, R4_Trials);
        R1_Tongue_Uncut(:, trials2removeR1) = [];
        R4_Tongue_Uncut(:, trials2removeR4) = [];
    
        % --- Adjusted contact timestamps --------------------------------
        FCs_Adj_R1  = ceil(FCs_R1_clean  * SR + SR);
        FCs_Adj_R4  = ceil(FCs_R4_clean  * SR + SR);
        LRCs_Adj_R1 = ceil(LRCs_R1_clean * SR + SR);
        LRCs_Adj_R4 = ceil(LRCs_R4_clean * SR + SR);
    
        % --- Jaw --------------------------------------------------------
        jaw_disp = kin.dat(pre_gc_points-SR+1:end, condtrix, strcmp(kin.featLeg,'jaw_ydisp_view1'));
        jaw_vel  = kin.dat(pre_gc_points-SR+1:end, condtrix, strcmp(kin.featLeg,'jaw_yvel_view1'));
    
        jaw_R4     = jaw_disp(:,R4_Trials); jaw_R4(:,trials2removeR4)     = [];
        jaw_R1     = jaw_disp(:,R1_Trials); jaw_R1(:,trials2removeR1)     = [];
        jaw_vel_R4 = jaw_vel(:,R4_Trials);  jaw_vel_R4(:,trials2removeR4) = [];
        jaw_vel_R1 = jaw_vel(:,R1_Trials);  jaw_vel_R1(:,trials2removeR1) = [];
    
        jaw_R4     = normalize_trials(jaw_R4,     'zscore');
        jaw_R1     = normalize_trials(jaw_R1,     'zscore');
        jaw_vel_R4 = normalize_trials(jaw_vel_R4, 'zscore');
        jaw_vel_R1 = normalize_trials(jaw_vel_R1, 'zscore');
    
        jawfeats_R1_Uncut = [jaw_R1(:), jaw_vel_R1(:)];
        jawfeats_R4_Uncut = [jaw_R4(:), jaw_vel_R4(:)];
    
        jaw_R1_3d     = reshape(jaw_R1,     size(jaw_R1,1),     1, size(jaw_R1,2));
        jaw_R4_3d     = reshape(jaw_R4,     size(jaw_R4,1),     1, size(jaw_R4,2));
        jaw_vel_R1_3d = reshape(jaw_vel_R1, size(jaw_vel_R1,1), 1, size(jaw_vel_R1,2));
        jaw_vel_R4_3d = reshape(jaw_vel_R4, size(jaw_vel_R4,1), 1, size(jaw_vel_R4,2));
    
        jaw_R1_cut     = chop_and_stack_neural_data(jaw_R1_3d,     LRCs_R1_clean, SR);
        jaw_vel_R1_cut = chop_and_stack_neural_data(jaw_vel_R1_3d, LRCs_R1_clean, SR);
        jaw_R4_cut     = chop_and_stack_neural_data(jaw_R4_3d,     LRCs_R4_clean, SR);
        jaw_vel_R4_cut = chop_and_stack_neural_data(jaw_vel_R4_3d, LRCs_R4_clean, SR);
    
        jawfeats_R1_Cut = [jaw_R1_cut, jaw_vel_R1_cut];
        jawfeats_R4_Cut = [jaw_R4_cut, jaw_vel_R4_cut];
    
        % --- Output folder ----------------------------------------------
        outputFolder = fullfile(outRoot, folderName);
        if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
    
        % --- Save -------------------------------------------------------
        csvwrite(fullfile(outputFolder, "R1_Trial_Track.csv"),          R1_Trial_Track);
        csvwrite(fullfile(outputFolder, "R4_Trial_Track.csv"),          R4_Trial_Track);
    
        csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Uncut.csv"), R1_Keypoints_Uncut);
        csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Uncut.csv"), R4_Keypoints_Uncut);
        csvwrite(fullfile(outputFolder, "Keypoint_Feats_R1_Cut.csv"),   R1_Keypoints_Cut);
        csvwrite(fullfile(outputFolder, "Keypoint_Feats_R4_Cut.csv"),   R4_Keypoints_Cut);
    
        csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Uncut.csv"),     Probe1_PCs_R1_Uncut);
        csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Uncut.csv"),     Probe1_PCs_R4_Uncut);
        csvwrite(fullfile(outputFolder, "PCA_Probe1_R1_Cut.csv"),       Probe1_PCs_R1_Cut);
        csvwrite(fullfile(outputFolder, "PCA_Probe1_R4_Cut.csv"),       Probe1_PCs_R4_Cut);
    
        csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Uncut.csv"),     Probe2_PCs_R1_Uncut);
        csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Uncut.csv"),     Probe2_PCs_R4_Uncut);
        csvwrite(fullfile(outputFolder, "PCA_Probe2_R1_Cut.csv"),       Probe2_PCs_R1_Cut);
        csvwrite(fullfile(outputFolder, "PCA_Probe2_R4_Cut.csv"),       Probe2_PCs_R4_Cut);
    
        csvwrite(fullfile(outputFolder, "PCA_Probe3_R1_Uncut.csv"),     Probe3_PCs_R1_Uncut);
        csvwrite(fullfile(outputFolder, "PCA_Probe3_R4_Uncut.csv"),     Probe3_PCs_R4_Uncut);
        csvwrite(fullfile(outputFolder, "PCA_Probe3_R1_Cut.csv"),       Probe3_PCs_R1_Cut);
        csvwrite(fullfile(outputFolder, "PCA_Probe3_R4_Cut.csv"),       Probe3_PCs_R4_Cut);
    
        csvwrite(fullfile(outputFolder, "JawFeats_R1_Uncut.csv"),       jawfeats_R1_Uncut);
        csvwrite(fullfile(outputFolder, "JawFeats_R4_Uncut.csv"),       jawfeats_R4_Uncut);
        csvwrite(fullfile(outputFolder, "JawFeats_R1_Cut.csv"),         jawfeats_R1_Cut);
        csvwrite(fullfile(outputFolder, "JawFeats_R4_Cut.csv"),         jawfeats_R4_Cut);
    
        csvwrite(fullfile(outputFolder, "Tongue_R1.csv"),               R1_Tongue_Uncut);
        csvwrite(fullfile(outputFolder, "Tongue_R4.csv"),               R4_Tongue_Uncut);
    
        csvwrite(fullfile(outputFolder, "FCs_R1.csv"),                  FCs_Adj_R1);
        csvwrite(fullfile(outputFolder, "FCs_R4.csv"),                  FCs_Adj_R4);
        csvwrite(fullfile(outputFolder, "LRCs_R1.csv"),                 LRCs_Adj_R1);
        csvwrite(fullfile(outputFolder, "LRCs_R4.csv"),                 LRCs_Adj_R4);
    
        % Metadata
        fid = fopen(fullfile(outputFolder, 'metadata.txt'), 'w');
        fprintf(fid, 'Processing Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fprintf(fid, 'Script Name: %s\n',     mfilename('fullpath'));
        fprintf(fid, 'Session ID: %s\n',      anmName);
        fprintf(fid, 'Session Date: %s\n',    sessionDate);
        fclose(fid);
    
        fprintf('  Saved to %s\n', outputFolder);
    catch ME_err
        fprintf('  !! ERROR on session %s %s — skipping.\n', anmName, sessionDate);
        fprintf('     Message : %s\n', ME_err.message);
        fprintf('     Location: %s, line %d\n', ME_err.stack(1).file, ME_err.stack(1).line);
        continue
    end
end  % sessIdx
fprintf('\nAll sessions complete.\n');


%% ============================================================
%  HELPER FUNCTIONS
%% ============================================================
function meta = loadTD_session(anmName, sessionDate, datapth)
    meta.datapth = fullfile(datapth, 'DataObjects', anmName);
    meta.anm     = anmName;
    meta.date    = sessionDate;
    meta.probe   = [1 2 3];
    meta.datafn  = findDataFn(meta);
end

function objfn = findDataFn(meta)
    contents  = dir(meta.datapth);
    contents  = {contents.name}';
    strToFind = {'data_structure', meta.anm, meta.date};
    [fn, ~]   = patternMatchCellArray(contents, strToFind, 'all');
    objfn     = fn{1};
end

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
    data_pregc        = data(1:pre_gc_points, :, :);
    concatenated_data = reshape(data_pregc, [], size(data_pregc, 2));
    neuron_means      = mean(concatenated_data, 1);
    neuron_stds       = std(concatenated_data, 0, 1);
    zscored_data      = (data - neuron_means) ./ neuron_stds;
    S                 = SR * 2;
    zscored_data      = zscored_data(pre_gc_points-S+1:end, :, :);
end