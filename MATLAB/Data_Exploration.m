% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% May 2026
% Exploratory analyses of neural data


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


%% === LOOP THROUGH ANIMALS IN TABLE ===

meanFR_dir  = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Visualizations\meanFR_figures';
engMode_dir = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Visualizations\engMode_figures';

% Create folders if they don't exist
if ~exist(meanFR_dir,  'dir'), mkdir(meanFR_dir);  end
if ~exist(engMode_dir, 'dir'), mkdir(engMode_dir); end

A = readtable("SC_Animals.xlsx");
for rowix = 1:height(A)

    % Extract animal and date from table
    anm  = A.Animal{rowix};
    date = A.Date{rowix};
    % skip = A.Skip(rowix);

    % if skip==1
    %     disp("Skipping")
    %     continue;
    % end

    % Convert date format from '2026.03.10' to '2026-03-10'
    date = strrep(date, '.', '-');

    fprintf('Loading %s | %s ...\n', anm, date);

    try
        meta = [];
        meta = loadTD(meta, datapth, anm, date);
        params.probe = {meta.probe};
        
        % LOAD DATA
        [obj,params] = loadSessionData(meta,params,params.behav_only);
        % [obj,params] = loadSessionData(meta,params);
        
        for sessix = 1:numel(meta)
            me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
        end

        fprintf('  Success: %d probes\n', numel(params.probe{1}));

    catch ME
        fprintf('  SKIPPED: %s | %s — %s\n', anm, date, ME.message);
        continue
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
    % nSessions = numel(meta);
    % for sessix = 1:numel(meta)
    %     message = strcat('----Getting kinematic data for session',{' '},num2str(sessix), {' '},'out of',{' '},num2str(nSessions),'----');
    %     disp(message)
    %     kin(sessix) = getKinematics(obj(sessix), me(sessix), params(sessix));
    %     pos = getKeypointsFromVideo(obj(sessix), me(sessix), params(sessix));
    % end
    
    % clearvars -except kin meta obj params
    
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
    
    
    %% -- Extract All Trial Tongue Lengths -- %%
    % conds2use = [1];
    % condtrix = params(sessix).trialid{conds2use};                 % Get the trials from this condition
    % 
    % diff = nTrials_kinematic - nTrials_neural;
    % condtrix = condtrix(1:(end-diff), :);
    % 
    NTRIALS = nTrials_neural;
    
    % kinfeat = 'tongue_length';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1
    % sessix = 1;
    % kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
    % all_length = kin.dat(:,condtrix,kinix);
    
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
    
    %% -- Prep the neural data -- %%
    % Get trial IDs
    R1_Trials = params.trialid{8};
    R4_Trials = params.trialid{9};
    
    R1_Trials = R1_Trials(R1_Trials <= NTRIALS);
    R4_Trials = R4_Trials(R4_Trials <= NTRIALS);
    
    % R1_Trial_Track = R1_Trials;
    % R4_Trial_Track = R4_Trials;
    % 
    % % Filter Tongue Length
    % R1_Tongue = all_length((pre_gc_points-SR+1):end, R1_Trials);
    % R4_Tongue = all_length((pre_gc_points-SR+1):end, R4_Trials);
    % 
    % % Filter LP Contacts
    % R1_Contacts = all_contacts(R1_Trials);  % these contacts are relative to the gc
    % R4_Contacts = all_contacts(R4_Trials);  % these contacts are relative to the gc
    % 
    Ncells_P1 = numel(params.cluid{1, 1});
    Ncells_P2 = numel(params.cluid{1, 2});
    Ncells_P3 = numel(params.cluid{1, 3});

    clu_1 = 1:Ncells_P1;
    clu_2 = Ncells_P1+1:Ncells_P1+Ncells_P2;
    clu_3 = (Ncells_P1+Ncells_P2+1):Ncells_P1+Ncells_P2+Ncells_P3;
    % 
    % probe1_trialdat = obj.trialdat(:,clu_1, :);
    % probe2_trialdat = obj.trialdat(:,clu_2, :);
    % probe3_trialdat = obj.trialdat(:,clu_3, :);
    % 
    % % Limit to the desired trials
    % % This is done in multiple rouds since the final_RN_trials are indices into
    % % the RN_trials, not absolute trial numbers
    % probe1_R4 = probe1_trialdat(:,:,R4_Trials);  % -2 to 4s 100Hz
    % probe2_R4 = probe2_trialdat(:,:,R4_Trials);
    % probe3_R4 = probe3_trialdat(:,:,R4_Trials);
    % 
    % probe1_R1 = probe1_trialdat(:,:,R1_Trials);
    % probe2_R1 = probe2_trialdat(:,:,R1_Trials);
    % probe3_R1 = probe3_trialdat(:,:,R1_Trials);
    
    
    %% === SETUP ===
    all11 = 1:obj.bp.Ntrials;
    hit11 = all11(obj.bp.hit == 1);
    r111  = all11(obj.bp.rewardedLick == 1);
    r444  = all11(obj.bp.rewardedLick == 4);
    P8 = intersect(hit11, r111)';
    P9 = intersect(hit11, r444)';
    
    % Filter P9 to trials that actually have a 4th lick
    fourthLickTimes_r4 = nan(length(P9), 1);
    for i = 1:length(P9)
        tr = P9(i);
        licks = obj.bp.ev.lickL{tr};
        licks_postGo = sort(licks(licks > obj.bp.ev.goCue(tr)));
        if length(licks_postGo) >= 4
            fourthLickTimes_r4(i) = licks_postGo(4) - licks_postGo(1);
        end
    end
    validR4  = ~isnan(fourthLickTimes_r4);
    P9_valid = P9(validR4);
    
    col_r1  = [0 0 0];
    col_r4  = [1 0 0];
    lw_proj = 1.5;
    reg_all = {clu_1, clu_2, clu_3};
    
    %% === ENGAGEMENT MODE WEIGHTS ===
    pre_win_sec  = [-0.3, -0.05];
    post_win_sec = [ 0.1,  0.35];
    nNeurons_total = size(obj.trialdat, 2);
    m_pre_all  = zeros(length(P9_valid), nNeurons_total);
    m_post_all = zeros(length(P9_valid), nNeurons_total);
    
    for i = 1:length(P9_valid)
        tr    = P9_valid(i);
        t4    = fourthLickTimes_r4(validR4);
        t4    = t4(i);
        t_rel = obj.time - t4;
        pre_idx  = t_rel >= pre_win_sec(1)  & t_rel < pre_win_sec(2);
        post_idx = t_rel >= post_win_sec(1) & t_rel < post_win_sec(2);
        trial_data = squeeze(obj.trialdat(:, :, tr));
        if sum(pre_idx) > 0 && sum(post_idx) > 0
            m_pre_all(i, :)  = mean(trial_data(pre_idx,  :), 1);
            m_post_all(i, :) = mean(trial_data(post_idx, :), 1);
        end
    end
    
    m_pre  = mean(m_pre_all, 1);
    m_post = mean(m_post_all, 1);
    w_p1 = m_pre(clu_1) - m_post(clu_1);
    w_p2 = m_pre(clu_2) - m_post(clu_2);
    w_p3 = m_pre(clu_3) - m_post(clu_3);
    w_all = {w_p1, w_p2, w_p3};
    
    %% === MEAN FR PLOT ===
    figure('Name', 'Mean FR | All Regions x Shanks');
    
    for regionID = 1:3
        reg = reg_all{regionID};
        N   = length(reg);
        shanks_r = zeros(N, 1);
    
        for i = 1:N
            cluIdx      = params.cluid{1, regionID}(i);
            shanks_r(i) = obj.clu{1, regionID}(cluIdx).shank;
        end
    
        for shankID = 0:3
            idx  = find(shanks_r == shankID);
            spIx = (regionID - 1) * 4 + (shankID + 1);
            subplot(3, 4, spIx);
            hold on; box off;
    
            if isempty(idx)
                title(sprintf('R%d | Shank %d | empty', regionID, shankID), 'FontSize', 8);
                continue;
            end
    
            dat_r1_sh = obj.trialdat(:, reg(idx), P8);
            dat_r4_sh = obj.trialdat(:, reg(idx), P9_valid);
    
            mfr_r1 = movmean(squeeze(mean(dat_r1_sh, 2)), 10, 1);
            mfr_r4 = movmean(squeeze(mean(dat_r4_sh, 2)), 10, 1);
    
            if isvector(mfr_r1), mfr_r1 = mfr_r1(:); end
            if isvector(mfr_r4), mfr_r4 = mfr_r4(:); end
    
            mean_r1 = mean(mfr_r1, 2);
            ci_r1   = 1.96 * std(mfr_r1, 0, 2) / sqrt(size(mfr_r1, 2));
            mean_r4 = mean(mfr_r4, 2);
            ci_r4   = 1.96 * std(mfr_r4, 0, 2) / sqrt(size(mfr_r4, 2));
    
            fill([obj.time, fliplr(obj.time)], [mean_r1+ci_r1; flipud(mean_r1-ci_r1)]', ...
                 col_r1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            plot(obj.time, mean_r1, 'Color', col_r1, 'LineWidth', lw_proj);
    
            fill([obj.time, fliplr(obj.time)], [mean_r4+ci_r4; flipud(mean_r4-ci_r4)]', ...
                 col_r4, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            plot(obj.time, mean_r4, 'Color', col_r4, 'LineWidth', lw_proj);
    
            xline(0, 'k', 'LineWidth', 1);
            xlim([-0.5 2]);
            xlabel('time from go cue (s)', 'FontSize', 8);
            ylabel('mean FR (sp/s)', 'FontSize', 8);
            title(sprintf('R%d | Shank %d | n=%d', regionID, shankID+1, length(idx)), 'FontSize', 8);
            set(gca, 'FontSize', 8);
        end
    end
    legend({'R1 CI','R1 mean','R4 CI','R4 mean'}, 'Location', 'best');
    sgtitle('Mean FR | R1 (black) vs R4 (red)', 'FontSize', 12, 'FontWeight', 'bold');
    saveas(gcf, fullfile(meanFR_dir, sprintf('%s_%s_meanFR.png', anm, date)));
    
    %% === ENGAGEMENT MODE PROJECTION PLOT ===
    figure('Name', 'Eng Mode | All Regions x Shanks');
    
    for regionID = 1:3
        w   = w_all{regionID};
        reg = reg_all{regionID};
        N   = length(reg);
        shanks_r = zeros(N, 1);
    
        for i = 1:N
            cluIdx      = params.cluid{1, regionID}(i);
            shanks_r(i) = obj.clu{1, regionID}(cluIdx).shank;
        end
    
        for shankID = 0:3
            idx  = find(shanks_r == shankID);
            spIx = (regionID - 1) * 4 + (shankID + 1);
            subplot(3, 4, spIx);
            hold on; box off;
    
            if isempty(idx)
                title(sprintf('R%d | Shank %d | empty', regionID, shankID), 'FontSize', 8);
                continue;
            end
    
            w_shank = w(idx);
    
            dat_r1_sh  = obj.trialdat(:, reg(idx), P8);
            proj_r1_sh = movmean(squeeze(sum(dat_r1_sh .* reshape(w_shank, 1, [], 1), 2)), 10, 1);
    
            dat_r4_sh  = obj.trialdat(:, reg(idx), P9_valid);
            proj_r4_sh = movmean(squeeze(sum(dat_r4_sh .* reshape(w_shank, 1, [], 1), 2)), 10, 1);
    
            if isvector(proj_r1_sh), proj_r1_sh = proj_r1_sh(:); end
            if isvector(proj_r4_sh), proj_r4_sh = proj_r4_sh(:); end
    
            mean_r1 = mean(proj_r1_sh, 2);
            ci_r1   = 1.96 * std(proj_r1_sh, 0, 2) / sqrt(size(proj_r1_sh, 2));
            mean_r4 = mean(proj_r4_sh, 2);
            ci_r4   = 1.96 * std(proj_r4_sh, 0, 2) / sqrt(size(proj_r4_sh, 2));
    
            fill([obj.time, fliplr(obj.time)], [mean_r1+ci_r1; flipud(mean_r1-ci_r1)]', ...
                 col_r1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            plot(obj.time, mean_r1, 'Color', col_r1, 'LineWidth', lw_proj);
    
            fill([obj.time, fliplr(obj.time)], [mean_r4+ci_r4; flipud(mean_r4-ci_r4)]', ...
                 col_r4, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            plot(obj.time, mean_r4, 'Color', col_r4, 'LineWidth', lw_proj);
    
            xline(0, 'k', 'LineWidth', 1);
            xlim([-0.5 2]);
            xlabel('time from go cue (s)', 'FontSize', 8);
            ylabel('eng mode proj (a.u.)', 'FontSize', 8);
            title(sprintf('R%d | Shank %d | n=%d', regionID, shankID+1, length(idx)), 'FontSize', 8);
            set(gca, 'FontSize', 8);
        end
    end
    legend({'R1 CI','R1 mean','R4 CI','R4 mean'}, 'Location', 'best');
    sgtitle('Eng Mode Proj | R1 (black) vs R4 (red)', 'FontSize', 12, 'FontWeight', 'bold');
    saveas(gcf, fullfile(engMode_dir, sprintf('%s_%s_engMode.png', anm, date)));

    close all;

end  % of per animal loop