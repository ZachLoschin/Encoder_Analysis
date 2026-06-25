% Finding "Kinematic Modes"
clear,clc

sz = 14;
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

% addpath 'C:\Users\LabTech\Documents\Cortical Disengagement Code and Data\uninstructedMovements_v2-main\base code\functions_td'
% addpath 'C:\Users\LabTech\Documents\Cortical Disengagement Code and Data\uninstructedMovements_v2-main\ObjVis\warp'
% addpath 'C:\Users\LabTech\Documents\Cortical Disengagement Code and Data\uninstructedMovements_v2-main\base code\other_codes\functions_td'

%% PARAMETERS
params.alignEvent          = 'firstLick'; % 'fourthLick' 'goCue'  'moveOnset'  'firstLick' 'thirdLick' 'lastLick' 'reward'

% time warping only operates on neural data for now.
params.behav_only          = 0;
params.timeWarp            = 0;  % piecewise linear time warping - each lick duration on each trial gets warped to median lick duration for that lick across trials
params.nLicks              = 20; % number of post go cue licks to calculate median lick duration for and warp individual trials to

params.lowFR               = 0.1; % remove clusters with firing rates across all trials less than this val

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
% 
% params.condition(1) = {'hit==1 | hit==0' };    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 6'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 6'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 6'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & rewardedLick == 6'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1' };    % left to right         % right hits, no stim, aw off

params.tmin = -2.75;
params.tmax = 7;
params.dt = 1/200;

% smooth with causal gaussian kernel
params.smooth = 10;

% cluster qualities to use
% params.quality = {'good','excellent','fair','ok',' '}; % accepts any cell array of strings - special character 'all' returns clusters of any quality
% params.quality = {'good','excellent','fair'}; % accepts any cell array of strings - special character 'all' returns clusters of any quality
params.quality = {'good'}; % accepts any cell array of strings - special character 'all' returns clusters of any quality


params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_nostril','bottom_nostril'}};
params.feat_varToExplain = 80;  % num factors for dim reduction of video features should explain this much variance
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

animal = 'TDsa7';
date = '2026-03-15';
meta = loadTD(meta,datapth, animal, date);

params.probe = {meta.probe}; 

%% LOAD DATA

[obj,params] = loadSessionData(meta,params,params.behav_only );
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


%%

% Path to channel map
cm = load('chanMap_NPtype24_first96_allShanks.mat');

% Shank number for each channel
kcoords = cm.kcoords;          % 384×1 vector
% Sanity check plot of channel map organization
xcoords = cm.xcoords;   % x-location (in microns) of each channel
ycoords = cm.ycoords;   % y-location/depth (in microns) of each channel
    
chanMap = cm.chanMap;

nChans = length(xcoords);
for cc = 1:nChans
    scatter(xcoords(cc),ycoords(cc)); hold on
    text(xcoords(cc)+3,ycoords(cc),num2str(chanMap(cc)))

end
title('Channel Organization - chanMap-NPtype24-first96-allShanks')
xlabel('microns (x direction)')
ylabel('microns (y direction)')

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









%% Calculate Last Lick

hitTrials = params.trialid{1,1};

LastL = [];
for i = 1:length(hitTrials)

    tr = hitTrials(i);
    lickL = obj.bp.ev.lickL{i};
    if isempty(lickL)
        LastL = [LastL 0];
    elseif ~isempty(lickL)
        lickL = lickL(lickL > obj.bp.ev.goCue(i));
        if ~isempty(lickL)
            LastL = [LastL lickL(end) - obj.bp.ev.goCue(i)];
        else
            LastL = [LastL 0];
        end
    end

end


%%

% obj.bp.hit
% obj.bp.rewardedLick
% 
% params.trialid{1, 1}  = [1:length(obj.bp.hit)];
% a = [1:length(obj.bp.hit)];
% a = a(1:end-1);
% params.trialid{1, 1}  = a;

%% TONGUE
conds2use = [1];                      % With reference to 'params.condition'
kinfeat = 'tongue_length';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1
% tongue_length
% top_tongue_xdisp_view2
% jaw_ydisp_view1
sessix = 1;

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials
end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));

% Ncells = size(obj.psth, 2);
% 
% % clu_m1TJ = 1:size(params.cluid{1, 1},1);
% % clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;
% 
% clu_m1TJ = 1:size(params.cluid);
% clu_ALM = 1:size(params.cluid);
% 
% 
% % first lick mode
% reg = clu_ALM;
% reg1 = clu_m1TJ;

%%



Ncells_P1 = numel(params.cluid{1, 1});
Ncells_P2 = numel(params.cluid{1, 2});
Ncells_P3 = numel(params.cluid{1, 3});

clu_1 = 1:Ncells_P1;
clu_2 = Ncells_P1+1:Ncells_P1+Ncells_P2;
clu_3 = (Ncells_P1+Ncells_P2+1):Ncells_P1+Ncells_P2+Ncells_P3;


% first lick mode
reg1 = clu_1;
reg2 = clu_2;
reg3 = clu_3;


Kinematics1 = kin.dat(:,condtrix,kinix);



% Ncells_P1 = numel(params.cluid{1, 1});
% Ncells_P2 = numel(params.cluid{1, 2});
% 
% clu_1 = 1:Ncells_P1;
% clu_2 = Ncells_P1+1:Ncells_P1+Ncells_P2;
% 
% 
% % first lick mode
% reg1 = clu_1;
% reg2 = clu_2;

%%

% trial = 20;
% 
% jaw_disp = kin.dat(:,trial,2);
% lick_traj = Kinematics1(:,trial);
% go_cue = obj.bp.ev.goCue(trial);
% lick_contacts = obj.bp.ev.lickL{trial} - go_cue;
% % lick_contacts = lick_contacts - 0.5;
% 
% 
% figure; 
% plot(obj.time, lick_traj)
% hold on
% xline(lick_contacts)


%%
conds2use = [8];                      % With reference to 'params.condition'
kinfeat = 'tongue_length';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1
sessix = 1;

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials

condtrix(end) = [];
end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
Length = kin.dat(:,condtrix,kinix);

kinfeat = 'tongue_angle';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials
condtrix(end) = [];
end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
angle = kin.dat(:,condtrix,kinix);

kinfeat = 'top_tongue_xvel_view2';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials
condtrix(end) = [];
end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
velocity = kin.dat(:,condtrix,kinix);

%%
conds2use = [9];                      % With reference to 'params.condition'
kinfeat = 'tongue_length';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1
sessix = 1;

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials

condtrix(end) = [];
end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
Length4 = kin.dat(:,condtrix,kinix);

kinfeat = 'tongue_angle';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials
condtrix(end) = [];
end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
angle4 = kin.dat(:,condtrix,kinix);

kinfeat = 'top_tongue_xvel_view2';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials
condtrix(end) = [];
end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
velocity4 = kin.dat(:,condtrix,kinix);

%% Angle Kinematic Plot 

% figure; 
% subplot(1,2,1)
% trials = params.trialid{1, 8};
% 
% cmp = jet;
% L = angle;
% 
% imagesc(obj.time, [1:size(L,2)], L');
% xline(0,'LineWidth', 1.5)
% colormap(cmp)
% %     caxis([rang]);
% colorbar; grid off;
% % axis tight; axis square;
% box off; xlabel('time(s)'); ylabel('Trial #'); title([kinfeat,' plot'], 'Interpreter', 'none');
% set(gca,'FontSize',15)
% xlim([-0.5 2])
% 
% subplot(1,2,2)
% trials = params.trialid{1, 9};
% 
% L = angle4;
% 
% imagesc(obj.time, [1:size(L,2)], L');
% xline(0,'LineWidth', 1.5)
% colormap(cmp)
% %     caxis([rang]);
% colorbar; grid off;
% % axis tight; axis square;
% box off; xlabel('time(s)'); ylabel('Trial #'); title([kinfeat,' plot'], 'Interpreter', 'none');
% set(gca,'FontSize',15)
% xlim([-0.5 2])
% 
% 
% 
% px = 500;
% py = 500;
% width = 800;
% height = 250;
% set(gcf, 'Position', [px, py, width, height]); % Set figure position and size


%%
% 
% figure;
% 
% subplot(1,3,1)
% 
% histogram(params.cluid{1, 1},100);
% 
% subplot(1,3,2)
% 
% histogram(params.cluid{1, 2},100);
% 
% subplot(1,3,3)
% 
% histogram(params.cluid{1, 3},100);
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT ALL NEURONS IN ONE FIGURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sz = 7;
% lw = 1.75;
% 
% neurons = reg3;
% N = length(neurons);
% 
% trialTypes = [8 9];
% col = {[0 0 0],[1 0 0]};   % 8 = black, 9 = red
% 
% time = obj.time;
% 
% % choose grid automatically
% nCols = ceil(sqrt(N));
% nRows = ceil(N / nCols);
% 
% figure;
% tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');
% 
% for i = 1:N
%     ax = nexttile; %#ok<LAXES>
%     hold on
% 
%     % plot trial types
%     plot(time, movmean(obj.psth(:,neurons(i),8),5), ...
%         'Color', col{1}, 'LineWidth', lw);
% 
%     plot(time, movmean(obj.psth(:,neurons(i),9),5), ...
%         'Color', col{2}, 'LineWidth', lw);
% 
%     % formatting
%     xline(-2,'k');
%     xline(0,'k');
%     xlim([-0.5 1.5]);
% 
%     title(['Cell ', num2str(neurons(i))], 'FontSize', sz);
%     set(gca,'FontSize', sz);
%     box off
% 
%     % only label edges (reduces clutter)
%     if i > (nRows-1)*nCols
%         xlabel(['time from ', num2str(params.alignEvent)]);
%     end
%     if mod(i-1,nCols) == 0
%         ylabel('firing rate');
%     end
% end
% 
% % global legend
% lg = legend({'Trial 8','Trial 9'});
% lg.Layout.Tile = 'north';
%%
sz = 2;
lw = 1.25;

% ===== MODULAR SETTINGS: CHANGE THESE FOR REG1 vs REG2 =====
neurons = reg1;           % Change to reg1 or reg2
regionID = 1;             % 1 = reg1, 2 = reg2
% ============================================================

N = length(neurons);

trialTypes = [8 9];
col = {[0 0 0],[1 0 0]};   % 8 = black, 9 = red

time = obj.time;

% --- Extract shank and channel for each neuron ---
shanks = zeros(N,1);
chans = zeros(N,1);
for i = 1:N
    % Use LOCAL index i (not neurons(i)) to access params.cluid
    cluIdx = params.cluid{1, regionID}(i);  % i goes from 1 to N
    shanks(i) = obj.clu{1, regionID}(cluIdx).shank;
    chans(i) = obj.clu{1, regionID}(cluIdx).channel;
end

% --- Plot one figure per shank (shanks 0, 1, 2, 3) ---
for shankID = 0:3
    
    % Get neurons on this shank
    idx = find(shanks == shankID);
    
    if isempty(idx)
        warning('No neurons found on shank %d', shankID);
        continue;
    end
    
    neurons_thisShank = neurons(idx);
    chans_thisShank = chans(idx);
    
    % Sort by channel number (DESCENDING: highest channel at top)
    [chans_sorted, sortIdx] = sort(chans_thisShank, 'descend');
    neurons_sorted = neurons_thisShank(sortIdx);
    
    nCells = length(neurons_sorted);
    
    % Create grid layout
    nCols = ceil(sqrt(nCells));
    nRows = ceil(nCells / nCols);
    
    figure('Name', sprintf('Shank %d (Region %d)', shankID, regionID));
    tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');
    
    for i = 1:nCells
        neuronIdx = neurons_sorted(i);
        chanNum = chans_sorted(i);
        
        ax = nexttile; %#ok<LAXES>
        hold on
        
        % plot trial types
        plot(time, movmean(obj.psth(:,neuronIdx,8),8), ...
            'Color', col{1}, 'LineWidth', lw);
        
        plot(time, movmean(obj.psth(:,neuronIdx,9),8), ...
            'Color', col{2}, 'LineWidth', lw);
        
        % formatting
        xline(-2,'k');
        xline(0,'k');
        xlim([-0.5 2]);
        
        % title shows neuron index and channel
        title(sprintf('Cell %d | Ch%d', neuronIdx, chanNum), ...
            'FontSize', sz);
        set(gca,'FontSize', sz);
        box off
        
        % only label edges (reduces clutter)
        if i > (nRows-1)*nCols
            xlabel(['time from ', num2str(params.alignEvent)]);
        end
        if mod(i-1,nCols) == 0
            ylabel('firing rate');
        end
    end
    
    % global legend
    lg = legend({'Trial 8','Trial 9'});
    lg.Layout.Tile = 'north';
    
    sgtitle(sprintf('Shank %d (Region %d)', shankID, regionID), 'FontSize', 12, 'FontWeight', 'bold');
end

%% === MEAN SPIKE RATE PER SHANK (R1 vs R4) ===
reg_all = {clu_1, clu_2, clu_3};
% reg_all = {clu_1, clu_2};
all11 = 1:obj.bp.Ntrials;
hit11 = all11(obj.bp.hit == 1);
r111  = all11(obj.bp.rewardedLick == 1);
r444  = all11(obj.bp.rewardedLick == 4);
P8 = intersect(hit11, r111)';
P9 = intersect(hit11, r444)';

fourthLickTimes_r4 = nan(length(P9), 1);
for i = 1:length(P9)
    tr = P9(i);
    licks = obj.bp.ev.lickL{tr};
    licks_postGo = sort(licks(licks > obj.bp.ev.goCue(tr)));
    if length(licks_postGo) >= 4
        firstLick_abs  = licks_postGo(1);
        fourthLick_abs = licks_postGo(4);
        fourthLickTimes_r4(i) = fourthLick_abs - firstLick_abs;
    end
end
validR4  = ~isnan(fourthLickTimes_r4);
P9_valid = P9(validR4);

col_r1  = [0 0 0];
col_r4  = [1 0 0];
lw_proj = 1.5;

figure('Name', 'Mean FR | All Regions x Shanks');

for regionID = 1:3

    reg  = reg_all{regionID};
    N    = length(reg);

    shanks_r = zeros(N, 1);
    for i = 1:N
        cluIdx      = params.cluid{1, regionID}(i);
        shanks_r(i) = obj.clu{1, regionID}(cluIdx).shank;
    end

    for shankID = 0:3

        idx = find(shanks_r == shankID);

        spIx = (regionID - 1) * 4 + (shankID + 1);   % subplot index: row=region, col=shank
        subplot(3, 4, spIx);
        hold on

        if isempty(idx)
            title(sprintf('R%d | Shank %d | empty', regionID, shankID), 'FontSize', 8);
            box off; continue;
        end

        dat_r1_sh = obj.trialdat(:, reg(idx), P8);
        dat_r4_sh = obj.trialdat(:, reg(idx), P9_valid);

        mfr_r1 = movmean(squeeze(mean(dat_r1_sh, 2)), 10, 1);   % [nTime x nR1]
        mfr_r4 = movmean(squeeze(mean(dat_r4_sh, 2)), 10, 1);   % [nTime x nR4]

        mean_r1_sh = mean(mfr_r1, 2);
        ci_r1_sh   = 1.96 * std(mfr_r1, 0, 2) / sqrt(size(mfr_r1, 2));

        mean_r4_sh = mean(mfr_r4, 2);
        ci_r4_sh   = 1.96 * std(mfr_r4, 0, 2) / sqrt(size(mfr_r4, 2));

        fill([obj.time, fliplr(obj.time)], ...
             [mean_r1_sh + ci_r1_sh; flipud(mean_r1_sh - ci_r1_sh)]', ...
             col_r1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        plot(obj.time, mean_r1_sh, 'Color', col_r1, 'LineWidth', lw_proj);

        fill([obj.time, fliplr(obj.time)], ...
             [mean_r4_sh + ci_r4_sh; flipud(mean_r4_sh - ci_r4_sh)]', ...
             col_r4, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        plot(obj.time, mean_r4_sh, 'Color', col_r4, 'LineWidth', lw_proj);

        xline(0, 'k', 'LineWidth', 1);
        xlim([-1.5 2]);
        xlabel('time from first lick (s)', 'FontSize', 8);
        ylabel('mean FR (sp/s)', 'FontSize', 8);
        title(sprintf('R%d | Shank %d | n=%d', regionID, shankID, length(idx)), 'FontSize', 8);
        box off;
        set(gca, 'FontSize', 8);
    end
end

legend({'R1 CI','R1 mean','R4 CI','R4 mean'}, 'Location', 'best');
sgtitle('Mean FR | R1 (black) vs R4 (red)', 'FontSize', 12, 'FontWeight', 'bold');



%% === ENGAGEMENT MODE PROJECTION PER SHANK ===
pre_win_sec  = [-0.3, -0.05];
post_win_sec = [ 0.1,  0.35];
nNeurons_total = size(obj.trialdat, 2);
m_pre_all  = zeros(length(P9_valid), nNeurons_total);
m_post_all = zeros(length(P9_valid), nNeurons_total);

for i = 1:length(P9_valid)
    tr    = P9_valid(i);
    t4    = fourthLickTimes_r4(i);
    t_rel = obj.time - t4;
    pre_idx  = t_rel >= pre_win_sec(1)  & t_rel < pre_win_sec(2);
    post_idx = t_rel >= post_win_sec(1) & t_rel < post_win_sec(2);
    trial_data = squeeze(obj.trialdat(:, :, tr));
    if sum(pre_idx) > 0 && sum(post_idx) > 0
        m_pre_all(i, :)  = mean(trial_data(pre_idx,  :), 1);
        m_post_all(i, :) = mean(trial_data(post_idx, :), 1);
    end
end

m_pre  = mean(m_pre_all,  1);
m_post = mean(m_post_all, 1);

w_p1 = m_pre(clu_1) - m_post(clu_1);
w_p2 = m_pre(clu_2) - m_post(clu_2);
w_p3 = m_pre(clu_3) - m_post(clu_3);
w_all = {w_p1, w_p2, w_p3};

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

        idx = find(shanks_r == shankID);

        spIx = (regionID - 1) * 4 + (shankID + 1);
        subplot(3, 4, spIx);
        hold on

        if isempty(idx)
            title(sprintf('R%d | Shank %d | empty', regionID, shankID), 'FontSize', 8);
            box off; continue;
        end

        w_shank = w(idx);

        dat_r1_sh  = obj.trialdat(:, reg(idx), P8);
        proj_r1_sh = movmean(squeeze(sum(dat_r1_sh .* reshape(w_shank, 1, [], 1), 2)), 10, 1);

        dat_r4_sh  = obj.trialdat(:, reg(idx), P9_valid);
        proj_r4_sh = movmean(squeeze(sum(dat_r4_sh .* reshape(w_shank, 1, [], 1), 2)), 10, 1);

        mean_r1_sh = mean(proj_r1_sh, 2);
        ci_r1_sh   = 1.96 * std(proj_r1_sh, 0, 2) / sqrt(size(proj_r1_sh, 2));

        mean_r4_sh = mean(proj_r4_sh, 2);
        ci_r4_sh   = 1.96 * std(proj_r4_sh, 0, 2) / sqrt(size(proj_r4_sh, 2));

        fill([obj.time, fliplr(obj.time)], ...
             [mean_r1_sh + ci_r1_sh; flipud(mean_r1_sh - ci_r1_sh)]', ...
             col_r1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        plot(obj.time, mean_r1_sh, 'Color', col_r1, 'LineWidth', lw_proj);

        fill([obj.time, fliplr(obj.time)], ...
             [mean_r4_sh + ci_r4_sh; flipud(mean_r4_sh - ci_r4_sh)]', ...
             col_r4, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        plot(obj.time, mean_r4_sh, 'Color', col_r4, 'LineWidth', lw_proj);

        xline(0, 'k', 'LineWidth', 1);
        xlim([-1.5 2]);
        xlabel('time from first lick (s)', 'FontSize', 8);
        ylabel('eng mode proj', 'FontSize', 8);
        title(sprintf('R%d | Shank %d | n=%d', regionID, shankID, length(idx)), 'FontSize', 8);
        box off;
        set(gca, 'FontSize', 8);
    end
end

legend({'R1 CI','R1 mean','R4 CI','R4 mean'}, 'Location', 'best');
sgtitle('Eng Mode | R1 (black) vs R4 (red)', 'FontSize', 12, 'FontWeight', 'bold');


%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT ALL NEURONS IN ONE FIGURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sz = 2;
% lw = 1.75;
% 
% neurons = reg1;
% N = length(neurons);
% 
% trialTypes = [8 9];
% col = {[0 0 0],[1 0 0]};   % 8 = black, 9 = red
% 
% time = obj.time;
% 
% % choose grid automatically
% nCols = ceil(sqrt(N));
% nRows = ceil(N / nCols);
% 
% figure;
% tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');
% 
% for i = 1:N
%     ax = nexttile; %#ok<LAXES>
%     hold on
% 
%     % plot trial types
%     plot(time, movmean(obj.psth(:,neurons(i),8),5), ...
%         'Color', col{1}, 'LineWidth', lw);
% 
%     plot(time, movmean(obj.psth(:,neurons(i),9),5), ...
%         'Color', col{2}, 'LineWidth', lw);
% 
%     % formatting
%     xline(-2,'k');
%     xline(0,'k');
%     xlim([-0.5 1.5]);
% 
%     title(['Cell ', num2str(neurons(i))], 'FontSize', sz);
%     set(gca,'FontSize', sz);
%     box off
% 
%     % only label edges (reduces clutter)
%     if i > (nRows-1)*nCols
%         xlabel(['time from ', num2str(params.alignEvent)]);
%     end
%     if mod(i-1,nCols) == 0
%         ylabel('firing rate');
%     end
% end
% 
% % global legend
% lg = legend({'Trial 8','Trial 9'});
% lg.Layout.Tile = 'north';



%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POPULATION MODE PROJECTION
% Mode = activity(200:300) - activity(500:550)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
% neurons = reg3;
% neurons = [reg1 reg2 reg3];
% 
% % 1) pick your window
% tsel    = obj.time >= -0.75 & obj.time <= 1.5;
% timeVec = obj.time(tsel);
% 
% % 2) extract PSTHs and form difference matrix D (neurons × time)
% P1 = squeeze( obj.psth(tsel, neurons, 8) )';   % [N × T]
% P2 = squeeze( obj.psth(tsel, neurons, 9) )';   % [N × T]
% D  = P2 - P1;                                  % [N × T]
% 
% % 3) normalize each row
% Dnorm = zscore(D, 0, 2);  % subtract mean & divide by std over time, per neuron
% 
% % 4) PCA (optional, you keep it here)

% [ coeff, score, ~, ~, explained ] = pca(Dnorm);
% 
% % --- UMAP ---
% addpath('C:\Users\LabUser\Desktop\Cortical Disengagement\uninstructedMovements_v2-main\UMAP\umap');
% [Yumap, umapStruct] = run_umap( ...
%     Dnorm, ...
%     'n_components',   3, ...
%     'metric',         'correlation',...
%     'n_neighbors',    15, ...
%     'min_dist',       0.01, ...
%     'verbose',        'none' );
% 
% % k-means in the 3-D UMAP space
% K   = 5;
% idx = kmeans(Yumap, K, 'Replicates', 20);
% 
% % --- 3-D scatter of the embedding ---
% figure;
% scatter3( ...
%   Yumap(:,1), Yumap(:,2), Yumap(:,3), ...
%   36, idx, 'filled' );
% xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
% title('3-D UMAP embedding (colored by k-means idx)');
% colormap(jet); colorbar;
% grid on; rotate3d on;
% 
% % ------------------------------------------------------
% %  A) SORT CLUSTERS BY SIZE
% % ------------------------------------------------------
% % compute cluster sizes
% clusterSizes = arrayfun(@(c) sum(idx==c), 1:K);
% % sort descending
% [~, sortedClusters] = sort(clusterSizes, 'descend');
% 
% % ------------------------------------------------------
% %  B) REPORT WHERE reg1, reg2, reg3 NEURONS FELL
% % ------------------------------------------------------
% % after you compute Yumap (N×3) and have neurons = [reg1 reg2 reg3]
% 
% % define logical indices
% isReg1 = ismember(neurons, reg1);
% isReg2 = ismember(neurons, reg2);
% isReg3 = ismember(neurons, reg3);
% 
% % 3-D scatter colored by region instead of cluster
% figure; hold on;
% % choose three colors:
% col = lines(3);  % or manually: col = [1 0 0; 0 1 0; 0 0 1];
% 
% scatter3( Yumap(isReg1,1), Yumap(isReg1,2), Yumap(isReg1,3), ...
%     36, col(1,:), 'filled' );
% scatter3( Yumap(isReg2,1), Yumap(isReg2,2), Yumap(isReg2,3), ...
%     36, col(2,:), 'filled' );
% scatter3( Yumap(isReg3,1), Yumap(isReg3,2), Yumap(isReg3,3), ...
%     36, col(3,:), 'filled' );
% 
% xlabel('UMAP 1');
% ylabel('UMAP 2');
% zlabel('UMAP 3');
% title('3-D UMAP embedding (colored by reg1/2/3)');
% legend('reg1','reg2','reg3','Location','best');
% grid on;
% rotate3d on;
% 
% 
% % ------------------------------------------------------
% %  C) PLOT AVERAGE PSTH PER CLUSTER, IN DESCENDING SIZE
% % ------------------------------------------------------
% figure;
% tiledlayout(3,4,'Padding','compact','TileSpacing','compact');
% for i = 1:K
%     c = sortedClusters(i);          % cluster ID in rank-order
%     members = find(idx==c);         % row-indices for this cluster
%     m1 = mean(P1(members,:), 1);
%     m2 = mean(P2(members,:), 1);
% 
%     nexttile;
%     plot(timeVec, m1, 'r', 'LineWidth', 1.5); hold on;
%     plot(timeVec, m2, 'b', 'LineWidth', 1.5);
%     xlabel('Time (s)');
%     ylabel('Firing rate (Hz)');
%     title(sprintf('Cluster %d (n=%d)', c, numel(members)));
%     xlim([timeVec(1), timeVec(end)]);
%     if i==1
%       legend('Cond 8','Cond 9','Location','best');
%     end
%     box off;
% end
% % suptitle('Average PSTHs by Cluster (sorted by cluster size)');
% 
% 
% % package your regions into a cell
% regs     = {reg1, reg2, reg3};
% regNames = {'reg1','reg2','reg3'};
% 
% % preallocate: rows = regions, cols = clusters
% perc = zeros(numel(regs), K);
% 
% % compute percentages
% for r = 1:numel(regs)
%     thisMask = ismember(neurons, regs{r});   % logical mask into idx
%     nR       = sum(thisMask);                % total neurons in region r
%     for c = 1:K
%         perc(r,c) = sum(idx(thisMask)==c) / nR * 100;
%     end
% end
% 
% % plot grouped bar
% figure;
% bar( perc.' , 'grouped' );    % transpose so x-axis = clusters 1..K
% xticks(1:K);
% xlabel('Cluster ID');
% ylabel('Percentage of neurons (%)');
% legend(regNames, 'Location','best');
% title('For each region, % of its neurons in each cluster');

%% PCA
% % PCA - Position
% figure;
% for i = 1:3
% [~, coeffs] = pca(obj.psth(:,[reg1],i+1), 'NumComponents', 8);
% subplot(2,3,i)
% plot(obj.time,coeffs(:, 1:3),'LineWidth',2); xline(0);
% grid off; axis square; box off; xlim([-1.85 2]);
% xlabel(['time from ',params.alignEvent])
% ylabel('Activity (a.u) ALM')
% set(gca,'FontSize',sz)
% if i ==1
%     title('Position 1')
% elseif i ==2
%     title('Position 2')
% elseif i ==3
%     title('Position 3')
% end
% end
% 
% for i = 1:3
% [~, coeffs] = pca(obj.psth(:,[reg2],i+1), 'NumComponents', 8);
% subplot(2,3,3+i)
% plot(obj.time,coeffs(:, 1:3),'LineWidth',2); xline(0);
% grid off; axis square; box off; xlim([-1.85 2]);
% xlabel(['time from ',params.alignEvent])
% ylabel('Activity (a.u) M1TJ')
% set(gca,'FontSize',sz)
% if i ==1
%     title('Position 1')
% elseif i ==2
%     title('Position 2')
% elseif i ==3
%     title('Position 3')
% end
% end

%%%%%%%%%%%%%%%%%%%%%%%%
nmean = 15;

figure;
% PCA - rewardedLick
subplot(3,3,1)
[~, coeffs] = pca(obj.psth(:,[reg1],8), 'NumComponents', 8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('Rewarded after Lick #1')
set(gca,'FontSize',sz)

subplot(3,3,2)
[~, coeffs] = pca(obj.psth(:,[reg1],9), 'NumComponents', 8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('Rewarded after Lick #4')
set(gca,'FontSize',sz)

subplot(3,3,3)
[~, coeffs] = pca(obj.psth(:,[reg1],9)-obj.psth(:,[reg1],8), 'NumComponents',8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('R4 - R1 P1')
set(gca,'FontSize',sz)


subplot(3,3,4)

[~, coeffs] = pca(obj.psth(:,[reg2],8), 'NumComponents', 8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('Rewarded after Lick #1')
set(gca,'FontSize',sz)

subplot(3,3,5)
[~, coeffs] = pca(obj.psth(:,[reg2],9), 'NumComponents', 8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('Rewarded after Lick #4')
set(gca,'FontSize',sz)

subplot(3,3,6)
[~, coeffs] = pca(obj.psth(:,[reg2],9)-obj.psth(:,[reg2],8), 'NumComponents',8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('R4 - R1 P1')
set(gca,'FontSize',sz)

subplot(3,3,7)
[~, coeffs] = pca(obj.psth(:,[reg3],8), 'NumComponents', 8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('Rewarded after Lick #1')
set(gca,'FontSize',sz)

subplot(3,3,8)
[~, coeffs] = pca(obj.psth(:,[reg3],9), 'NumComponents', 8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('Rewarded after Lick #4')
set(gca,'FontSize',sz)

subplot(3,3,9)
[~, coeffs] = pca(obj.psth(:,[reg3],9)-obj.psth(:,[reg3],8), 'NumComponents',8);
plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
grid off; axis square; box off; xlim([-1.85 2]);
xlabel(['time from ',params.alignEvent])
ylabel('Activity (a.u) ALM')
title('R4 - R1 P1')
set(gca,'FontSize',sz)


%%
% sz = 10;
% neurons = reg1;
% % trang = [450:600];
% % trang = [600:750];
% trang = [1:1900];
% 
% figure;
% 
% subplot(1,2,1)
% [~, ~, ~, ~, explained] = pca(obj.psth(trang,[reg1],8), 'NumComponents',8);% Compute PCA
% numPCs = length(explained); % Get actual number of PCs
% 
% plot(1:numPCs, explained, '-o', 'LineWidth', 2, 'MarkerSize',5);
% xlabel('Principal Component');
% ylabel('Variance Explained (%)');
% title('Variance Explained by Each PC');
% set(gca, 'FontSize', sz);
% ylim([0 80])
% xlim([0 10])
% 
% hold on 
% 
% [~, ~, ~, ~, explained] = pca(obj.psth(trang,[reg],8), 'NumComponents',8); % Compute PCA
% numPCs = length(explained); % Get actual number of PCs
% 
% plot(1:numPCs, explained, '-o', 'LineWidth', 2, 'MarkerSize',5);
% xlabel('Principal Component');
% ylabel('Variance Explained (%)');
% title('Variance Explained by Each PC');
% set(gca, 'FontSize', sz);
% ylim([0 80])
% xlim([0 10])
% 
% subplot(1,2,2)
% 
% [~, ~, ~, ~, explained] = pca(obj.psth(trang,[reg1],9), 'NumComponents',8); % Compute PCA
% numPCs = length(explained); % Get actual number of PCs
% 
% plot(1:numPCs, cumsum(explained), '-o', 'LineWidth', 2, 'MarkerSize',5);
% xlabel('Principal Component');
% ylabel('Cumulative Variance Explained (%)');
% title('Cumulative Variance Explained');
% set(gca, 'FontSize', sz);
% ylim([0 100])
% xlim([0 10])
% 
% hold on 
% 
% [~, ~, ~, ~, explained] = pca(obj.psth(trang,[reg],8), 'NumComponents',8); % Compute PCA
% numPCs = length(explained); % Get actual number of PCs
% 
% 
% plot(1:numPCs, cumsum(explained), '-o', 'LineWidth', 2, 'MarkerSize',5);
% xlabel('Principal Component');
% ylabel('Cumulative Variance Explained (%)');
% title('Cumulative Variance Explained');
% set(gca, 'FontSize', sz);
% ylim([0 100])
% xlim([0 10])

% 
% subplot(2,3,4)
% [~, coeffs] = pca(obj.psth(:,[reg],8), 'NumComponents', 8);
% plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
% grid off; axis square; box off; xlim([-1.85 2]);
% xlabel(['time from ',params.alignEvent])
% ylabel('Activity (a.u) M1TJ')
% title('Rewarded after Lick #1')
% set(gca,'FontSize',sz)
% 
% subplot(2,3,5)
% [~, coeffs1] = pca(obj.psth(:,[reg],9), 'NumComponents', 8);
% plot(obj.time,movmean(coeffs1(:, 1:3),nmean),'LineWidth',3); xline(0);
% grid off; axis square; box off; xlim([-1.85 2]);
% xlabel(['time from ',params.alignEvent])
% ylabel('Activity (a.u) M1TJ')
% title('Rewarded after Lick #4')
% set(gca,'FontSize',sz)
% 
% 
% subplot(2,3,6)
% [~, coeffs] = pca(obj.psth(:,[reg],9)-obj.psth(:,[reg],8), 'NumComponents', 8);
% plot(obj.time,movmean(coeffs(:, 1:3),nmean),'LineWidth',3); xline(0);
% grid off; axis square; box off; xlim([-1.85 2]);
% xlabel(['time from ',params.alignEvent])
% ylabel('Activity (a.u) M1TJ')
% title('R4 - R1 P2')
% set(gca,'FontSize',sz)

%% Mean Spike Rate

% figure;
% 
% windowSize = 10;
% 
% for k = 1:2
% 
%     if k == 1
% neurons = reg1;
%     else
%         neurons = reg2;
%     end
% 
% 
% trials = params.trialid{8};
% 
% aa_norm = [];
% aa = obj.trialdat(:,neurons,trials);
% min_val = min(aa, [], 'all'); max_val = max(aa, [], 'all');
% aa_norm = normalize((aa - min_val) / (max_val - min_val), 'range', [0 1]);
% % aa_norm = aa;
% aa_r1 = aa_norm;
% 
% trials = params.trialid{9};
% 
% aa_norm = [];
% aa = obj.trialdat(:,neurons,trials);
% min_val = min(aa, [], 'all'); max_val = max(aa, [], 'all');
% aa_norm = normalize((aa - min_val) / (max_val - min_val), 'range', [0 1]);
% % aa_norm = aa;
% aa_r4 = aa_norm;
% 
% 
% 
% 
% subplot(2,3,k*3)
% % trials = allTrials{i};
% D1 = mean(aa_r1,3);
% D2 = mean(aa_r4,3);
% hold on 
% plot(obj.time,movmean(mean(D1,2),windowSize), 'LineWidth',2,'Color','k')
% hold on 
% plot(obj.time,movmean(mean(D2,2),windowSize), 'LineWidth',2,'Color','r');
% axis tight;
% % legend('pre','post')
% xline(0)
% title(['Date : ', num2str(obj.pth.dt  )]);
% xlim([-0.5 2])
% ylabel('Mean Spike Rate')
% 
% 
% end
% 
% 
% %%%%%%%%%%%%
% 
% 
% for k = 1:2
% 
%     if k == 1
% neurons = reg1;
% ct = 1;
%     else
%         neurons = reg2;
%         ct = 4;
% 
%     end
% 
% 
% trials = params.trialid{8};
% aa_norm = [];
% aa = obj.trialdat(:,neurons,trials);
% min_val = min(aa, [], 'all'); max_val = max(aa, [], 'all');
% aa_norm = normalize((aa - min_val) / (max_val - min_val), 'range', [0 1]);
% aa_r1 = aa_norm;
% aa_mean = mean(aa, 2); % Compute mean across the second dimension
% aa_mean_r1 = squeeze(aa_mean); % Remove singleton dimension, resulting in 1900 × 155
% 
% trials = params.trialid{9};
% aa_norm = [];
% aa = obj.trialdat(:,neurons,trials);
% min_val = min(aa, [], 'all'); max_val = max(aa, [], 'all');
% aa_norm = normalize((aa - min_val) / (max_val - min_val), 'range', [0 1]);
% aa_r4 = aa_norm;
% aa_mean = mean(aa, 2); % Compute mean across the second dimension
% aa_mean_r4 = squeeze(aa_mean); % Remove singleton dimension, resulting in 1900 × 155
% 
% 
% subplot(2,3,ct)
% imagesc(obj.time, [1:size(aa_mean_r1,2)], aa_mean_r1');
% xline(0,'LineWidth', 1.5)
% colormap(cmp)
% %     caxis([rang]);
% colorbar; grid off;
% box off; xlabel('time(s)'); ylabel('Trial #');
% title(['MSR R1, Date: ', num2str(obj.pth.dt  )]);
% set(gca,'FontSize',8)
% xlim([-1 2])
% 
% subplot(2,3,ct+1)
% imagesc(obj.time, [1:size(aa_mean_r4,2)], aa_mean_r4');
% xline(0,'LineWidth', 1.5)
% colormap(cmp)
% %     caxis([rang]);
% colorbar; grid off;
% % axis tight; axis square;
% box off; xlabel('time(s)'); ylabel('Trial #');
% title(['MSR R4, Date: ', num2str(obj.pth.dt  )]);
% set(gca,'FontSize',8)
% xlim([-1 2])
% 
% end
% 
% px = 75;
% py = 75;
% width = 800;
% height = 400;
% set(gcf, 'Position', [px, py, width, height]); % Set figure position and size

%%

% figure;
% 
% Ncells = size(obj.psth, 2);
% clu_m1TJ = 1:numel(params.cluid{1, 1});
% clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;
% 
% clu = 1:Ncells;
% 
% % first lick mode
% reg = clu_ALM;
% reg1 = clu_m1TJ;
% 
% 
% 
% for k = 1:2
% 
% subplot(2,3,3*k)
% 
% if k == 1
% neurons = reg1;
% else
%     neurons = reg;
% end
% 
% a = obj.trialdat(:,neurons,trials);
% 
% 
% % NORMAL MODE
% fl_time = obj.time;
% trials = params.trialid{1, 8};
% D = mean(obj.trialdat(:,neurons,trials),3);
% trials1 = params.trialid{1, 9};
% D1 = mean(obj.trialdat(:,neurons,trials1),3);
% 
% 
% % trials1 = params.trialid{1, 5};
% % D1 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 3};
% % D2 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 6};
% % D3 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 4};
% % D4 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 7};
% % D5 = mean(obj.trialdat(:,neurons,trials1),3);
% %%%%%
% % fl_time = obj.time;
% % trials = params.trialid{1, 2}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 5}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D1 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 3}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D2 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 6}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D3 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 4}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D4 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 7}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D5 = mean(obj.trialdat(:,neurons,trials1),3);
% 
% 
% max_values = max(D,[],1); 
% max_values1 = max(D1,[],1); 
% 
% % max_values1 = max(D1,[],1);
% % max_values2 = max(D2,[],1); 
% % max_values3 = max(D3,[],1);
% % max_values4 = max(D4,[],1); 
% % max_values5 = max(D5,[],1);
% 
% % D = D./max_values;
% % D1 = D1./max_values1;
% % D1 = D1./max_values1;
% % D2 = D2./max_values2;
% % D3 = D3./max_values3;
% % D4 = D4./max_values4;
% % D5 = D5./max_values5;
% 
% % D = normalize(D,2);
% % D1 = normalize(D1,2);
% 
% FL_dat = D;
% FL_dat1 = D1;
% % FL_dat1 = D1;
% % FL_dat2 = D2;
% % FL_dat3 = D3;
% % FL_dat4 = D4;
% % FL_dat5 = D5;
% 
% rang = [450:505];
% 
% FLMode = mean(FL_dat(rang,:)) - mean(FL_dat(100:300,:));
% % FLMode1 = mean(FL_dat1(rang,:)) - mean(FL_dat1(50:200,:));
% % FLMode2 = mean(FL_dat2(rang,:)) - mean(FL_dat2(50:200,:));
% % FLMode3 = mean(FL_dat3(rang,:)) - mean(FL_dat3(50:200,:));
% % FLMode4 = mean(FL_dat4(rang,:)) - mean(FL_dat4(50:200,:));
% % FLMode5 = mean(FL_dat5(rang,:)) - mean(FL_dat5(50:200,:));
% 
% 
% % FLMode1 = normalize(FLMode1,'range');
% % FLProj1 = FLMode1.*FL_dat1;
% 
% % figure;
% % plot(FLProj1)
% % hold on 
% % plot(FLProj)
% %
% 
% sz = 3;
% 
% y_r1 = FLMode.*FL_dat;
% y_r4 = FLMode.*FL_dat1;
% % y1 = FLMode.*FL_dat1;
% % y2 = FLMode2.*FL_dat2;
% % y3 = FLMode2.*FL_dat3;
% % y4 = FLMode4.*FL_dat4;
% % y5 = FLMode4.*FL_dat5;
% 
% a_r1 = mean(y_r1,2);
% a_r4 = mean(y_r4,2);
% % a1 = mean(y1,2);
% % a2 = mean(y2,2);
% % a3 = mean(y3,2);
% % a4 = mean(y4,2);
% % a5 = mean(y5,2);
% 
% windowSize = 5;
% 
% plot(obj.time, movmean(a_r1,windowSize),'LineWidth',2, 'Color','k')
% hold on 
% plot(obj.time, movmean(a_r4,windowSize),'LineWidth',2, 'Color','r')
% % title('First Lick Mode')
% title(['FL Mode , Date: ', num2str(obj.pth.dt  )]);
% box off;
% 
% xlabel('time')
% ylabel('proj')
%     xlim([-0.75 2.5])
% xline(0)
% set(gca,'FontSize',8)
% 
% 
% end
% 
% 
% ct = 1;
% 
% for k = 1:2
% 
% 
% if k == 1
% neurons = reg1;
% ct = 1;
% else
%     neurons = reg;
%     ct = 4;
% end
% 
% a = obj.trialdat(:,neurons,trials);
% 
% 
% % NORMAL MODE
% fl_time = obj.time;
% trials = params.trialid{1, 8};
% D = mean(obj.trialdat(:,neurons,trials),3);
% trials1 = params.trialid{1, 9};
% D1 = mean(obj.trialdat(:,neurons,trials1),3);
% 
% 
% trials1 = params.trialid{1, 5};
% D1 = mean(obj.trialdat(:,neurons,trials1),3);
% trials = params.trialid{1, 3};
% D2 = mean(obj.trialdat(:,neurons,trials),3);
% trials1 = params.trialid{1, 6};
% D3 = mean(obj.trialdat(:,neurons,trials1),3);
% trials = params.trialid{1, 4};
% D4 = mean(obj.trialdat(:,neurons,trials),3);
% trials1 = params.trialid{1, 7};
% D5 = mean(obj.trialdat(:,neurons,trials1),3);
% %%%%%
% % fl_time = obj.time;
% % trials = params.trialid{1, 2}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 5}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D1 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 3}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D2 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 6}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D3 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 4}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D4 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 7}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D5 = mean(obj.trialdat(:,neurons,trials1),3);
% 
% 
% max_values = max(D,[],1); 
% max_values1 = max(D1,[],1); 
% 
% % max_values1 = max(D1,[],1);
% % max_values2 = max(D2,[],1); 
% % max_values3 = max(D3,[],1);
% % max_values4 = max(D4,[],1); 
% % max_values5 = max(D5,[],1);
% 
% % D = D./max_values;
% % D1 = D1./max_values1;
% % D1 = D1./max_values1;
% % D2 = D2./max_values2;
% % D3 = D3./max_values3;
% % D4 = D4./max_values4;
% % D5 = D5./max_values5;
% 
% % D = normalize(D,2);
% % D1 = normalize(D1,2);
% 
% FL_dat = D;
% FL_dat1 = D1;
% % FL_dat1 = D1;
% % FL_dat2 = D2;
% % FL_dat3 = D3;
% % FL_dat4 = D4;
% % FL_dat5 = D5;
% 
% rang = [460:500];
% 
% FLMode = mean(FL_dat(rang,:)) - mean(FL_dat(1:300,:));
% % FLMode1 = mean(FL_dat1(rang,:)) - mean(FL_dat1(50:200,:));
% % FLMode2 = mean(FL_dat2(rang,:)) - mean(FL_dat2(50:200,:));
% % FLMode3 = mean(FL_dat3(rang,:)) - mean(FL_dat3(50:200,:));
% % FLMode4 = mean(FL_dat4(rang,:)) - mean(FL_dat4(50:200,:));
% % FLMode5 = mean(FL_dat5(rang,:)) - mean(FL_dat5(50:200,:));
% 
% 
% % FLMode1 = normalize(FLMode1,'range');
% % FLProj1 = FLMode1.*FL_dat1;
% 
% % figure;
% % plot(FLProj1)
% % hold on 
% % plot(FLProj)
% %
% 
% sz = 3;
% 
% y_r1 = FLMode.*FL_dat;
% y_r4 = FLMode.*FL_dat1;
% % y1 = FLMode.*FL_dat1;
% % y2 = FLMode2.*FL_dat2;
% % y3 = FLMode2.*FL_dat3;
% % y4 = FLMode4.*FL_dat4;
% % y5 = FLMode4.*FL_dat5;
% 
% a_r1 = mean(y_r1,2);
% a_r4 = mean(y_r4,2);
% % a1 = mean(y1,2);
% % a2 = mean(y2,2);
% % a3 = mean(y3,2);
% % a4 = mean(y4,2);
% % a5 = mean(y5,2);
% 
% % result = squeeze(sum(a .* FL_mode, 2));
% 
% D_r1 = obj.trialdat(:,neurons,params.trialid{1, 8});
% D_r1 = squeeze(sum(D_r1 .* FLMode, 2));
% D1_r4 = obj.trialdat(:,neurons,params.trialid{1, 9});
% D1_r4 = squeeze(sum(D1_r4 .* FLMode, 2));
% 
% 
% cmp = jet;
% 
% 
% subplot(2,3,ct)
% imagesc(obj.time, [1:size(D_r1,2)], D_r1');
% xline(0,'LineWidth', 1.5)
% colormap(cmp)
% %     caxis([rang]);
% colorbar; grid off;
% box off; xlabel('time(s)'); ylabel('Trial #');
% title(['FL Mode R1, Date: ', num2str(obj.pth.dt  )]);
% set(gca,'FontSize',8)
% xlim([-1 2])
% 
% subplot(2,3,ct+1)
% imagesc(obj.time, [1:size(D1_r4,2)], D1_r4');
% xline(0,'LineWidth', 1.5)
% colormap(cmp)
% %     caxis([rang]);
% colorbar; grid off;
% % axis tight; axis square;
% box off; xlabel('time(s)'); ylabel('Trial #');
% title(['FL Mode R4, Date: ', num2str(obj.pth.dt  )]);
% set(gca,'FontSize',8)
% xlim([-1 2])
% 
% end
% 
% 
% 
% px = 75;
% py = 75;
% width = 800;
% height = 400;
% set(gcf, 'Position', [px, py, width, height]); % Set figure position and size







%%






% %% Activity Modes
% 
% figure;
% 
% Ncells = size(obj.psth, 2);
% clu_m1TJ = 1:numel(params.cluid{1, 1});
% clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;
% 
% clu = 1:Ncells;
% 
% % first lick mode
% reg = clu_ALM;
% reg1 = clu_m1TJ;
% 
% 
% 
% % for k = 1:2
% 
% 
% % if k == 1
% % neurons = reg1;
% % else
%     neurons = reg1;
% % end
% 
% 
% 
% % NORMAL MODE
% fl_time = obj.time;
% trials = params.trialid{1, 2};
% D = mean(obj.trialdat(:,neurons,trials),3);
% trials1 = params.trialid{1, 5};
% D1 = mean(obj.trialdat(:,neurons,trials1),3);
% trials = params.trialid{1, 3};
% D2 = mean(obj.trialdat(:,neurons,trials),3);
% trials1 = params.trialid{1, 6};
% D3 = mean(obj.trialdat(:,neurons,trials1),3);
% trials = params.trialid{1, 4};
% D4 = mean(obj.trialdat(:,neurons,trials),3);
% trials1 = params.trialid{1, 7};
% D5 = mean(obj.trialdat(:,neurons,trials1),3);
% %%%%%
% % fl_time = obj.time;
% % trials = params.trialid{1, 2}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 5}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D1 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 3}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D2 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 6}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D3 = mean(obj.trialdat(:,neurons,trials1),3);
% % trials = params.trialid{1, 4}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D4 = mean(obj.trialdat(:,neurons,trials),3);
% % trials1 = params.trialid{1, 7}';trials = trials(mod(trials, 10) >= 1 & mod(trials, 10) <= 4);
% % D5 = mean(obj.trialdat(:,neurons,trials1),3);
% 
% 
% max_values = max(D,[],1); 
% max_values1 = max(D1,[],1);
% max_values2 = max(D2,[],1); 
% max_values3 = max(D3,[],1);
% max_values4 = max(D4,[],1); 
% max_values5 = max(D5,[],1);
% 
% D = D./max_values;
% D1 = D1./max_values1;
% D2 = D2./max_values2;
% D3 = D3./max_values3;
% D4 = D4./max_values4;
% D5 = D5./max_values5;
% 
% % D = normalize(D,2);
% % D1 = normalize(D1,2);
% 
% 
% FL_dat = D;
% FL_dat1 = D1;
% FL_dat2 = D2;
% FL_dat3 = D3;
% FL_dat4 = D4;
% FL_dat5 = D5;
% 
% rang = [450:550];
% 
% FLMode = mean(FL_dat(rang,:)) - mean(FL_dat(100:300,:));
% FLMode1 = mean(FL_dat1(rang,:)) - mean(FL_dat1(50:200,:));
% FLMode2 = mean(FL_dat2(rang,:)) - mean(FL_dat2(50:200,:));
% FLMode3 = mean(FL_dat3(rang,:)) - mean(FL_dat3(50:200,:));
% FLMode4 = mean(FL_dat4(rang,:)) - mean(FL_dat4(50:200,:));
% FLMode5 = mean(FL_dat5(rang,:)) - mean(FL_dat5(50:200,:));
% 
% 
% % FLMode1 = normalize(FLMode1,'range');
% FLProj_r1 = FLMode.*FL_dat;
% FLProj1 = FLMode1.*FL_dat1;
% 
% % figure;
% % plot(FLProj1)
% % hold on 
% % plot(FLProj)
% 
% 
% %
% 
% windowSize = 2;
% sz = 3;
% 
% 
% y = FLMode.*FL_dat;
% y1 = FLMode.*FL_dat1;
% y2 = FLMode2.*FL_dat2;
% y3 = FLMode2.*FL_dat3;
% y4 = FLMode4.*FL_dat4;
% y5 = FLMode4.*FL_dat5;
% 
% alph = 0.3;
% 
% a = mean(y,2);
% a1 = mean(y1,2);
% a2 = mean(y2,2);
% a3 = mean(y3,2);
% a4 = mean(y4,2);
% a5 = mean(y5,2);
% 
% if k == 1
% ii = [1 2 3];
% else
% ii = [4 5 6];
% end
% 
% windowSize = 20;
% 
% subplot(2,3,ii(1))
% plot(obj.time, movmean(a,windowSize),'LineWidth',1)
% hold on 
% plot(obj.time, movmean(a1,windowSize),'LineWidth',1)
% xlim([-0.5 2.5])
% xline(0)
% subplot(2,3,ii(2))
% plot(obj.time, movmean(a2,windowSize),'LineWidth',1)
% hold on 
% plot(obj.time, movmean(a3,windowSize),'LineWidth',1)
% xlim([-0.5 2.5])
% xline(0)
% subplot(2,3,ii(3))
% plot(obj.time, movmean(a4,windowSize),'LineWidth',1)
% hold on 
% plot(obj.time, movmean(a5,windowSize),'LineWidth',1)
% xlim([-0.5 2])
% xline(0)
% 
% % end
% 
% 
% a = 2;
% 
% %%
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % %% F2 Mean Spike Rate
% % figure;
% % 
% % for k = 1:2
% % 
% %     if k == 1
% % neurons = reg1;
% %     else
% %         neurons = reg;
% %     end
% % 
% % 
% % trials = params.trialid{8};
% % 
% % aa_norm = [];
% % aa = obj.trialdat(:,neurons,trials);
% % min_val = min(aa, [], 'all'); max_val = max(aa, [], 'all');
% % aa_norm = normalize((aa - min_val) / (max_val - min_val), 'range', [0 1]);
% % aa_r1 = aa_norm;
% % 
% % trials = params.trialid{9};
% % 
% % aa_norm = [];
% % aa = obj.trialdat(:,neurons,trials);
% % min_val = min(aa, [], 'all'); max_val = max(aa, [], 'all');
% % aa_norm = normalize((aa - min_val) / (max_val - min_val), 'range', [0 1]);
% % aa_r4 = aa_norm;
% % 
% % 
% % 
% % 
% % subplot(1,2,k)
% % % trials = allTrials{i};
% % D1 = mean(aa_r1,3);
% % D2 = mean(aa_r4,3);
% % hold on 
% % plot(obj.time,movmean(mean(D1,2),windowSize), 'LineWidth',2,'Color','k')
% % hold on 
% % plot(obj.time,movmean(mean(D2,2),windowSize), 'LineWidth',2,'Color','r');
% % axis tight;
% % % legend('pre','post')
% % xline(0)
% % title(['Date : ', num2str(obj.pth.dt  )]);
% % xlim([-2 2.5])
% % ylabel('Mean Spike Rate')
% % 
% % 
% % end
% 
% 
% %%
% % 
% % 
% reg = clu_ALM; % P2
% reg1 = clu_m1TJ;% P1
% 
% 
% 
% sz = 10;
% NT = obj.bp.Ntrials;
% lw = 2;
% ct = 1;
% n = 30;
% 
% neurons = reg1;
% 
% col = {[1 0 0] [0 0 1] [0.5 0 0] [0 0 0.5]};
% % 2 is LR1, 4 RR1, 5 is LR16, 7 is RR16
% 
% for i = 1:n:length(neurons)
%     f = figure;
%     for j = 1:n
% 
%         try
%         ax = nexttile;
% 
% %         plot(obj.time,obj.psth(:,neurons(ct),2),'Color',col{1},'LineWidth',lw); hold on 
% %         plot(obj.time,obj.psth(:,neurons(ct),4),'Color',col{2},'LineWidth',lw); hold on 
% %         plot(obj.time,obj.psth(:,neurons(ct),5),'Color',col{3},'LineWidth',lw); hold on 
% %         plot(obj.time,obj.psth(:,neurons(ct),7),'Color',col{4},'LineWidth',lw); hold on 
% 
%         plot(obj.time,movmean(obj.psth(:,neurons(ct),8),5),'Color',col{1},'LineWidth',lw); hold on
%         % plot(obj.time,movmean(obj.psth(:,neurons(ct),2),15),'Color',col{2},'LineWidth',lw); hold on 
%         % plot(obj.time,movmean(obj.psth(:,neurons(ct),3),15),'Color',col{3},'LineWidth',lw); hold on 
% 
% %         legend('LR1','LR16','RR1','RR16')
% %         plot(objs.time,objs.psth(:,neurons(ct),3),'Color','b','LineWidth',1.5); hold on 
% 
% 
% title(['Cell = ', num2str(neurons(ct))]);
% xlabel(['time from ', num2str(params.alignEvent)]);
% % xlabel('time')
% xline(-2)
% xline(0)
% xlim([-0.5 1.5]);
% set(gca,'FontSize',sz)
% % axis square
% 
% ylabel('firing rate')
% box off
%         ct = ct + 1;
%     end
% end
% end
% % 
% % a = 2;
% 
% 
% % xlabel('Time from VTA Stim')
% a = 2;
% %%
% 
% 
% % % Finding Exploratory Trials %%
% % Length(isnan(Length)) = 0;
% % 
% % gc = obj.bp.ev.goCue;
% % allT = params(1).trialid{10};
% % clear numLicks
% % clear varianceAngle
% % 
% % 
% % for i = 1:obj.bp.Ntrials
% %     close all
% %     gc = obj.bp.ev.goCue(i);
% %     LickL = cell2mat(obj.bp.ev.lickL(i));
% %     LickL = LickL(LickL > gc);
% %     if ~isempty(LickL)
% %         first_element = LickL(1);
% %         difference = first_element - gc;
% % 
% %         plot(Length(:,i))
% %         hold on
% %         plot(angle(:,i))
% %         hold on
% %         xline(500,'Color','k','LineWidth',2)
% %         xline(500+round(difference/.005),'Color','r','LineWidth',2)
% % 
% % 
% %         nl = find_bouts(Length(495:505+round(difference/.005),i), gc,15,40);
% % 
% %         numLicks(i) = nansum(Length(500:500+round(difference/.005),i));
% %         numLicks(i) = numel(nl);
% %         varianceAngle(i) = nanstd(angle(495:505+round(difference/.005),i));
% % 
% % 
% % 
% % 
% %     else
% % 
% %         plot(Length(:,i))
% %         hold on
% %         plot(angle(:,i))
% %         hold on
% %         xline(500,'Color','k','LineWidth',2)
% %         xline(500+round(difference/.005),'Color','r','LineWidth',2)
% % 
% %         numLicks(i) = 0;
% %         varianceAngle(i) = 0;
% %     end
% % 
% % end
% % 
% % nanmean(varianceAngle(find(numLicks < 2)))
% % 
% % 
% % figure;
% % plot(numLicks,varianceAngle,'.')
% 
% 
% % 
% % % % Calculating Angle Variance
% % Length(isnan(Length)) = 0;
% % 
% % figure;
% % 
% % ttype = [1 2 3];
% % rw = [1 4];
% % 
% % for iii = 1:2
% % Ttype = rw(iii);
% % for kk = 1:numel(ttype)
% % gc = obj.bp.ev.goCue;
% % ie = intersect(find(obj.bp.trialTypes == ttype(kk)),find(obj.bp.rewardedLick == Ttype),'stable');
% % ie = intersect(ie,find(obj.bp.hit == 1),'stable');
% % allT = ie;
% % 
% % clear numLicks meanValues trialAngle_stdev
% % clear varianceAngle stdValues trialAngle_mean dat
% % clear ii cellData LickL maxLength angle_stdev angle_mean
% % 
% % for i = 1:length(allT)
% %     gc = obj.bp.ev.goCue(allT(i));
% %     LickL = cell2mat(obj.bp.ev.lickL(allT(i)));
% %     LickL = LickL(LickL > gc);
% %     if ~isempty(LickL) 
% % 
% %         hold on
% % %         plot(velocity(:,i))
% % %         plot(angle(:,i))
% % %         hold on
% % %         xline(500,'Color','k','LineWidth',2)
% % %         xline(500+round(difference/.005),'Color','r','LineWidth',2)
% % 
% % 
% %         nl = find_bouts(angle(:,allT(i)), gc,7,40);
% % %         nl = find_bouts(Length(:,allT(i)), gc,7,40);
% % %         velocity(find(velocity == 0)) = nan;
% % %         nl = find_bouts(velocity(:,allT(i)), gc,5,40);
% % 
% % %         numLicks(i) = nansum(Length(500:500+round(difference/.005),i));
% % 
% % %         varianceAngle{i} = nanstd(angle(495:505+round(difference/.005),i));
% %          
% %         
% %         for ii = 1:numel(nl)
% %             cellData = nl{ii};  % Get the data from the cell
% %         
% %             % Calculate mean and standard deviation
% %             meanValues(ii) = median(cellData);
% %             stdValues(ii) = std(cellData);
% % 
% %         end
% %         
% %         
% %         
% %         trialAngle_mean{i} = meanValues;
% %         trialAngle_stdev{i} = stdValues;
% % 
% % 
% % 
% % 
% %     else
% % 
% % %         plot(Length(:,i))
% % %         hold on
% % %         plot(angle(:,i))
% % %         hold on
% % %         xline(500,'Color','k','LineWidth',2)
% % %         xline(500+round(difference/.005),'Color','r','LineWidth',2)
% % 
% %         numLicks(i) = 0;
% %         varianceAngle(i) = 0;
% %     end
% % 
% % end
% % 
% % 
% % angle_stdev = trialAngle_stdev(~cellfun(@isempty, trialAngle_stdev));
% % maxLength = max(cellfun(@numel, angle_stdev));
% % angle_stdev_padded = cellfun(@(x) [x NaN(1, maxLength - numel(x))], angle_stdev, 'UniformOutput', false);
% % angle_stdev = cell2mat(angle_stdev_padded');
% % 
% % angle_mean = trialAngle_mean(~cellfun(@isempty, trialAngle_mean));
% % maxLength = max(cellfun(@numel, angle_mean));
% % angle_mean_padded = cellfun(@(x) [x NaN(1, maxLength - numel(x))], angle_mean, 'UniformOutput', false);
% % angle_mean = cell2mat(angle_mean_padded');
% % 
% % 
% % % angle_stdev
% % % angle_mean
% % 
% % % subplot(1,numel(ttype),kk)
% % 
% % dat = angle_mean;
% % 
% % mean_values = nanmean(dat,1);
% % std_values =  nanstd(dat,1);
% % 
% % n = size(dat, 2);
% % sem_values = std_values ./ sqrt(n);
% % 
% % alpha = 0.05;
% % z_critical = norminv(1 - alpha/2);
% % ci_values = z_critical * sem_values;
% % 
% % % 
% % if iii == 1
% % 
% %     if kk == 1
% %         c = [ 1 0 0];
% %     elseif kk == 2
% %         c = [ 0.35  0.35  0.35 ];
% % 
% %     elseif kk == 3
% %         c = [ 0 0 1];
% %     end
% % else
% %     if kk == 1
% %         c = [ 0.65 0 0];
% %     elseif kk == 2
% %         c = [ 0 0  0 ];
% % 
% %     elseif kk == 3
% %         c = [ 0 0 0.65];
% %     end
% % end
% % 
% % % subplot(1,3,kk)
% % 
% % hold on 
% % x = 1:size(dat, 2);  % x-axis values
% % % errorbar(x, mean_values, ci_values, '-o');
% % 
% % errorbar(x, mean_values, ci_values,'.-',...
% %         'MarkerSize', 25,'Color',c,'LineWidth',1.5,'CapSize', 10);
% % 
% % % plot(x, mean_values,'.-',...
% % %         'MarkerSize', 25,'Color',c,'LineWidth',1.5);
% % 
% % 
% % xlabel('Lick #');
% % ylabel('mean angle');
% % title(['Position: ' num2str(kk)]);
% % axis square; box off;
% % 
% % xlim([0, 9.5]);
% % 
% % 
% % end
% % 
% % end
% 
% 
% 
% 
% % %%
% % % 
% % % 
% % reg = clu_ALM; % P2
% % reg1 = clu_m1TJ;% P1
% % 
% % 
% % 
% % sz = 10;
% % NT = obj.bp.Ntrials;
% % lw = 2;
% % ct = 1;
% % n = 30;
% % 
% % neurons = reg1;
% % 
% % col = {[1 0 0] [0 0 1] [0.5 0 0] [0 0 0.5]};
% % % 2 is LR1, 4 RR1, 5 is LR16, 7 is RR16
% % 
% % for i = 1:n:length(neurons)
% %     f = figure;
% %     for j = 1:n
% % 
% %         try
% %         ax = nexttile;
% % 
% % %         plot(obj.time,obj.psth(:,neurons(ct),2),'Color',col{1},'LineWidth',lw); hold on 
% % %         plot(obj.time,obj.psth(:,neurons(ct),4),'Color',col{2},'LineWidth',lw); hold on 
% % %         plot(obj.time,obj.psth(:,neurons(ct),5),'Color',col{3},'LineWidth',lw); hold on 
% % %         plot(obj.time,obj.psth(:,neurons(ct),7),'Color',col{4},'LineWidth',lw); hold on 
% % 
% %         plot(obj.time,movmean(obj.psth(:,neurons(ct),8),5),'Color',col{1},'LineWidth',lw); hold on
% %         % plot(obj.time,movmean(obj.psth(:,neurons(ct),2),15),'Color',col{2},'LineWidth',lw); hold on 
% %         % plot(obj.time,movmean(obj.psth(:,neurons(ct),3),15),'Color',col{3},'LineWidth',lw); hold on 
% % 
% % %         legend('LR1','LR16','RR1','RR16')
% % %         plot(objs.time,objs.psth(:,neurons(ct),3),'Color','b','LineWidth',1.5); hold on 
% % 
% % 
% % title(['Cell = ', num2str(neurons(ct))]);
% % xlabel(['time from ', num2str(params.alignEvent)]);
% % % xlabel('time')
% % xline(-2)
% % xline(0)
% % xlim([-0.5 1.5]);
% % set(gca,'FontSize',sz)
% % % axis square
% % 
% % ylabel('firing rate')
% % box off
% %         ct = ct + 1;
% %     end
% % end
% % end
% % % 
% % % a = 2;
% % 
% 
% 
% 
% %%
% % data_zscored = zeros(size(data));
% % for neuron = 1:size(data, 2)
% %     data_zscored(:, neuron, :) = zscore(data(:, neuron, :), 0, [1 3]);
% % end
% 
% 
% 
% 
% 
% 
% %%
% 
% % Kinematics1 = Kinematics1(:,1:240);
% 
% 
% % cmp = jet;
% % figure;
% % subplot(1,2,1)
% % 
% % % n = 1;
% % rang = [0 40];
% % % rang = [-n n];
% % 
% % imagesc(obj.time, [1:size(Kinematics1,2)], Kinematics1', 'Interpolation', 'bilinear');
% % xline(0,'LineWidth', 1.5)
% % colormap(cmp)
% %     caxis([rang]);
% % colormap(cmp)
% % colorbar; grid off; axis tight; axis square;
% % box off; xlabel('time(s)'); ylabel('Trial #'); title([kinfeat,' plot'], 'Interpreter', 'none');
% % set(gca,'FontSize',15)
% 
% %%
% % rang = [1:1500];
% % 
% % subplot(1,2,2)
% % for i = 2:7
% %     plot(nanmean(Kinematics1([rang],params.trialid{1,i}),2));
% %     hold on
% % end
% %  axis tight; axis square; box off;
% 
% % figure;
% % 
% % cmp = jet;
% % 
% % % imagesc(obj.time, [1:size(Kinematics1,2)], Kinematics1', 'Interpolation', 'bilinear');
% % imagesc(obj.time, [1:size(Kinematics1,2)], Kinematics1');
% % xline(0,'LineWidth', 1.5)
% % colormap(jet)
% % %     caxis([rang]);
% % colorbar; grid off; axis tight; axis square;
% % box off; xlabel('time(s)'); ylabel('Trial #'); title([kinfeat,' plot'], 'Interpreter', 'none');
% % set(gca,'FontSize',15)
% 
% % %% Heatmap - rewardedLick
% 
% 
% %% Heatmap Position
% 
% % rang = [500:550];
% % crang = [-3.5 3.5];
% % cmp = linspecer;
% % 
% % P1 = obj.psth(:,[reg],2);
% % [max_vals,max_ind] = max(P1([rang],:), [], 1);
% % [~, idx] = sort(max_ind, 'ascend');
% % 
% % figure;
% % for i = 1:3
% % subplot(2,3,i)
% % P1 = obj.psth(:,[reg],i+1);
% % P1 = P1(:,idx);
% % P1 = normalize(P1,1);
% % imagesc(obj.time, 1:size(P1,2), P1');
% % colormap(cmp);
% % colorbar;
% % caxis([crang]);
% % view([0, 90]);
% % grid off; axis tight; axis square
% % box off; xlabel(['time from ',params.alignEvent], 'Interpreter', 'none'); ylabel('Neuron #  ALM');
% % title('Heatmap for Position 1 ')
% % set(gca,'FontSize',sz)
% % if i ==1
% %     title('Position 1')
% % elseif i ==2
% %     title('Position 2')
% % elseif i ==3
% %     title('Position 3')
% % end
% % end
% % 
% % 
% % 
% % P1 = obj.psth(:,[reg1],2);
% % [max_vals,max_ind] = max(P1([rang],:), [], 1);
% % [~, idx] = sort(max_ind, 'ascend');
% % 
% % for i = 1:3
% % subplot(2,3,3+i)
% % P1 = obj.psth(:,[reg1],i+1);
% % P1 = P1(:,idx);
% % P1 = normalize(P1,1);
% % imagesc(obj.time, 1:size(P1,2), P1');
% % colormap(cmp);
% % colorbar;
% % caxis([crang]);
% % view([0, 90]);
% % grid off; axis tight; axis square
% % box off; xlabel(['time from ',params.alignEvent], 'Interpreter', 'none'); ylabel('Neuron #  M1TJ');
% % title('Heatmap for Position 1 ')
% % set(gca,'FontSize',sz)
% % if i ==1
% %     title('Position 1')
% % elseif i ==2
% %     title('Position 2')
% % elseif i ==3
% %     title('Position 3')
% % end
% % end
% 
% 
% 
% 
% %% Heatmap Position
% 
% % rang = [400:550];
% % crang = [-3.5 3.5];
% % cmp = linspecer;
% % 
% % P1 = obj.psth(:,[reg1],2);
% % [max_vals,max_ind] = max(P1([rang],:), [], 1);
% % [~, idx] = sort(max_ind, 'ascend');
% % 
% % P2 = obj.psth(:,[reg1],5);
% % [max_vals,max_ind] = max(P2([rang],:), [], 1);
% % [~, idx1] = sort(max_ind, 'ascend');
% % 
% % 
% % 
% % figure;
% % for i = 1:3
% % subplot(2,3,i)
% % 
% % P1 = obj.psth(:,[reg1],i+1);
% % P1 = P1(:,idx);
% % P1 = mean(P1,2);
% % 
% % P2 = obj.psth(:,[reg1],i+4);
% % P2 = P2(:,idx1);
% % P2 = mean(P2,2);
% % 
% % % P1 = normalize(P1,1);
% % 
% % 
% % % imagesc(obj.time, 1:size(P1,2), P1');
% % % colormap(cmp);
% % % colorbar;
% % % caxis([crang]);
% % % view([0, 90]);
% % plot(obj.time, movmean(P1,20), 'LineWidth',1.5);
% % hold on 
% % plot(obj.time, movmean(P2,20), 'LineWidth',1.5);
% % 
% % 
% % grid off; axis tight; axis square
% % box off; xlabel(['time from ',params.alignEvent], 'Interpreter', 'none'); ylabel('Neuron #  ALM');
% % title('Heatmap for Position 1 ')
% % set(gca,'FontSize',sz)
% % xlim([-0.75 2])
% % 
% % if i ==1
% %     title('Position 1')
% % elseif i ==2
% %     title('Position 2')
% % elseif i ==3
% %     title('Position 3')
% % end
% % end
% % 
% % 
% % 
% % P1 = obj.psth(:,[reg],2);
% % [max_vals,max_ind] = max(P1([rang],:), [], 1);
% % [~, idx] = sort(max_ind, 'ascend');
% % 
% % P2 = obj.psth(:,[reg],5);
% % [max_vals,max_ind] = max(P2([rang],:), [], 1);
% % [~, idx1] = sort(max_ind, 'ascend');
% % 
% % 
% % for i = 1:3
% % subplot(2,3,3+i)
% % 
% % P1 = obj.psth(:,[reg],i+1);
% % P1 = P1(:,idx);
% % P1 = mean(P1,2);
% % 
% % P2 = obj.psth(:,[reg],i+4);
% % P2 = P2(:,idx1);
% % P2 = mean(P2,2);
% % 
% % % P1 = normalize(P1,1);
% % 
% % plot(obj.time, movmean(P1,20), 'LineWidth',1.5)
% % hold on 
% % plot(obj.time, movmean(P2,20), 'LineWidth',1.5)
% % 
% % grid off; axis tight; axis square
% % box off; xlabel(['time from ',params.alignEvent], 'Interpreter', 'none'); ylabel('Neuron #  M1TJ');
% % xlim([-0.75 2])
% % title('Heatmap for Position 1 ')
% % set(gca,'FontSize',sz)
% % if i ==1
% %     title('Position 1')
% % elseif i ==2
% %     title(['Position 2 ' num2str(obj.pth.dt  )]);
% % 
% % elseif i ==3
% %     title('Position 3')
% % end
% % end
% 
% 
% 
% %% PCA
% % PCA - Position
% % figure;
% % for i = 1:3
% % [~, coeffs] = pca(obj.psth(:,[reg],i+1), 'NumComponents', 8);
% % subplot(2,3,i)
% % plot(obj.time,coeffs(:, 1:3),'LineWidth',2); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) ALM')
% % set(gca,'FontSize',sz)
% % if i ==1
% %     title('Position 1')
% % elseif i ==2
% %     title('Position 2')
% % elseif i ==3
% %     title('Position 3')
% % end
% % end
% % 
% % for i = 1:3
% % [~, coeffs] = pca(obj.psth(:,[reg1],i+1), 'NumComponents', 8);
% % subplot(2,3,3+i)
% % plot(obj.time,coeffs(:, 1:3),'LineWidth',2); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) M1TJ')
% % set(gca,'FontSize',sz)
% % if i ==1
% %     title('Position 1')
% % elseif i ==2
% %     title('Position 2')
% % elseif i ==3
% %     title('Position 3')
% % end
% % end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%
% 
% % figure;
% % % PCA - rewardedLick
% % subplot(2,3,1)
% % [~, coeffs] = pca(obj.psth(:,[reg1],8), 'NumComponents', 8);
% % plot(obj.time,coeffs(:, 1:3),'LineWidth',3); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) ALM')
% % title('Rewarded after Lick #1')
% % set(gca,'FontSize',sz)
% % 
% % subplot(2,3,2)
% % [~, coeffs] = pca(obj.psth(:,[reg1],9), 'NumComponents', 8);
% % plot(obj.time,coeffs(:, 1:3),'LineWidth',3); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) ALM')
% % title('Rewarded after Lick #4')
% % set(gca,'FontSize',sz)
% % 
% % subplot(2,3,3)
% % [~, coeffs] = pca(obj.psth(:,[reg1],9)-obj.psth(:,[reg1],8), 'NumComponents',8);
% % plot(obj.time,coeffs(:, 1:3),'LineWidth',3); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) ALM')
% % title('R4 - R1 P1')
% % set(gca,'FontSize',sz)
% % 
% % subplot(2,3,4)
% % [~, coeffs] = pca(obj.psth(:,[reg],8), 'NumComponents', 8);
% % plot(obj.time,coeffs(:, 1:3),'LineWidth',3); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) M1TJ')
% % title('Rewarded after Lick #1')
% % set(gca,'FontSize',sz)
% % 
% % subplot(2,3,5)
% % [~, coeffs1] = pca(obj.psth(:,[reg],9), 'NumComponents', 8);
% % plot(obj.time,coeffs1(:, 1:3),'LineWidth',3); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) M1TJ')
% % title('Rewarded after Lick #4')
% % set(gca,'FontSize',sz)
% % 
% % 
% % subplot(2,3,6)
% % [~, coeffs] = pca(obj.psth(:,[reg],9)-obj.psth(:,[reg],8), 'NumComponents', 8);
% % plot(obj.time,coeffs(:, 1:3),'LineWidth',3); xline(0);
% % grid off; axis square; box off; xlim([-1.85 2]);
% % xlabel(['time from ',params.alignEvent])
% % ylabel('Activity (a.u) M1TJ')
% % title('R4 - R1 P2')
% % set(gca,'FontSize',sz)
% 
% %%
% % perform PCA on the data
% 
% % figure; 
% % subplot(1,2,1)
% % [coeffs, score, latent, tsquared, explained, mu] = pca(obj.psth(:,[reg],8), 'NumComponents', 8);
% % [coeffs1, score1, latent, tsquared, explained1, mu] = pca(obj.psth(:,[reg],9), 'NumComponents', 8);
% % 
% % scatter3(score(1:500,1),score(1:500,2),score(1:500,3),[],[0.7 0.7 1],'filled')
% % hold on 
% % scatter3(score(501:700,1),score(501:700,2),score(501:700,3),[],[0.25 0.25 0.85],'filled')
% % hold on 
% % scatter3(score(701:end,1),score(701:end,2),score(701:end,3),[],[0 0 0.5],'filled')
% % hold on 
% % 
% % scatter3(score1(1:500,1),score1(1:500,2),score1(1:500,3),[],[1 0.7 0.7],'filled')
% % hold on 
% % 
% % scatter3(score1(501:700,1),score1(501:700,2),score1(501:700,3),[],[0.8 0.2 0.2],'filled')
% % hold on 
% % 
% % scatter3(score1(701:end,1),score1(701:end,2),score1(701:end,3),[],[0.5 0 0],'filled')
% % 
% % xlabel('PC1')
% % ylabel('PC2')
% % zlabel('PC3')
% % axis square;
% % 
% % subplot(1,2,2)
% % [coeffs, score, latent, tsquared, explained, mu] = pca(obj.psth(:,[reg1],8), 'NumComponents', 8);
% % [coeffs1, score1, latent, tsquared, explained1, mu] = pca(obj.psth(:,[reg1],9), 'NumComponents', 8);
% % 
% % scatter3(score(1:500,1),score(1:500,2),score(1:500,3),[],[0.7 0.7 1],'filled')
% % hold on 
% % scatter3(score(501:700,1),score(501:700,2),score(501:700,3),[],[0.25 0.25 0.85],'filled')
% % hold on 
% % scatter3(score(701:end,1),score(701:end,2),score(701:end,3),[],[0 0 0.5],'filled')
% % hold on 
% % 
% % scatter3(score1(1:500,1),score1(1:500,2),score1(1:500,3),[],[1 0.7 0.7],'filled')
% % hold on 
% % 
% % scatter3(score1(501:700,1),score1(501:700,2),score1(501:700,3),[],[0.8 0.2 0.2],'filled')
% % hold on 
% % 
% % scatter3(score1(701:end,1),score1(701:end,2),score1(701:end,3),[],[0.5 0 0],'filled')
% % 
% % xlabel('PC1')
% % ylabel('PC2')
% % zlabel('PC3')
% % axis square;
% 
% %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [boutss] = find_bouts(arr, gc,min,max)
% 
% % boutsy = find_bouts(y, gc);
% 
% % This function finds continuous sets of non-NaN values in a given array
% % that meet the length criteria of >10 and <40 and returns them in a cell
% % array.
% 
% non_nans = find(~isnan(arr));
% boutss = {};
% start_index = 1;  % Initialize start_index variable
% for j = 1:length(gc)
%     for i = 1:length(non_nans)
%         % Check if the current index in arr is greater than or equal to the threshold in gc
%         if non_nans(i) >= gc(j)
%             % Check if the current index is the start of a new continuous set of non-NaN values
%             if i == 1 || non_nans(i) ~= non_nans(i-1) + 1
%                 start_index = non_nans(i);
%             end
% 
%             % Check if the current index is the end of the current continuous set of non-NaN values
%             if i == length(non_nans) || non_nans(i) ~= non_nans(i+1) - 1
%                 end_index = non_nans(i);
%                 bout = arr(start_index:end_index);
%                 if length(bout) > min && length(bout) < max
%                     boutss{end+1} = bout;
%                 end
%             end
%         end
%     end
% end
% end