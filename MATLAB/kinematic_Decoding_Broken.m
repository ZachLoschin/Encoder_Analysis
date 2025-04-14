% Finding "Kinematic Modes"
clear,clc

sz = 14;

% d = 'C:\Users\LabUser\Desktop\Cortical Disengagement\uninstructedMovements_v2-main';
% addpath(genpath(fullfile(d,'utils')))
% addpath(genpath(fullfile(d,'DataLoadingScripts')))
% addpath(genpath(fullfile(d,'funcs')))
% rmpath(genpath(fullfile(d,'fig1')));
% 
% addpath 'C:\Users\LabUser\Desktop\Cortical Disengagement\uninstructedMovements_v2-main\base code\functions_td'
% addpath 'C:\Users\LabUser\Desktop\Cortical Disengagement\uninstructedMovements_v2-main\ObjVis\warp'
% 
d = 'C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\Disengagement_Analysis_2025';
addpath(d)
addpath(genpath(fullfile(d,'utils')))
addpath(genpath(fullfile(d,'zutils')))
addpath(genpath(fullfile(d,'DataLoadingScripts')))
addpath(genpath(fullfile(d,'funcs')))
addpath("C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\disengagement\ObjVis")

%% PARAMETERS
params.alignEvent          = 'firstLick'; % 'jawOnset' 'goCue'  'moveOnset'  'firstLick'  'lastLick' 'fourthLick'
params.behav_only = 0;
% time warping only operates on neural data for now.
params.timeWarp            = 0;  % piecewise linear time warping - each lick duration on each trial gets warped to median lick duration for that lick across trials
params.nLicks              = 12; % number of post go cue licks to calculate median lick duration for and warp individual trials to

params.lowFR               = 0.5; % remove clusters with firing rates across all trials less than this val

params.condition(1) = {'hit==1 | hit==0' };    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==0'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1 & rewardedLick == 4'};    % left to right         % right hits, no stim, aw off
params.condition(end+1) = {'hit==1' };    % left to right         % right hits, no stim, aw off
% 
% params.condition(1) = {'hit==1' };    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & rewardedLick == 6'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 1'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 1& rewardedLick == 6'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 2& rewardedLick == 6'};    % left to right         % right hits, no stim, aw off
% params.condition(end+1) = {'hit==1 & trialTypes == 3& rewardedLick == 6'};    % left to right         % right hits, no stim, aw off

% time from align event to grab data for
params.tmin = -2.5;
params.tmax = 7;
params.dt = 1/200;

% smooth with causal gaussian kernel
params.smooth = 5;

% cluster qualities to use
% params.quality = {'ok','good','mua','great','fair'}; % accepts any cell array of strings - special character 'all' returns clusters of any quality
% params.quality = {'ok','good'}; % accepts any cell array of strings - special character 'all' returns clusters of any quality
% params.quality = {'good','excellent','fair','ok',' '}; % accepts any cell array of strings - special character 'all' returns clusters of any quality
params.quality = {'good','excellent',' good', 'good '}; % accepts any cell array of strings - special character 'all' returns clusters of any quality


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

% datapth = 'C:\Users\LabUser\Desktop\Cortical Disengagement\uninstructedMovements_v2-main\data';
datapth = 'C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\Disengagement_Analysis_2025\processed sessions\r14';

meta = [];

meta = loadTD(meta,datapth);
% meta = loadYH(meta,datapth);

params.probe = {meta.probe}; 
%% LOAD DATA

[obj,params] = loadSessionData(meta,params,params.behav_only );

for sessix = 1:numel(meta)
    me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
end

%% Define the folder where the preprocessed data should go
preprocessed_folder = 'C:\Users\zachl\OneDrive\BU_YEAR1\Research\Tudor_Data\Disengagement_Analysis_2025\preprocessed_data';

% Construct the full folder path based on the animal number and date
save_folder = fullfile(preprocessed_folder, meta.anm, meta.date);

% Create the directory if it doesn't exist
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
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

% if obj.pth.anm == 'TD13d'
% for i = 1:numel(params.trialid)
%     params.trialid{i}(params.trialid{i} > 278) = []; % Remove values > 278
% end
% end




conds2use = [1];                      % With reference to 'params.condition'
kinfeat = 'tongue_length';    % top_tongue_xvel_view2 | motion_energy | nose_xvel_view1 | jaw_yvel_view2 | trident_yvel_view1
sessix = 1;

psthForProj = [];
for c = conds2use
    condtrix = params(sessix).trialid{c};                                           % Get the trials from this condition
% if obj.pth.anm == 'TD13d'
% condtrix = condtrix(1:278);
% end
    condpsth = obj(sessix).trialdat(:,:,condtrix);                                  % Take the single trial PSTHs for these trials

condtrix(end) = [];
end




% 
% if obj.pth.anm == 'TD8d'
% condtrix = condtrix(1:298);
% end

kinix =  find(strcmp(kin(sessix).featLeg,kinfeat));
Length = kin.dat(:,condtrix,kinix);


%%

for i = 1:obj.bp.Ntrials



    nlicks = obj.bp.ev.lickL{i};
    gc = obj.bp.ev.goCue(i);

    if ~isempty(nlicks)

        lickspost = nlicks - gc;
    num_licks_postGC(i) = numel(find(lickspost < 0.7));
    else

    num_licks_postGC(i) = 0;
    end

end


trials = find(num_licks_postGC > 3);
trials1 = find(num_licks_postGC < 4);

%%

% ax = figure;
% allvel = kdat;
% % allvel1 = kin.dat(:,:,4); 27 28
% % allvel2 = kin.dat(:,:,3);
% % allvel3 = kin.dat(:,:,3);
% a = size(allvel); 
% 
% s = size(allvel);
% for kk = 1:a(2)
% v = allvel(:,kk); v(find(v == 0)) = NaN;
% % v1 = allvel1(:,kk); v1(find(v1 == 0)) = NaN;
% % v2 = allvel2(:,kk); v2(find(v2 == 0)) = NaN;
% % v3 = allvel3(:,kk); v3(find(v3 == 0)) = NaN;
% 
%     cla(ax)
%     plot(obj.time, movmean(v,1), 'LineWidth',1.5)
%     hold on
%     % plot(obj.time, movmean(v1,3), 'LineWidth',1.5)
%      % hold on
%     % plot(obj.time, movmean(v2,3), 'LineWidth',1.5)
%     %     hold on
%     % plot(obj.time, movmean(v3,3), 'LineWidth',1.5)
% 
%     xlim([-1 2])
% xline(0)
% yline(0)
% xline(obj.time(rang(1)),'LineWidth',2,'Color','r')
% xline(obj.time(rang(end)),'LineWidth',2,'Color','r')
% 
%     pause
% 
% end


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
% rang = [480:520];
% % rang = [200:250];
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
% rang = [450:500];
% % rang = [200:250];
% 
% % FLMode = mean(FL_dat(rang,:)) - mean(FL_dat(100:300,:));
% FLMode = mean(FL_dat(rang,:)) - mean(FL_dat(50:180,:));
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

% 
% figure; 
% % subplot(1,2,1)
% trials = params.trialid{1, 1};
% 
% cmp = jet;
% L = Length;
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
% % subplot(1,2,2)
% % trials = params.trialid{1, 9};
% % 
% % L = Length(:, trials);
% % 
% % imagesc(obj.time, [1:size(L,2)], L');
% % xline(0,'LineWidth', 1.5)
% % colormap(cmp)
% % %     caxis([rang]);
% % colorbar; grid off;
% % % axis tight; axis square;
% % box off; xlabel('time(s)'); ylabel('Trial #'); title([kinfeat,' plot'], 'Interpreter', 'none');
% % set(gca,'FontSize',15)
% % xlim([-0.5 2])
% 
% 
% 
% px = 500;
% py = 500;
% width = 800;
% height = 250;
% set(gcf, 'Position', [px, py, width, height]); % Set figure position and size


%%

Ncells = size(obj.psth, 2);
clu_m1TJ = 1:size(params.cluid,1);
clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;

reg = clu_ALM; % P2
reg1 = clu_m1TJ;% P1

Ncells = size(obj.psth, 2);
clu_m1TJ = 1:size(params.cluid{1, 1},1);
clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;

reg = clu_ALM; % P2
reg1 = clu_m1TJ;% P1

neurons = reg1;

sz = 10;
NT = obj.bp.Ntrials;
lw = 2;
ct = 1;
n = 30;

col = {[1 0 0] [0 0 1] [0.5 0 0] [0 0 0.5]};

% 2 is LR1, 4 RR1, 5 is LR16, 7 is RR16


for i = 1:n:length(neurons)
    f = figure;
    for j = 1:n

        try
        ax = nexttile;


%         plot(obj.time,obj.psth(:,neurons(ct),2),'Color',col{1},'LineWidth',lw); hold on 
%         plot(obj.time,obj.psth(:,neurons(ct),4),'Color',col{2},'LineWidth',lw); hold on 
%         plot(obj.time,obj.psth(:,neurons(ct),5),'Color',col{3},'LineWidth',lw); hold on 
%         plot(obj.time,obj.psth(:,neurons(ct),7),'Color',col{4},'LineWidth',lw); hold on 

        plot(obj.time,movmean(obj.psth(:,neurons(ct),8),10),'Color',col{1},'LineWidth',lw); hold on
        plot(obj.time,movmean(obj.psth(:,neurons(ct),9),10),'Color',col{2},'LineWidth',lw); hold on 

        % plot(obj.time,movmean(obj.psth(:,neurons(ct),5),1),'Color',col{1},'LineWidth',lw); hold on
        % plot(obj.time,movmean(obj.psth(:,neurons(ct),6),1),'Color',col{2},'LineWidth',lw); hold on 
        % plot(obj.time,movmean(obj.psth(:,neurons(ct),7),1),'Color',[0 1 0],'LineWidth',lw); hold on 


%         legend('LR1','LR16','RR1','RR16')
%         plot(objs.time,objs.psth(:,neurons(ct),3),'Color','b','LineWidth',1.5); hold on 


title(['Cell = ', num2str(neurons(ct))]);
xlabel(['time from ', num2str(params.alignEvent)]);
% xlabel('time')
xline(-2)
% xline(0)
% xline(0.25)
xline(-0.4)
xline(0.05)

xlim([-0.5 1.5]);
% xlim([-0.4 0.05]);
set(gca,'FontSize',sz)
% axis square

ylabel('firing rate')
box off
        ct = ct + 1;
    end
end
end
% 
a = 2;
%%

Ncells = size(obj.psth, 2);
clu_m1TJ = 1:size(params.cluid,1);
clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;

reg = clu_ALM; % P2
reg1 = clu_m1TJ;% P1

Ncells = size(obj.psth, 2);
clu_m1TJ = 1:size(params.cluid{1, 1},1);
clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;

reg = clu_ALM; % P2
reg1 = clu_m1TJ;% P1

neurons = reg;

sz = 10;
NT = obj.bp.Ntrials;
lw = 2;
ct = 1;
n = 30;

col = {[1 0 0] [0 0 1] [0.5 0 0] [0 0 0.5]};

% 2 is LR1, 4 RR1, 5 is LR16, 7 is RR16
tnew = obj.trialdat(:,:,:);

left_trials = obj.bp.trialTypes == 1;
t_left = tnew(:,:,left_trials);

tdat_reshaped = reshape(t_left, [], size(t_left, 2)); % size: [time * trials, neurons]

mean_tdat = mean(tdat_reshaped, 1); % Mean across (time * trials), size: [1, neurons]
std_tdat = std(tdat_reshaped, 0, 1); % Standard deviation across (time * trials), size: [1, neurons]

tdat_zscored = (tdat_reshaped - mean_tdat) ./ std_tdat;
% tdat_zscored = (tdat_reshaped - mean_tdat) ;
tdat_zscored = reshape(tdat_zscored, size(t_left, 1), size(t_left, 2), size(t_left, 3));
t_left = tdat_zscored;


center_trials = obj.bp.trialTypes == 2;
t_center = tnew(:,:,center_trials);

tdat_reshaped = reshape(t_center, [], size(t_center, 2)); % size: [time * trials, neurons]

mean_tdat = mean(tdat_reshaped, 1); % Mean across (time * trials), size: [1, neurons]
std_tdat = std(tdat_reshaped, 0, 1); % Standard deviation across (time * trials), size: [1, neurons]

tdat_zscored = (tdat_reshaped - mean_tdat) ./ std_tdat;
% tdat_zscored = (tdat_reshaped - mean_tdat) ;
tdat_zscored = reshape(tdat_zscored, size(t_center, 1), size(t_center, 2), size(t_center, 3));
t_center = tdat_zscored;

right_trials = obj.bp.trialTypes == 3;
t_right = tnew(:,:,right_trials);

tdat_reshaped = reshape(t_right, [], size(t_right, 2)); % size: [time * trials, neurons]

mean_tdat = mean(tdat_reshaped, 1); % Mean across (time * trials), size: [1, neurons]
std_tdat = std(tdat_reshaped, 0, 1); % Standard deviation across (time * trials), size: [1, neurons]

tdat_zscored = (tdat_reshaped - mean_tdat) ./ std_tdat;
% tdat_zscored = (tdat_reshaped - mean_tdat) ;
tdat_zscored = reshape(tdat_zscored, size(t_right, 1), size(t_right, 2), size(t_right, 3));
t_right = tdat_zscored;


tnew(:,:,left_trials) = t_left;
tnew(:,:,center_trials) = t_center;
tnew(:,:,right_trials) = t_right;
% 
for i = 1:n:length(neurons)
    f = figure;
    for j = 1:n

        try
        ax = nexttile;


%         plot(obj.time,obj.psth(:,neurons(ct),2),'Color',col{1},'LineWidth',lw); hold on 
%         plot(obj.time,obj.psth(:,neurons(ct),4),'Color',col{2},'LineWidth',lw); hold on 
%         plot(obj.time,obj.psth(:,neurons(ct),5),'Color',col{3},'LineWidth',lw); hold on 
%         plot(obj.time,obj.psth(:,neurons(ct),7),'Color',col{4},'LineWidth',lw); hold on 

        % plot(obj.time,movmean(obj.psth(:,neurons(ct),8),1),'Color',col{1},'LineWidth',lw); hold on
        % plot(obj.time,movmean(obj.psth(:,neurons(ct),9),1),'Color',col{2},'LineWidth',lw); hold on 
r1 = find(obj.bp.rewardedLick == 1);
r4 = find(obj.bp.rewardedLick == 4);
allT = 1:length(obj.bp.rewardedLick);


        left_trials = intersect(allT(left_trials),r1);
        center_trials = intersect(allT(center_trials),r1);
        right_trials = intersect(allT(right_trials),r1);

        % left_trials = intersect(allT(left_trials),r4);
        % center_trials = intersect(allT(center_trials),r4);
        % right_trials = intersect(allT(right_trials),r4);

        a1 = tnew(:,neurons(ct),left_trials);
        a2 = tnew(:,neurons(ct),center_trials);
        a3 = tnew(:,neurons(ct),right_trials);




        plot(obj.time,movmean(mean(squeeze(a1),2),15),'Color',col{1},'LineWidth',lw); hold on
        plot(obj.time,movmean(mean(squeeze(a2),2),15),'Color',col{2},'LineWidth',lw); hold on
        plot(obj.time,movmean(mean(squeeze(a3),2),15),'Color',[0 1 0],'LineWidth',lw); hold on


%         legend('LR1','LR16','RR1','RR16')
%         plot(objs.time,objs.psth(:,neurons(ct),3),'Color','b','LineWidth',1.5); hold on 


title(['Cell = ', num2str(neurons(ct))]);
xlabel(['time from ', num2str(params.alignEvent)]);
% xlabel('time')
xline(-2)
% xline(0)
% xline(0.25)
xline(-0.4)
xline(0.05)

xlim([-0.5 1.5]);
% xlim([-0.4 0.05]);
set(gca,'FontSize',sz)
% axis square

ylabel('firing rate')
box off
        ct = ct + 1;
    end
end
end
% 
a = 2;


%%
% mval = {};
% 
% Ncells = size(obj.psth, 2);
% clu_m1TJ = 1:size(params.cluid{1, 1},1);
% clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;
% 
% reg = clu_ALM; % P2
% reg1 = clu_m1TJ;% P1
% 
% neurons = reg1;
% 
% sz = 10;
% NT = obj.bp.Ntrials;
% lw = 2;
% ct = 1;
% n = 30;
% 
% col = {[1 0 0] [0 0 1] [0.5 0 0] [0 0 0.5]};
% 
% % 2 is LR1, 4 RR1, 5 is LR16, 7 is RR16
% tnew = obj.trialdat(:,neurons,:);
% 
% for kk = 1:length(clu_m1TJ)
% 
% tnew = obj.trialdat(:,clu_m1TJ(kk),:);
% t = squeeze(tnew); t = t(500:700,:);
% 
% meanFR = mean(t,1);
% a1 = obj.bp.trialTypes == 1;
% a2 = obj.bp.trialTypes == 2;
% a3 = obj.bp.trialTypes == 3;
% a11 = obj.bp.hit == 1; a11 = a11';
% a22 = obj.bp.hit == 1; a22 = a22';
% a33 = obj.bp.hit == 1; a33 = a33';
% 
% a1 = a1 & a11;
% a2 = a2 & a22;
% a3 = a3 & a33;
% 
% 
% mval1{kk} = meanFR(1,a1)';
% mval2{kk} = meanFR(1,a2)';
% mval3{kk} = meanFR(1,a3)';
% 
% end
% 
% % ax = figure;
% 
% for i = 1:numel(mval1)
% 
%     % cla(ax)
% figure;
% plot(movmean(mval1{1, i},15),'LineWidth',2)  
% hold on 
% plot(movmean(mval2{1, i},15),'LineWidth',2)    
% hold on
% plot(movmean(mval3{1, i},15),'LineWidth',2)    
% 
%     % pause
% 
% end
% 
% a = 5;
%% DECODING PARAMETERS

figure; 
% tLength = kin.dat(:,:,3); % (time,trials,feats)
tLength = kin.dat(:,:,54); % (time,trials,feats)
imagesc(obj.time, [1:size(tLength,2)], tLength');
xline(0,'LineWidth', 1.5)
colormap(jet)
%     caxis([rang]);
colorbar; grid off; axis tight; axis square;
box off; xlabel('time(s)'); ylabel('Trial #'); 
set(gca,'FontSize',15)
% 
% figure; 
% tAngle = kin.dat(:,:,53); % (time,trials,feats)
% imagesc(obj.time, [1:size(tLength,2)], tAngle', 'Interpolation', 'bilinear');
% xline(0,'LineWidth', 1.5)
% colormap(jet)
% %     caxis([rang]);
% colorbar; grid off; axis tight; axis square;
% box off; xlabel('time(s)'); ylabel('Trial #'); 
% set(gca,'FontSize',15)


figure; 
tAngle = kin.dat(:,:,3); % (time,trials,feats)
imagesc(obj.time, [1:size(tLength,2)], tAngle', 'Interpolation', 'bilinear');
xline(0,'LineWidth', 1.5)
colormap(jet)
%     caxis([rang]);
colorbar; grid off; axis tight; axis square;
box off; xlabel('time(s)'); ylabel('Trial #'); 
set(gca,'FontSize',15)
%% FOR VTA


for i = 1:obj.bp.Ntrials



    nlicks = obj.bp.ev.lickL{i};
    gc = obj.bp.ev.goCue(i);

    if ~isempty(nlicks)

        lickspost = nlicks - gc;
    num_licks_postGC(i) = numel(find(lickspost < 0.7));
    else

    num_licks_postGC(i) = 0;
    end

end


trials = find(num_licks_postGC > 3);
trials1 = find(num_licks_postGC < 4);

%%
sz = 15;

yAll = [];
yHat = [];
yall_1 = [];
yall_2 = [];
yall_3 = [];
yall_4 = [];
r2 = [];

for aa = 1:1
% input data = neural data (time*trials,neurons)
% output data = kin data   (time*trials,kin feats)

par.pre= 12; % time bins prior to output used for decoding
par.post= 12; % time bins after output used for decoding
par.dt = params(1).dt; % moving time bin
par.pre_s = par.pre .* params(1).dt; % dt, pre_s, and post_s just here to know how much time you're using. Set params.dt and pre/post appropriately for you analysis
par.post_s = par.post .* params(1).dt;

par.train = 0.7; % fraction of trials
par.test = 1 - par.train;
par.validation = 1 - par.train - par.test;

% feature to decode
% par.feats = {'tongue_angle'};
% par.feats = {'tongue_yvel_view1'};
% par.feats = {'motion_energy'};
% par.feats = {'tongue_length'};
par.feats = {'jaw_ydisp_view1'};
% par.feats = {'jaw_yvel_view1'};

% trials
% par.cond2use = [9];
par.cond2use = [8];

par.regularize = 1; % if 0, linear regression. if 1, ridge regression

% DECODING
% % Ncells = size(obj.psth, 2);
% clu_m1TJ = 1:size(params.cluid,1);
% % clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;
% 
% % reg = clu_ALM; % P2
% clu = clu_m1TJ;% P1

Ncells = size(obj.psth, 2);
clu_m1TJ = 1:size(params.cluid{1, 1},1);
clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;

% Ncells = size(obj.psth, 2);
% clu_m1TJ = 1:length(params.cluid);
% clu_ALM = size(params.cluid{1, 1},1)+1:Ncells;

reg = clu_ALM;
reg1 = clu_m1TJ;

clu = clu_m1TJ;

% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%% SELECT TRIALS + PARTITION %%%%%%%%%%%%%%%%%%%%%%%%
% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------

par.trials.all = cell2mat(params(1).trialid(par.cond2use)');
par.trials.all1 = cell2mat(params(1).trialid(8)');
par.trials.all4 = cell2mat(params(1).trialid(9)');
% par.trials.all4 = par.trials.all4(find(par.trials.all4 < 279));

% in_trials = find(obj.bp.trialTypes == 3);
par.trials.all = intersect(par.trials.all, trials);

nTrials = numel(par.trials.all);
nTrain = floor(nTrials*par.train);
par.trials.train = randsample(par.trials.all,nTrain,false);
par.trials.test = par.trials.all(~ismember(par.trials.all,par.trials.train));
par.trials.test1 = par.trials.all1(~ismember(par.trials.all1,par.trials.train));
par.trials.test4 = par.trials.all4(~ismember(par.trials.all4,par.trials.train));
% omitTrials = obj.bp.omitTrial;indices = find(cell2mat(omitTrials) == 1);

% rang = [420:520];
% rang = [300:820];
% rang = [430:900];

% This is the one we were using for engaged state
rang = [410:515];

% This is the one I am testing for disengaged state
% rang = [1000:1100];  %2.5s-3.0s after gc
% rang = [900:1000]  %2-2.5s


% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%% SELECT INPUT (NEURAL) DATA %%%%%%%%%%%%%%%%%%%%%%%%
% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------

% Input data
trang = 10:430;
tdat = obj(1).trialdat;
tdat = reshape((reshape(tdat, [], size(tdat, 2)) - mean(reshape(tdat(trang, :, :), [], size(tdat, 2)), 1)) ...
    ./ std(reshape(tdat(trang, :, :), [], size(tdat, 2)), 0, 1), size(tdat));

X.train = tdat(rang,clu,par.trials.train); % (time,clu,trials)
X.train = permute(X.train,[1 3 2]);
X.train = reshape(X.train, size(X.train,1)*size(X.train,2),size(X.train,3));

X.test = tdat(:,clu,par.trials.test);
X.test = permute(X.test,[1 3 2]);
X.size = size(X.test);
X.test = reshape(X.test, size(X.test,1)*size(X.test,2),size(X.test,3));

X.test1 = tdat(:,clu,par.trials.test1);
X.test1 = permute(X.test1,[1 3 2]);
X.size1 = size(X.test1);
X.test1 = reshape(X.test1, size(X.test1,1)*size(X.test1,2),size(X.test1,3));

X.test4 = tdat(:,clu,par.trials.test4);
X.test4 = permute(X.test4,[1 3 2]);
X.size4 = size(X.test4);
X.test4 = reshape(X.test4, size(X.test4,1)*size(X.test4,2),size(X.test4,3));

% reshape train and test data to account for prediction bin size
X.train = reshapePredictors(X.train,par);
X.test = reshapePredictors(X.test,par);
X.test1 = reshapePredictors(X.test1,par);
X.test4 = reshapePredictors(X.test4,par);

% flatten inputs
% if you're using a model with recurrence, don't flatten
X.train = reshape(X.train,size(X.train,1),size(X.train,2)*size(X.train,3));
X.test = reshape(X.test,size(X.test,1),size(X.test,2)*size(X.test,3));
X.test1 = reshape(X.test1,size(X.test1,1),size(X.test1,2)*size(X.test1,3));
X.test4 = reshape(X.test4,size(X.test4,1),size(X.test4,2)*size(X.test4,3));

% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%% SELECT OUTPUT (NEURAL) DATA %%%%%%%%%%%%%%%%%%%%%%%%
% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------


% output data
par.featix = find(ismember(kin(1).featLeg,par.feats));


kdat = kin(1).dat(:,:,par.featix);

if strcmp(par.feats{1}, 'tongue_length') 

% kdat(isnan(kdat)) = 0; % Replace NaNs with zero
% kdat = reshape(normalize(kdat(:) - mean(kdat(:)), 'range') - mean(normalize(kdat(:) - mean(kdat(:)), 'range')), size(kdat));

save1 = kdat(:,par.trials.test1); % (time,trials,feats)
save4 = kdat(:,par.trials.test4); % (time,trials,feats)

% Saving tongue info for heatmap plotting
save_path = fullfile(save_folder, 'R1_Tongue.csv');
writematrix(save1, save_path);

save_path = fullfile(save_folder, 'R4_Tongue.csv');
writematrix(save4, save_path);

kdat(isnan(kdat)) = 0; % Replace NaNs with zero
kdat = reshape(normalize(kdat(:) - mean(kdat(:)), 'range') - mean(normalize(kdat(:) - mean(kdat(:)), 'range')), size(kdat));
end

if strcmp(par.feats{1}, 'tongue_yvel_view1') || strcmp(par.feats{1}, 'jaw_yvel_view1')

kdat(isnan(kdat)) = 0; % Replace NaNs with zero
kdat = reshape(normalize(kdat(:) - mean(kdat(:)), 'range',[-1 1]) - mean(normalize(kdat(:) - mean(kdat(:)), 'range',[-1 1])), size(kdat));

end

if strcmp(par.feats{1}, 'jaw_ydisp_view1') 

    % figure;
    % imagesc(kdat')
kdat(isnan(kdat)) = 0; % Replace NaNs with zero
kdat = reshape(normalize(kdat(:) - mean(kdat(:)), 'range') - mean(normalize(kdat(:) - mean(kdat(:)), 'range')), size(kdat));
% hold on 
% imagesc(kdat')

end

if strcmp(par.feats{1}, 'tongue_angle') 

    % kdat_flat = kdat(:);  % Flattening into a 1x(1000*111) vector
    % 
    % % Normalize the flattened vector between 0 and 1
    % kdat_normalized_flat = normalize(kdat_flat,'range',[-1 1]);
    % 
    % % Reshape the normalized vector back to the original dimensions [1000 x 111]
    % kdat_normalized = reshape(kdat_normalized_flat, size(kdat));
    % 
    % kdat = kdat_normalized;

% Flatten the kdat matrix into a single vector
kdat_flat = kdat(:);

% Remove NaN values for normalization
kdat_flat_no_nan = kdat_flat(~isnan(kdat_flat));

% Find the global min and max values (excluding NaNs)
min_kdat = min(kdat_flat_no_nan);
max_kdat = max(kdat_flat_no_nan);

% Normalize the data to the range [-1, 1]
kdat_normalized = 2 * ((kdat - min_kdat) / (max_kdat - min_kdat)) - 1;

% Mean-center the data (subtract the overall mean of the data)
kdat_normalized_centered = kdat_normalized - mean(kdat_normalized(:), 'omitnan');

% Reshape the normalized, mean-centered data back to the original size of kdat
kdat_normalized_centered = reshape(kdat_normalized_centered, size(kdat));
kdat = kdat_normalized_centered;

end

% figure;
% imagesc(kdat')


Y.train = kdat([rang],par.trials.train); % (time,trials,feats)
Y.train = reshape(Y.train, size(Y.train,1)*size(Y.train,2),size(Y.train,3));

Y.test = kdat(:,par.trials.test); % (time,trials,feats)
Y.test1 = kdat(:,par.trials.test1); % (time,trials,feats)
Y.test4 = kdat(:,par.trials.test4); % (time,trials,feats)

Y.size = size(Y.test);
Y.size1 = size(Y.test1);
Y.size4 = size(Y.test4);

Y.test = reshape(Y.test, size(Y.test,1)*size(Y.test,2),size(Y.test,3));
Y.test1 = reshape(Y.test1, size(Y.test1,1)*size(Y.test1,2),size(Y.test1,3));
Y.test4 = reshape(Y.test4, size(Y.test4,1)*size(Y.test4,2),size(Y.test4,3));


% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%% DECODE %%%%%%%%%%%%%%%%%%%%%%%%
% -----------------------------------------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------------------------------------

if ~par.regularize
    % use an unregularized linear regression model
    mdl = fitlm(X.train,Y.train);
%     mdl = fitrlinear(X.train,Y.train, 'Regularization','ridge');
    pred = predict(mdl,X.test);
    pred(isnan(Y.test)) = 0; % kinematic data is messy, so there's always some nans. let's make sure the prediction only contains non-nan values where we had kinemtic data

else

    ct = 1;
    % kridges = logspace(-3,5,20); % shoudlnt really go over 3
    % kridges = logspace(-1,5,50); % shoudlnt really go over 3
    kridges = logspace(2,6,50); % shoudlnt really go over 3
    % kridges = logspace(3,5,20); % shoudlnt really go over 3

    for k = kridges
        B = ridge(Y.train,X.train,k);
        % [B, FitInfo] = lasso(X.train, Y.train, 'Lambda', k);
        pred = X.test*B;
        pred(isnan(Y.test)) = nan;

nTrials_test = numel(par.trials.test); % Number of test trials
rang1 = rang;
% rang1 = [500:900];
t = 1:numel(obj.time);
idx_rang = arrayfun(@(t) rang1 + (t-1) * numel(obj.time), 1:nTrials_test, 'UniformOutput', false);
idx_rang = cell2mat(idx_rang); % Convert to usable indices

% Use the selected time points for computing r2
Y_test_rang = Y.test(idx_rang, :);
pred_rang = pred(idx_rang, :);

% figure;
% subplot(2,1,1)
% plot(Y_test_rang)
% hold on 
% plot(pred_rang)


SStot = nansum((Y_test_rang - nanmean(Y_test_rang, 1)).^2);
SSres = nansum((Y_test_rang - pred_rang).^2);
r2(ct) = 1 - (SSres ./ SStot);

modeValue = mode(rmmissing(Y_test_rang));
indices = find(Y_test_rang ~= modeValue); % Find indices where values are NOT equal to the mode
% % 
test_vals = Y_test_rang(indices);
pred_vals = pred_rang(indices);

% subplot(2,1,2)
% plot(test_vals)
% hold on 
% plot(pred_vals)

SStot = nansum((test_vals - nanmean(test_vals, 1)).^2);
SSres = nansum((test_vals - pred_vals).^2);
r2(ct) = 1 - SSres ./ SStot;

% 
Yt = Y_test_rang;
predt = pred_rang;

Yt(isnan(Y_test_rang)) = [];
predt(isnan(pred_rang)) = [];

r = corrcoef(Yt, predt);
% r2(ct) = r(2)^2;

% figure;
% subplot(2,1,1)
% plot(Y_test_rang)
% hold on
% plot(pred_rang)
% 
% subplot(2,1,2)
% plot(test_vals)
% hold on
% plot(pred_vals)

a = 2;

        ct = ct + 1
    end

    % 
    [~,ix] = max(r2);
    ix
    kridge = kridges(ix);

% kridge = 0;

    B = ridge(Y.train,X.train,kridge);
    % [B, FitInfo] = lasso(X.train, Y.train, 'Lambda', kridge);
    pred = X.test*B;
    pred(isnan(Y.test)) = nan;

    pred1 = X.test1*B;
    pred1(isnan(Y.test1)) = nan;

    pred4 = X.test4*B;
    pred4(isnan(Y.test4)) = nan;

    % B = ridge(Y.train,X.train,0);
    % pred = X.test*B;
    % pred(isnan(Y.test)) = nan;

end

a = 8;
%
% figure;
% plot(Y.test)
% hold on 
% plot(pred)
% figure;

%
% 
% figure;
% subplot(2,1,1)
% hold on;
% allpre = []; allpost = [];
% numNeurons = numel(B) / (par.pre + par.post ); % 46 neurons
% binwin = par.pre + par.post; % 45 bins per neuron
% 
% for neuron = 1:numNeurons
%     idx_start = (neuron - 1) * binwin + 1;
%     idx_pre = idx_start : idx_start + par.pre - 1;
%     idx_post = idx_start + par.pre : idx_start + binwin - 1;
% 
%     % Plot pre bins in red
%     plot(idx_pre, abs(B(idx_pre)), 'r','LineWidth',2);
% 
%     % Plot post bins in blue
%     plot(idx_post, abs(B(idx_post)), 'b','LineWidth',2);
% 
%     xline(idx_start, '-', 'LineWidth', 1);
% 
% 
%     del = par.pre + par.post;
% 
%     neuronStarts(neuron) = idx_start;
% 
%     allpre = [allpre abs(B(idx_pre))'];
%     allpost = [allpost abs(B(idx_post))'];
% end
% 
% xticks(neuronStarts(1:end));  % X-tick positions at neuron start points
% xticklabels(1:numNeurons);  % Labels from 1 to numNeurons
% 
% 
% 
% % Add X-ticks at the start of each neuron
% 
% 
% hold off;
% xlabel('Predictor Index');
% ylabel('Value of B');
% title('Color-Coded Pre (Red) and Post (Blue) Bins');
% yline(0)
% subplot(2,1,2)
% plot(1,mean(allpre),'o','Color','r','MarkerSize',20);
% hold on 
% plot(1, mean(allpost),'o','Color','b','MarkerSize',20);
% 
% px = 500;
% py = 500;
% width = 1000;
% height = 400;
% set(gcf, 'Position', [px, py, width, height]); % Set figure position and size
% 
% figure;
% subplot(2,1,1)
% plot(kridges, r2,'.')
% subplot(2,1,2)
% plot(pred,Y.test,'.')

% SStot = nansum((Y.test-mean(Y.test,1,'omitmissing')).^2);     % Total Sum-Of-Squares
% SSres = nansum((Y.test-pred).^2);                             % Residual Sum-Of-Squares
% r2 = 1-SSres/SStot;

% hold on 
% plot(Y.test);

y = reshape(Y.test,Y.size(1),Y.size(2)); % original input data (centered)
y1 = reshape(Y.test1,Y.size1(1),Y.size1(2)); % original input data (centered)
y4 = reshape(Y.test4,Y.size4(1),Y.size4(2)); % original input data (centered)
yhat = reshape(pred,Y.size(1),Y.size(2)); % prediction
yhat1 = reshape(pred1,Y.size1(1),Y.size1(2)); % prediction
yhat4 = reshape(pred4,Y.size4(1),Y.size4(2)); % prediction


% if strcmp(par.feats{1}, 'tongue_angle') 
% yhat = yhat - 0.5;
% end

% Plotting

alph = 0.5;

if strcmp(par.feats{1}, 'jaw_ydisp_view1') || strcmp(par.feats{1}, 'tongue_length') || strcmp(par.feats{1}, 'tongue_yvel_view1') || strcmp(par.feats{1}, 'jaw_yvel_view1')
% if strcmp(par.feats{1}, 'tongue_length') 

    figure;

% subtrials = par.trials.test;
% subplot(1,3,1)
%%% Length %%%
tol = 1e-9;
Kinematics1 = y; 
last_nonzero_index = max((1:size(Kinematics1,1)).' .* (abs(Kinematics1)),[],1);
[~,idx1] = sort(last_nonzero_index);
last_nonzero_index = last_nonzero_index(idx1);
y = y(:,idx1);
yhat = yhat(:, idx1);

ax = nexttile;
hold on;
mu.y = nanmean(y(:,:),2);
mu.yhat = nanmean(yhat(:,:),2);
sd.y = nanstd(y,[],2) ./ sqrt(size(y,2));
sd.yhat = nanstd(yhat(:,:),[],2) ./ sqrt(size(yhat(:,:),2));
shadedErrorBar(obj.time,mu.y,sd.y,{'Color','k','LineWidth',3},alph,ax)
shadedErrorBar(obj.time,mu.yhat,sd.yhat,{'Color',[1 0 0 0.5],'LineWidth',2},alph,ax)
ylabel('tongue length')
xlabel(['time from ', num2str(params.alignEvent)]);
set(gca,'FontSize',sz)
axis tight
xlim([-0.6 1.75])

xline(obj.time(rang(1)),'LineWidth',2,'Color','b')
xline(obj.time(rang(end)),'LineWidth',2,'Color','b')

figure;

if par.cond2use == 9
yd = y1;
ydh = yhat1;
else
    yd = y4;
    ydh = yhat4;
end

tol = 1e-9;
Kinematics1 = yd; 
last_nonzero_index = max((1:size(Kinematics1,1)).' .* (abs(Kinematics1)),[],1);
[~,idx1] = sort(last_nonzero_index);
last_nonzero_index = last_nonzero_index(idx1);
yd = yd(:,idx1);
ydh = ydh(:, idx1);

ax = nexttile;
hold on;
mu.y = nanmean(yd(:,:),2);
mu.yhat = nanmean(ydh(:,:),2);
sd.y = nanstd(yd,[],2) ./ sqrt(size(yd,2));
sd.yhat = nanstd(ydh(:,:),[],2) ./ sqrt(size(yhat(:,:),2));
shadedErrorBar(obj.time,mu.y,sd.y,{'Color','k','LineWidth',3},alph,ax)
shadedErrorBar(obj.time,mu.yhat,sd.yhat,{'Color',[1 0 0 0.5],'LineWidth',2},alph,ax)
ylabel('tongue length')
xlabel(['time from ', num2str(params.alignEvent)]);
set(gca,'FontSize',sz)
axis tight
xlim([-0.6 1.75])

xline(obj.time(rang(1)),'LineWidth',2,'Color','b')
xline(obj.time(rang(end)),'LineWidth',2,'Color','b')








% subtrials = t2;
%     figure;
% %%% Length %%%
% tol = 1e-9;
% Kinematics1 = y; 
% last_nonzero_index = max((1:size(Kinematics1,1)).' .* (abs(Kinematics1)),[],1);
% [~,idx1] = sort(last_nonzero_index);
% last_nonzero_index = last_nonzero_index(idx1);
% y = y(:,idx1);
% yhat = yhat(:, idx1);
% 
% 
% ax = nexttile;
% hold on;
% mu.y = nanmean(y(:,subtrials),2);
% mu.yhat = nanmedian(yhat(:,subtrials),2);
% sd.y = nanstd(y,[],2) ./ sqrt(size(y,2));
% sd.yhat = nanstd(yhat(:,subtrials),[],2) ./ sqrt(size(yhat(:,subtrials),2));
% shadedErrorBar(obj.time,mu.y,sd.y,{'Color','k','LineWidth',3},alph,ax)
% shadedErrorBar(obj.time,mu.yhat,sd.yhat,{'Color',[1 0 0 0.5],'LineWidth',2},alph,ax)
% % ylabel(par.feats,'Interpreter','none')
% ylabel('tongue length')
% xlabel(['time from ', num2str(params.alignEvent)]);
% set(gca,'FontSize',sz)
% axis tight
% xlim([-1 3.5])


% subtrials = t3;
%     figure;
% %%% Length %%%
% tol = 1e-9;
% Kinematics1 = y; 
% last_nonzero_index = max((1:size(Kinematics1,1)).' .* (abs(Kinematics1)),[],1);
% [~,idx1] = sort(last_nonzero_index);
% last_nonzero_index = last_nonzero_index(idx1);
% y = y(:,idx1);
% yhat = yhat(:, idx1);
% 
% 
% ax = nexttile;
% hold on;
% mu.y = nanmean(y(:,subtrials),2);
% mu.yhat = nanmedian(yhat(:,subtrials),2);
% sd.y = nanstd(y,[],2) ./ sqrt(size(y,2));
% sd.yhat = nanstd(yhat(:,subtrials),[],2) ./ sqrt(size(yhat(:,subtrials),2));
% shadedErrorBar(obj.time,mu.y,sd.y,{'Color','k','LineWidth',3},alph,ax)
% shadedErrorBar(obj.time,mu.yhat,sd.yhat,{'Color',[1 0 0 0.5],'LineWidth',2},alph,ax)
% % ylabel(par.feats,'Interpreter','none')
% ylabel('tongue length')
% xlabel(['time from ', num2str(params.alignEvent)]);
% set(gca,'FontSize',sz)
% axis tight
% xlim([-1 3.5])

% title('R4')

yAll = [yAll mu.y];
yHat = [yHat mu.yhat];

a = 5;

elseif strcmp(par.feats{1}, 'tongue_angle') 
    
    if par.cond2use == 9
P1 = find(ismember(par.trials.test, params.trialid{5}));
P2 = find(ismember(par.trials.test, params.trialid{6}));
P3 = find(ismember(par.trials.test, params.trialid{7}));

    else
P1 = find(ismember(par.trials.test, params.trialid{2}));
P2 = find(ismember(par.trials.test, params.trialid{3}));
P3 = find(ismember(par.trials.test, params.trialid{4}));

    end


gc = obj.bp.ev.goCue(par.trials.test);
minLim = 234;

arr1 = nanmean(y([minLim:end],P1),2);
arr2 = nanmean(yhat([minLim:end],P1),2);
arr3 = nanmean(y([minLim:end],P2),2);
arr4 = nanmean(yhat([minLim:end],P2),2);
arr5 = nanmean(y([minLim:end],P3),2);
arr6 = nanmean(yhat([minLim:end],P3),2);


% figure;
% subplot(2,1,1)
% plot(arr1); hold on;
% plot(arr2); hold on;
% % % plot(arr3); hold on;
% % % plot(arr4); hold on;
% plot(arr5); hold on;
% plot(arr6); hold on;

% val = arr1;
% realIndices = find(~isnan(val) );
% consecutiveSets = findConsecutiveSets(realIndices, 4, 20);
% for i = 1:length(consecutiveSets)
% temp = consecutiveSets{1,i};
% y1(i)=median(val(temp));
% end
% 
% val = arr2;
% realIndices = find(~isnan(val));
% consecutiveSets = findConsecutiveSets(realIndices, 4, 20);
% for i = 1:length(consecutiveSets)
% temp = consecutiveSets{1,i};
% y2(i)=median(val(temp));
% end
% 
% val = arr5;
% realIndices = find(~isnan(val) );
% consecutiveSets = findConsecutiveSets(realIndices, 4, 20);
% for i = 1:length(consecutiveSets)
% temp = consecutiveSets{1,i};
% y3(i)=median(val(temp));
% end
% 
% val = arr6;
% realIndices = find(~isnan(val));
% consecutiveSets = findConsecutiveSets(realIndices, 4, 20);
% for i = 1:length(consecutiveSets)
% temp = consecutiveSets{1,i};
% y4(i)=median(val(temp));
% end

%%%%%%%%%%%

% figure; 
% plot(mu.y)
% hold on 
% plot(movmean(mu.y,12, 'omitnan'))


%%%%%%%%%%%


winSize = 3;

figure;
% subplot(2,1,1)
ax = nexttile;
% ax = subplot(1,2,1);
% ax = subplot(1,1,1);
hold(ax, 'on');
c = P1;
hold on;
mu.y = nanmean(y(:,c),2);
mu.yhat = nanmean(yhat(:,c),2);
sd.y = nanstd(y(:,c),[],2) ./ sqrt(size(y(:,c),2));
sd.yhat = nanstd(yhat(:,c),[],2) ./ sqrt(size(yhat(:,c),2));
plot(obj.time,movmean(mu.y,winSize, 'omitnan'),'Color',[1 0 0 1],'LineWidth',4)
plot(obj.time,movmean(mu.yhat,winSize, 'omitnan'),'Color',[1 0 0 0.4 ],'LineWidth',4)

ylabel('tongue angle')
xlabel(['time from ', num2str(params.alignEvent)]);
set(gca,'FontSize',sz)
ylim([-1.5 1.5])
xlim([-0.6 0.8])

hold(ax, 'on');
c = P2;
hold on;
mu.y = nanmean(y(:,c),2);
mu.yhat = nanmean(yhat(:,c),2);
sd.y = nanstd(y(:,c),[],2) ./ sqrt(size(y(:,c),2));
sd.yhat = nanstd(yhat(:,c),[],2) ./ sqrt(size(yhat(:,c),2));
plot(obj.time,movmean(mu.y,winSize, 'omitnan'),'Color',[0 0 0 1],'LineWidth',4)
plot(obj.time,movmean(mu.yhat,winSize, 'omitnan'),'Color',[0 0 0 0.4],'LineWidth',4)

% ax = nexttile;
% ax = subplot(1,1,1);
hold(ax, 'on');
c = P3;
hold on;
mu.y = nanmean(y(:,c),2);
mu.yhat = nanmean(yhat(:,c),2);
sd.y = nanstd(y(:,c),[],2) ./ sqrt(size(y(:,c),2));
sd.yhat = nanstd(yhat(:,c),[],2) ./ sqrt(size(yhat(:,c),2));
plot(obj.time,movmean(mu.y,winSize, 'omitnan'),'Color',[0 0 1 1],'LineWidth',4)
plot(obj.time,movmean(mu.yhat,winSize, 'omitnan'),'Color',[0 0 1 0.4],'LineWidth',4)

ylabel('tongue angle')
xlabel(['time from ', num2str(params.alignEvent)]);
set(gca,'FontSize',sz)
xlim([-0.6 0.8])

windowSize = 2;

end



end



%%

% 
% ax = figure;
% 
% 
% s = size(y4);
% for kk = 1:s(2)
% 
% 
%  yy = y4(:,kk);
%  yhh = yhat4(:,kk);
% 
%  fl = find(yy == mode(yy));
% 
% yy(fl) = NaN;
% 
%  yhh(fl) = NaN;
% 
%     cla(ax)
%     plot(obj.time, movmean(yy,3), 'LineWidth',1.5)
%     hold on
%     plot(obj.time, movmean(yhh,3), 'LineWidth',1.5)
%     xlim([-1 2])
% 
% xline(obj.time(rang(1)),'LineWidth',2,'Color','b')
% xline(obj.time(rang(end)),'LineWidth',2,'Color','b')
% 
%     pause
% 
% end



%%

%%


% range = [-0.2 0.2];
range = [0 1];
cmp = linspecer;

figure;
subplot(2,2,1)
% imagesc(obj.time,1:size(y,2),y');
imagesc(obj.time,1:size(y,2),y');
xline(0,'LineWidth', 1.5)
colormap(cmp)
caxis([range]);
colorbar; grid off; axis tight;
% axis square;
box off; xlabel(['time from ', num2str(params.alignEvent)]); ylabel('Trial #');
title('actual')
axis tight
xlim([-1 2])

set(gca,'FontSize',sz)
hold on 
subplot(2,2,2)
% range = [-2 2];
% range = [-1. 1.];

imagesc(obj.time,1:size(yhat,2),yhat');
% imagesc(obj.time,1:size(yhat,2),yhat');

xline(0,'LineWidth', 1.5)
colormap(cmp)
caxis([range]);
colorbar; grid off; axis tight; 
% axis square;
box off; xlabel(['time from ', num2str(params.alignEvent)]); ylabel('Trial #');
set(gca,'FontSize',sz)
title('predicted')
axis tight
xlim([-1 2])


if par.cond2use == 9
yd = y1;
ydh = yhat1;
else
    yd = y4;
    ydh = yhat4;
end


subplot(2,2,3)
% imagesc(obj.time,1:size(y,2),y');
imagesc(obj.time,1:size(yd,2),yd');
xline(0,'LineWidth', 1.5)
colormap(cmp)
caxis([range]);
colorbar; grid off; axis tight;
% axis square;
box off; xlabel(['time from ', num2str(params.alignEvent)]); ylabel('Trial #');
title('actual')
axis tight
xlim([-1 2])

set(gca,'FontSize',sz)
hold on 
subplot(2,2,4)
% range = [-2 2];
% range = [-1. 1.];

imagesc(obj.time,1:size(ydh,2),ydh');
% imagesc(obj.time,1:size(yhat,2),yhat');

xline(0,'LineWidth', 1.5)
colormap(cmp)
caxis([range]);
colorbar; grid off; axis tight; 
% axis square;
box off; xlabel(['time from ', num2str(params.alignEvent)]); ylabel('Trial #');
set(gca,'FontSize',sz)
title('predicted')
axis tight
xlim([-1 2])

px = 50;
py = 50;
width = 900;
height = 400;
set(gcf, 'Position', [px, py, width, height]); % Set figure position and size

%%
figure;

for k = 1:2

if strcmp(par.feats{1}, 'tongue_length') 

numLicks = 10;
lickRange = 7;

rang1 = [480:800];
trang = [450:520];

tnew = obj.time(rang1);

if k == 1
ynew = y(rang1, :);
yneww = y(trang, :);
yhatnew = yhat(rang1, :);
else
ynew = y4(rang1, :);
yneww = y4(trang, :);
yhatnew = yhat4(rang1, :);
end

a = size(ynew);

% Find the mode of ynew (ignoring NaNs, if any)
mode_y = mode(ynew(:), 'all');
mean_y = mean(yneww(:), 'all');

% Initialize a cell array to store indices of licks for each trial
lick_indices = cell(a(2), 1); % 86 trials

% Loop through each trial
for trial = 1:a(2)
    % Get indices where values are not equal to the mode
    non_mode_idx = find(ynew(:, trial) ~= mode_y);
    
    % Identify sequences of consecutive indices
    if ~isempty(non_mode_idx)
        d = diff(non_mode_idx); % Compute differences between consecutive indices
        split_points = find(d > 1); % Find where the sequence breaks
        start_idx = [1; split_points + 1]; % Start of each segment
        end_idx = [split_points; length(non_mode_idx)]; % End of each segment
        
        % Extract sequences of at least 5 consecutive values
        licks = {};
        count = 0;
        for i = 1:length(start_idx)
            if (end_idx(i) - start_idx(i) + 1) >= lickRange % At least 5 values
                count = count + 1;
                licks{count} = non_mode_idx(start_idx(i):end_idx(i));
                if count == numLicks % Store up to 8 licks per trial
                    break;
                end
            end
        end
        
        % Store the licks for this trial
        lick_indices{trial} = licks;
    end
end



% Initialize a cell array to store concatenated values for each lick (8x1)
lick_values = cell(numLicks, 1);
lick_values_yhat = cell(numLicks, 1);

% Loop through each trial
for trial = 1:a(2)
    % Get the licks for the current trial
    licks = lick_indices{trial};
    
    % Add values to the corresponding lick index across all trials
    for lick_num = 1:min(numLicks, length(licks)) % Ensure we don’t exceed 8 licks
        lick_values{lick_num} = [lick_values{lick_num}; ynew(licks{lick_num}, trial)];
        lick_values_yhat{lick_num} = [lick_values_yhat{lick_num}; yhatnew(licks{lick_num}, trial)];
    end
end


% Initialize R² values
r2_values = nan(numLicks,1);

% Compute R² for each lick
for lick_num = 1:numLicks
    yy = lick_values{lick_num};      % Actual values
    yyhat = lick_values_yhat{lick_num}; % Predicted values

    % Ensure no empty data
    if ~isempty(yy) && ~isempty(yhat)
        % Compute R²
        % SStot = nansum((yy - nanmean(yy)).^2);
        SStot = nansum((yy - mean_y).^2);
        SSres = nansum((yy - yyhat).^2);
        r2_values(lick_num) = 1 - (SSres / SStot);
    end
end

% Compute mean and standard error
r2_mean = nanmean(r2_values);
r2_std = nanstd(r2_values);
r2_sem = r2_std / sqrt(8); % Standard Error of the Mean (SEM)

% Plot R² values without error bars
hold on;

% Add horizontal bars at each R² value
bar_length = 1; % Length of the bar in x-dimension
for lick_num = 1:numLicks
    x_range = [lick_num - bar_length / 2, lick_num + bar_length / 2]; % Define start and end points
    y_value = r2_values(lick_num); % Get corresponding R² value
    
    % Check if R² value is below 0 and plot in red
    if k==1
        plot(x_range, [y_value, y_value], 'b-', 'LineWidth', 2); % Black horizontal line
        plot(lick_num, y_value, 'o-', 'MarkerSize', 10, 'LineWidth', 2, 'Color', [0 0.447 0.741]); % MATLAB blue
    else
        plot(x_range, [y_value, y_value], 'r-', 'LineWidth', 2); % Black horizontal line
        plot(lick_num, y_value, 'o-', 'MarkerSize', 10, 'LineWidth', 2, 'Color', [1 0. 0]); % MATLAB blue
    end
        end


% Formatting the plot
xlabel('Lick Number', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('R² Value', 'FontSize', 14, 'FontWeight', 'bold');
title('R² Across Licks', 'FontSize', 16, 'FontWeight', 'bold');


title('M1 R4 R² Across Licks', 'FontSize', 16, 'FontWeight', 'bold');


xlim([0.5 numLicks + 0.5]); % Adjust x-axis limits for spacing
ylim([min(r2_values) - 0.05, max(r2_values) + 0.05]); % Adjust y-axis limits dynamically
xline(0)
yline(0)
xlim([-1 10])
ylim([-3 1])
box off; % Remove box outline for a cleaner look

hold off;

end

end

px = 50;
py = 50;
width = 350;
height = 750;
set(gcf, 'Position', [px, py, width, height]); % Set figure position and size



%%


%%


ypL = y(480:1000,:);
yhatpL = yhat(480:1000,:); 

num_trials = size(ypL, 2); % Number of trials
y_perlick = cell(1, num_trials);
yhat_perlick = cell(1, num_trials);

for trial = 1:num_trials
    y_trial = ypL(:, trial);      % Extract y for the current trial
    yhat_trial = yhatpL(:, trial);% Extract yhat for the current trial
    
    nonzero_idx = find(y_trial > 0); % Find indices where y is nonzero
    lick_groups = diff([0; nonzero_idx]) > 1; % Detect breaks between licks
    lick_start_idx = find(lick_groups); % Start indices of each lick
    
    % Store means of each lick
    y_means = [];
    yhat_means = [];
    
    for i = 1:length(lick_start_idx)
        if i < length(lick_start_idx)
            lick_range = nonzero_idx(lick_start_idx(i):lick_start_idx(i+1)-1);
        else
            lick_range = nonzero_idx(lick_start_idx(i):end);
            lick_range = lick_range(1:end);
            % lick_range = [lick_range(1)-4:1:lick_range(end)+6];
            % lick_range(find(lick_range >501)) = [];

        end
        
        y_means = [y_means; max(y_trial(lick_range))]; % Mean of y for this lick
        yhat_means = [yhat_means; max(yhat_trial(lick_range))]; % Mean of yhat for this lick
    end
    
    % Store in cell arrays
    y_perlick{trial} = y_means;
    yhat_perlick{trial} = yhat_means;
end


max_licks = 8; % Maximum number of licks to analyze
num_trials = length(y_perlick); % Get the number of trials

% Initialize matrices with NaNs
y_licks = nan(max_licks, num_trials);
yhat_licks = nan(max_licks, num_trials);

for trial = 1:num_trials
    y_trial_licks = y_perlick{trial}; % Extract lick means for this trial
    yhat_trial_licks = yhat_perlick{trial};
    
    num_licks = length(y_trial_licks);
    
    if num_licks > 0
        % Ensure correct assignment by taking the first min(num_licks, max_licks) elements
        y_licks(1:min(num_licks, max_licks), trial) = y_trial_licks(1:min(num_licks, max_licks));
        yhat_licks(1:min(num_licks, max_licks), trial) = yhat_trial_licks(1:min(num_licks, max_licks));
    end
end

% Compute mean and standard error (SE)
y_mean = nanmean(y_licks, 2);
y_se = nanstd(y_licks, 0, 2) ./ sqrt(sum(~isnan(y_licks), 2)); % SE = std/sqrt(n)

yhat_mean = nanmean(yhat_licks, 2);
yhat_se = nanstd(yhat_licks, 0, 2) ./ sqrt(sum(~isnan(yhat_licks), 2));

% Plot
figure;
subplot(2,1,1)
hold on;
errorbar(1:max_licks, y_mean, y_se, '.', 'LineWidth', 2, 'MarkerSize', 25, 'DisplayName', 'y (actual)');
errorbar(1:max_licks, yhat_mean, yhat_se, '.', 'LineWidth', 2, 'MarkerSize', 25, 'DisplayName', 'yhat (predicted)');

xlabel('Lick Number');
ylabel('Mean ± SE T Length');
title('Mean and SE of y and yhat across licks');
% legend();
% grid on;
% 
xlim([-1 9])
ylim([0 0.8])

%


if par.cond2use == 9
yd = y1;
ydh = yhat1;
else
    yd = y4;
    ydh = yhat4;
end




ypL = yd(480:1000,:);
yhatpL = ydh(480:1000,:); 

num_trials = size(ypL, 2); % Number of trials
y_perlick = cell(1, num_trials);
yhat_perlick = cell(1, num_trials);

for trial = 1:num_trials
    y_trial = ypL(:, trial);      % Extract y for the current trial
    yhat_trial = yhatpL(:, trial);% Extract yhat for the current trial
    
    nonzero_idx = find(y_trial > 0); % Find indices where y is nonzero
    lick_groups = diff([0; nonzero_idx]) > 1; % Detect breaks between licks
    lick_start_idx = find(lick_groups); % Start indices of each lick
    
    % Store means of each lick
    y_means = [];
    yhat_means = [];
    
    for i = 1:length(lick_start_idx)
        if i < length(lick_start_idx)
            lick_range = nonzero_idx(lick_start_idx(i):lick_start_idx(i+1)-1);
        else
            lick_range = nonzero_idx(lick_start_idx(i):end);
            lick_range = lick_range(1:end);
            % lick_range = [lick_range(1)-4:1:lick_range(end)+6];
            % lick_range(find(lick_range >501)) = [];

        end
        
        y_means = [y_means; max(y_trial(lick_range))]; % Mean of y for this lick
        yhat_means = [yhat_means; max(yhat_trial(lick_range))]; % Mean of yhat for this lick
    end
    
    % Store in cell arrays
    y_perlick{trial} = y_means;
    yhat_perlick{trial} = yhat_means;
end


max_licks = 8; % Maximum number of licks to analyze
num_trials = length(y_perlick); % Get the number of trials

% Initialize matrices with NaNs
y_licks = nan(max_licks, num_trials);
yhat_licks = nan(max_licks, num_trials);

for trial = 1:num_trials
    y_trial_licks = y_perlick{trial}; % Extract lick means for this trial
    yhat_trial_licks = yhat_perlick{trial};
    
    num_licks = length(y_trial_licks);
    
    if num_licks > 0
        % Ensure correct assignment by taking the first min(num_licks, max_licks) elements
        y_licks(1:min(num_licks, max_licks), trial) = y_trial_licks(1:min(num_licks, max_licks));
        yhat_licks(1:min(num_licks, max_licks), trial) = yhat_trial_licks(1:min(num_licks, max_licks));
    end
end

% Compute mean and standard error (SE)
y_mean = nanmean(y_licks, 2);
y_se = nanstd(y_licks, 0, 2) ./ sqrt(sum(~isnan(y_licks), 2)); % SE = std/sqrt(n)

yhat_mean = nanmean(yhat_licks, 2);
yhat_se = nanstd(yhat_licks, 0, 2) ./ sqrt(sum(~isnan(yhat_licks), 2));

% Plot
% figure;
subplot(2,1,2)
hold on;
errorbar(1:max_licks, y_mean, y_se, '.', 'LineWidth', 2, 'MarkerSize', 25, 'DisplayName', 'y (actual)');
errorbar(1:max_licks, yhat_mean, yhat_se, '.', 'LineWidth', 2, 'MarkerSize', 25, 'DisplayName', 'yhat (predicted)');

xlabel('Lick Number');
ylabel('Mean ± SE T Length');
title('Mean and SE of y and yhat across licks');
% legend();
% grid on;
% 
px = 50;
py = 50;
width = 200;
height = 750;
set(gcf, 'Position', [px, py, width, height]); % Set figure position and size
xlim([-1 9])
ylim([0 0.8])


% % subplot(2,1,2)
% figure;
% plot(1:max_licks,y_mean-yhat_mean,'o')
% px = 150;
% py = 50;
% width = 200;
% height = 450;
% set(gcf, 'Position', [px, py, width, height]); % Set figure position and size

%% For Jaw 


if strcmp(par.feats{1}, 'jaw_ydisp_view1') || strcmp(par.feats{1}, 'tongue_yvel_view1') || strcmp(par.feats{1}, 'jaw_yvel_view1')

if par.cond2use == 9
yd = y1;
ydh = yhat1;
else
    yd = y4;
    ydh = yhat4;
end



for kk = 1:2

trang = [480:1000];


if kk == 1
ypL = y(trang,:);
yhatpL = yhat(trang,:); 
else
if par.cond2use == 9
ypL = y1(trang,:);
yhatpL = yhat1(trang,:); 
else
    ypL = y4(trang,:);
yhatpL = yhat4(trang,:); 
end

end



num_trials = size(ypL, 2); % Number of trials
y_perlick = cell(1, num_trials);
yhat_perlick = cell(1, num_trials);

for trial = 1:num_trials
    y_trial = ypL(:, trial);      % Extract y for the current trial
    yhat_trial = yhatpL(:, trial);% Extract yhat for the current trial
    
% 
% figure; 
% plot(y_trial)
% hold on 
% plot(movmean(diff(y_trial),5))






% Assuming your data is stored in variable `y` with corresponding time `t`
yF = movmean(y_trial,12); % Replace with actual signal
t = obj.time(trang); % Replace with actual time vector if available

% Find local maxima (peaks) with prominence threshold to filter out noise
[peaks, locs_max] = findpeaks(yF, t, 'MinPeakProminence', 0.1);

% Find local minima (valleys) using the inverted signal
[valleys, locs_min] = findpeaks(-yF, t, 'MinPeakProminence', 0.02);
% [valleys, locs_min] = findpeaks(-yF, t);
valleys = -valleys; % Restore original values

% figure;
% % subplot(2,1,1)
% plot(t, yF, 'b'); hold on;
% plot(locs_max, peaks, 'ro', 'MarkerFaceColor', 'r'); % Maxima in red
% plot(locs_min, valleys, 'go', 'MarkerFaceColor', 'g'); % Minima in green
% title('Detected Oscillations');
% legend('Signal', 'Peaks', 'Valleys');

num_oscillations = min(length(locs_max), length(locs_min)) - 1;

% Initialize lick start indices and extracted oscillation ranges
lick_start_idx = nan(num_oscillations, 1);

% Extract oscillations
for i = 1:num_oscillations
    % Get the time range for this oscillation cycle
    osc_start = min(locs_min(i), locs_max(i));
    osc_end = max(locs_min(i+1), locs_max(i));
    
    % Find indices corresponding to this range in the time vector
    lick_range = find(t >= osc_start & t <= osc_end);
    
    % Store lick start index (beginning of oscillation cycle)
    lick_start_idx(i) = lick_range(1);
    
    % Store the extracted oscillation data
end



    nonzero_idx = find(yF); % Find indices where y is nonzero
    lick_groups = diff([0; nonzero_idx]) > 1; % Detect breaks between licks
    % lick_start_idx = find(lick_groups); % Start indices of each lick
    
    % Store means of each lick
    y_means = [];
    yhat_means = [];
    
    for i = 1:length(lick_start_idx)
        if i < length(lick_start_idx)
            lick_range = nonzero_idx(lick_start_idx(i):lick_start_idx(i+1)-1);
        else
            lick_range = nonzero_idx(lick_start_idx(i):end);
            lick_range = lick_range(1:end);
            % lick_range = [lick_range(1)-4:1:lick_range(end)+6];
            % lick_range(find(lick_range >501)) = [];

        end
        
        y_means = [y_means; max(y_trial(lick_range))]; % Mean of y for this lick
        yhat_means = [yhat_means; max(yhat_trial(lick_range))]; % Mean of yhat for this lick
    end
    
    % Store in cell arrays
    y_perlick{trial} = y_means;
    yhat_perlick{trial} = yhat_means;
end


max_licks = 8; % Maximum number of licks to analyze
num_trials = length(y_perlick); % Get the number of trials

% Initialize matrices with NaNs
y_licks = nan(max_licks, num_trials);
yhat_licks = nan(max_licks, num_trials);

for trial = 1:num_trials
    y_trial_licks = y_perlick{trial}; % Extract lick means for this trial
    yhat_trial_licks = yhat_perlick{trial};
    
    num_licks = length(y_trial_licks);
    
    if num_licks > 0
        % Ensure correct assignment by taking the first min(num_licks, max_licks) elements
        y_licks(1:min(num_licks, max_licks), trial) = y_trial_licks(1:min(num_licks, max_licks));
        yhat_licks(1:min(num_licks, max_licks), trial) = yhat_trial_licks(1:min(num_licks, max_licks));
    end
end

% Compute mean and standard error (SE)
y_mean = nanmean(y_licks, 2);
y_se = nanstd(y_licks, 0, 2) ./ sqrt(sum(~isnan(y_licks), 2)); % SE = std/sqrt(n)

yhat_mean = nanmean(yhat_licks, 2);
yhat_se = nanstd(yhat_licks, 0, 2) ./ sqrt(sum(~isnan(yhat_licks), 2));

% Plot
figure;
% subplot(2,1,1)
hold on;
errorbar(1:max_licks, y_mean, y_se, '.', 'LineWidth', 2, 'MarkerSize', 25, 'DisplayName', 'y (actual)');
errorbar(1:max_licks, yhat_mean, yhat_se, '.', 'LineWidth', 2, 'MarkerSize', 25, 'DisplayName', 'yhat (predicted)');

xlabel('Lick Number');
ylabel('Mean ± SE T Length');
title('Mean and SE of y and yhat across licks');
% legend();
% grid on;
% 
xlim([-1 9])
ylim([0 0.4])
px = 50;
py = 50;
width = 200;
height = 750;
set(gcf, 'Position', [px, py, width, height]); % Set figure position and size
xlim([-1 9])
ylim([0 0.4])

end

end

%% Saving data to export to HMM-GLM Analysis
% Construct the full folder path based on the animal number and date
save_folder_engaged = fullfile(save_folder, "Engaged");
save_folder_disengaged = fullfile(save_folder, "Disengaged");

% Create the directory if it doesn't exist
if ~exist(save_folder_engaged, 'dir')
    mkdir(save_folder_engaged);
end

if ~exist(save_folder_disengaged, 'dir')
    mkdir(save_folder_disengaged);
end

% SET WHETHER YOU ARE ENGAGED OR DISENGAGED FITTING
folder = save_folder_engaged;

disp("Saving Y R1")
save_path = fullfile(folder, 'Y_test_R1.csv');
writematrix(Y.test1, save_path);

disp("Saving Y R4")
save_path = fullfile(folder, 'Y_test_R4.csv');
writematrix(Y.test4, save_path)

disp("Saving Y train")
save_path = fullfile(folder, 'Y_train.csv');
writematrix(Y.train, save_path)

disp("Saving X train")
save_path = fullfile(folder, 'X_train.csv');
writematrix(X.train, save_path)

disp("Saving X R1")
save_path = fullfile(folder, 'X_test_R1.csv');
writematrix(X.test1, save_path)

disp("Saving X R4")
save_path = fullfile(folder, 'X_test_R4.csv');
writematrix(X.test4, save_path)

disp("Saving Kridge")
save_path = fullfile(folder, 'Kridge.csv');
writematrix(kridge, save_path)

disp("Saving Betas")
save_path = fullfile(folder, 'Betas.csv');
writematrix(B, save_path)

%%


% 
% n = 10;
% s = size(y);
% alph = 0.1;
% ct = 1;
% 
% for i = 1:n:s(2)
%     f = figure;
%     for j = 1:n
% 
%         ax = nexttile;
% 
% hold on;
% 
% plot(obj.time,y(:,ct)','Color','k','LineWidth',0.5)
% hold on 
% plot(obj.time,yhat(:,ct)','Color','r','LineWidth',0.75)
% 
% % plot(obj.time,y(:,ct)',{'Color','k','LineWidth',3})
% % hold on 
% % plot(obj.time,yhat(:,ct)',{'Color',[1 0 0 0.5],'LineWidth',2})
% 
% title(['Trial = ', num2str(j)]);
% xlabel(['time from ', num2str(params.alignEvent)]);
% ylabel('Tongue Length');
% xlim([-2 3.5])
% 
%         ct = ct + 1;
% 
%     end
% end









%%

function sets = findConsecutiveSets(indices, minLength, maxLength)
    sets = {};
    currentSet = [];
    
    i = 1;
    while i <= length(indices)-1
        if indices(i+1) - indices(i) == 1
            currentSet = [currentSet, indices(i)];
        else
            currentSet = [currentSet, indices(i)];
            while length(currentSet) >= minLength
                % Truncate to maxLength
                truncatedSet = currentSet(1:min(length(currentSet), maxLength));
                sets{end+1} = truncatedSet;
                currentSet = currentSet(min(length(currentSet), maxLength)+1:end);
            end
            currentSet = [];
        end
        
        i = i + 1;
    end
    
    % Check the last set
    currentSet = [currentSet, indices(end)];
    while length(currentSet) >= minLength
        % Truncate to maxLength
        truncatedSet = currentSet(1:min(length(currentSet), maxLength));
        sets{end+1} = truncatedSet;
        currentSet = currentSet(min(length(currentSet), maxLength)+1:end);
    end
end







