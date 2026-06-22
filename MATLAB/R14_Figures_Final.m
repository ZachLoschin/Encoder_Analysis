% Zachary Loschinskey
% Drs. Mike Economo and Brian DePasquale
% GLM-HMM figure creation for disengagement analysis
% October 2025

clear; clc; close all;

% ****************************** %
%  User parameters & setup      %
% ****************************** %
base_dir       = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Final\';
alt_base_dir   = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Processed_Encoder\R14';
subfolder      = '';

animalList     = {'TD13d', 'TD15d', 'TD1d', 'TD22d', 'TD23d', 'TD4d'};  
% <-- add all animals you want to process

% alignment params
pre_points     = 10;
post_points    = 200;
window_size    = pre_points + 1 + post_points;  % =211
GCidx          = pre_points + 1;                % GC index

% plotting params
R1_color       = [0 0 1];
R4_color       = [0 0.76 1];
alpha_sess     = 0.5;
alpha_mean     = 1.0;
maxPostVisible = 150;  % only plot first 150 samples post-contact
timeSec        = (-pre_points : post_points) / 100;  
vis_idx        = timeSec <= maxPostVisible/100;

% figure output folder
fig_dir = fullfile(base_dir,'Figures');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

% prepare storage for grand means
nAnimals     = numel(animalList);
GM_R1_all    = cell(nAnimals,1);
GM_R4_all    = cell(nAnimals,1);

% ****************************** %
%  Main loop over animals       %
% ****************************** %
for a = 1:nAnimals
  animalName = animalList{a};
  fprintf('Processing animal %s …\n', animalName);
  
  % find the sessions for this animal
  pattern     = fullfile(base_dir, [animalName '_*']);
  sessionInfo = dir(pattern);
  sessionInfo = sessionInfo([sessionInfo.isdir]);
  nSess = numel(sessionInfo);
  if nSess==0
    warning('No sessions found for %s, skipping.', animalName);
    continue;
  end
  
  % storage for per-session averages
  avgR1_sessions = nan(nSess, window_size);
  avgR4_sessions = nan(nSess, window_size);

  % loop sessions
  for s = 1:nSess
    sessName = sessionInfo(s).name;
    save_dir = fullfile(base_dir, sessName, subfolder);
    alt_dir  = fullfile(alt_base_dir, sessName);
    if ~isfolder(save_dir) || ~isfolder(alt_dir)
      warning('Missing dirs for %s, skipping.', sessName);
      continue;
    end
    
    % load
    R1s = readmatrix(fullfile(save_dir,'R1_States_Reg.csv'));   % [nTrials×211]
    R4s = readmatrix(fullfile(save_dir,'R14_States_Reg.csv'));
    C1  = readmatrix(fullfile(alt_dir,'FCs_R1.csv')) - 100;
    C4  = readmatrix(fullfile(alt_dir,'FCs_R4.csv')) - 100;
    
    % align & exponentiate per trial
    nT1 = size(R1s,1);
    aligned1 = nan(nT1,window_size);
    for t = 1:nT1
      ap = GCidx + round(C1(t));
      st = ap - pre_points; en = ap + post_points;
      vs = max(st,1); ve = min(en,size(R1s,2));
      is_ = vs - st + 1; ie_ = is_ + (ve-vs);
      aligned1(t,is_:ie_) = exp( R1s(t,vs:ve) );
    end
    avgR1_sessions(s,:) = nanmean(aligned1,1);
    
    nT4 = size(R4s,1);
    aligned4 = nan(nT4,window_size);
    for t = 1:nT4
      ap = GCidx + round(C4(t));
      st = ap - pre_points; en = ap + post_points;
      vs = max(st,1); ve = min(en,size(R4s,2));
      is_ = vs - st + 1; ie_ = is_ + (ve-vs);
      aligned4(t,is_:ie_) = exp( R4s(t,vs:ve) );
    end
    avgR4_sessions(s,:) = nanmean(aligned4,1);
  end
  
  % compute grand means for this animal
  GM_R1 = mean(avgR1_sessions,1);
  GM_R4 = mean(avgR4_sessions,1);
  
  GM_R1_all{a} = GM_R1;
  GM_R4_all{a} = GM_R4;
  
  % ---- plot for this animal ----
  hF = figure('Visible','off'); hold on;
    % faded session traces
    plot(timeSec(vis_idx)', avgR1_sessions(:,vis_idx)', 'Color',[R1_color,alpha_sess]);
    plot(timeSec(vis_idx)', avgR4_sessions(:,vis_idx)', 'Color',[R4_color,alpha_sess]);
    % solid grand means
    % solid grand means (capture handles)
    hGM1 = plot(timeSec(vis_idx), GM_R1(vis_idx), 'Color',[R1_color,alpha_mean],'LineWidth',4);
    hGM4 = plot(timeSec(vis_idx), GM_R4(vis_idx), 'Color',[R4_color,alpha_mean],'LineWidth',4);

    xline(0,'--k','LineWidth',1);
    xlabel('Time from first port contact (s)');
    ylabel('Engaged State Probability');
    title(sprintf('%s: %d sessions', animalName, nSess), 'Interpreter','none');
    xlim([timeSec(find(vis_idx,1,'first')), timeSec(find(vis_idx,1,'last'))]);
    ylim([0 1]);
    xticks([-0.1, 0, 0.5, 1.0, 1.5]);
    yticks(0:0.2:0.8);
    
    grid on;

    % add the legend back
    legend([hGM1,hGM4], {'R1 session average','R4 session average'}, ...
           'Location','Best','Interpreter','none');
    
  savefig(hF, fullfile(fig_dir, sprintf('%s_AverageInference.fig',animalName)));
  print(hF, fullfile(fig_dir, sprintf('%s_AverageInference.png',animalName)), '-dpng','-r300');
  close(hF);
  
  fprintf('  → saved figure for %s\n', animalName);
end

% At this point you have:
%   cell array GM_R1_all and GM_R4_all
%   vector timeSec
%   list animalList
% You can now continue in this script to make a combined plot across animals.


%% Combine across animals: average + std, then plot

% find valid (non‐empty) entries
validR1 = ~cellfun(@isempty, GM_R1_all);
matR1   = vertcat(GM_R1_all{validR1});   % [nValidAnimals × window_size]
validR4 = ~cellfun(@isempty, GM_R4_all);
matR4   = vertcat(GM_R4_all{validR4});

% compute across‐animal mean & std
meanR1 = mean(matR1,1);    stdR1 = std(matR1,0,1);
meanR4 = mean(matR4,1);    stdR4 = std(matR4,0,1);

% now plot
f1 = figure; hold on;
% faded per‐animal grand means
for a = 1:size(matR1,1)
  h1 = plot(timeSec(vis_idx), matR1(a,vis_idx), 'LineWidth',1.5);
  h1.Color = [R1_color, alpha_sess];
  h4 = plot(timeSec(vis_idx), matR4(a,vis_idx), 'LineWidth',1.5);
  h4.Color = [R4_color, alpha_sess];
end

% thick overall mean
hGM1 = plot(timeSec(vis_idx), meanR1(vis_idx), 'LineWidth',4, 'Color',[R1_color, alpha_mean]);
hGM4 = plot(timeSec(vis_idx), meanR4(vis_idx), 'LineWidth',4, 'Color',[R4_color, alpha_mean]);

% mark t=0
xline(0, '--k', 'LineWidth',1);

% formatting
xlabel('Time from first port contact (s)');
ylabel('Engaged State Probability');
title('Average Inference across animals','Interpreter','none');
xlim([timeSec(find(vis_idx,1,'first')), timeSec(find(vis_idx,1,'last'))]);
ylim([0,1]);
xticks([-0.1, 0, 0.5, 1.0, 1.5]);
yticks(0:0.2:0.8);
grid on;

% legend for the two mean lines
legend([hGM1,hGM4], {'R1 mean across animals','R4 mean across animals'}, ...
       'Location','Best','Interpreter','none');

% save the first figure
saveas(f1, fullfile(fig_dir, 'AverageInference_Line.png'));
% optionally also save as .fig
savefig(f1, fullfile(fig_dir, 'AverageInference_Line.fig'));


%% Plot mean ± STD cloud across animals

f2 = figure; hold on;
alpha_cloud = 0.3;
x   = timeSec(vis_idx);
% R1 cloud
y1u = meanR1(vis_idx) + stdR1(vis_idx);
y1l = meanR1(vis_idx) - stdR1(vis_idx);
hC1 = fill([x, fliplr(x)], [y1u, fliplr(y1l)], R1_color, ...
           'FaceAlpha', alpha_cloud, 'EdgeColor','none');

% R4 cloud
y4u = meanR4(vis_idx) + stdR4(vis_idx);
y4l = meanR4(vis_idx) - stdR4(vis_idx);
hC4 = fill([x, fliplr(x)], [y4u, fliplr(y4l)], R4_color, ...
           'FaceAlpha', alpha_cloud, 'EdgeColor','none');

% Plot mean lines on top
hGM1 = plot(x, meanR1(vis_idx), 'LineWidth',4, 'Color',[R1_color, alpha_mean]);
hGM4 = plot(x, meanR4(vis_idx), 'LineWidth',4, 'Color',[R4_color, alpha_mean]);

% Mark t=0
xline(0, '--k', 'LineWidth',1);

% Formatting
xlabel('Time from first port contact (s)');
ylabel('Engaged State Probability');
title('Average Inference across animals','Interpreter','none');
xlim([x(1), x(end)]);
ylim([0,1]);
xticks([-0.1, 0, 0.5, 1.0, 1.5]);
yticks(0:0.2:0.8);
grid on;

% Legend (lines only)
legend([hGM1,hGM4], {'R1 mean ±1 SD','R4 mean ±1 SD'}, ...
       'Location','Best','Interpreter','none');

% save the second figure
saveas(f2, fullfile(fig_dir, 'AverageInference_Cloud.png'));
savefig(f2, fullfile(fig_dir, 'AverageInference_Cloud.fig'));
