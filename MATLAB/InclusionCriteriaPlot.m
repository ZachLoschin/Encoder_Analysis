% Define path
path = 'C:\Research\Encoder_Modeling\Encoder_Analysis\Results_Window_R14\All_R2_Means.csv';

% Read in comma delimited excel file
opts = detectImportOptions(path, 'Delimiter', ',');
T = readtable(path, opts);

% Extract data from PC1 and PC2 R2 columns
dat = T{:, 2:3};
thresh = -1e3;
mask = all(dat > thresh, 2);  % Keep rows where both values are above threshold
dat_clean = dat(mask, :);


scatter(dat_clean(:,1), dat_clean(:,2))
xlabel("PC1 R^2")
ylabel("PC2 R^2")
title("Scatter of GC to First Contact Encodability")
xlim([0, 1])
ylim([0, 1])