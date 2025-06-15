clear;clc;

dataset_name = ["mg_", "lorenz_"];
reg_name = ["l2_", "smoothl1_"];
timestep = 1:1:1000;


% load predictions and ground-truth
% mackey-glass
mg_l2_preds = table2array(readtable(dataset_name(1)+reg_name(1)+'preds.csv'));
mg_l2_trues = table2array(readtable(dataset_name(1)+reg_name(1)+'trues.csv'));

mg_smoothl1_preds = table2array(readtable(dataset_name(1)+reg_name(2)+'preds.csv'));
mg_smoothl1_trues = table2array(readtable(dataset_name(1)+reg_name(2)+'trues.csv'));


% lorenz
lorenz_l2_preds = table2array(readtable(dataset_name(2)+reg_name(1)+'preds.csv'));
lorenz_l2_trues = table2array(readtable(dataset_name(2)+reg_name(1)+'trues.csv'));

lorenz_smoothl1_preds = table2array(readtable(dataset_name(2)+reg_name(2)+'preds.csv'));
lorenz_smoothl1_trues = table2array(readtable(dataset_name(2)+reg_name(2)+'trues.csv'));


% plot
close all
figure
set(gcf, 'Units','centimeters','Position',[5,0,30,15])
t = tiledlayout(4,2, "TileSpacing","compact");
% mg
nexttile
plot(timestep, mg_l2_trues, 'r', timestep, mg_l2_preds, '--k', "LineWidth", 1)
title("Mackey-Glass with ESN", "FontSize", 12)
ylim([-0.55, 0.35])
% xlabel('t')
ylabel('x')
nexttile
plot(timestep, mg_smoothl1_trues, 'r', timestep, mg_smoothl1_preds, '--k', "LineWidth", 1)
title("Mackey-Glass with SmoothL1-ESN", "FontSize", 12)
ylim([-0.55, 0.35])
% xlabel('t')
% ylabel('x')
% lorenz dim 1
nexttile
plot(timestep, lorenz_l2_trues(:,1), 'r', timestep, lorenz_l2_preds(:,1),'--k', "LineWidth", 1)
title("Lorenz Dim 1 with ESN", "FontSize", 12)
ylim([-20, 20])
% xlabel('t')
ylabel('x')
nexttile
plot(timestep, lorenz_smoothl1_trues(:,1), 'r', timestep, lorenz_smoothl1_preds(:,1),'--k', "LineWidth", 1)
title("Lorenz Dim 1 with SmoothL1-ESN", "FontSize", 12)
ylim([-20, 20])
% xlabel('t')
% ylabel('x')
% lorenz dim 2
nexttile
plot(timestep, lorenz_l2_trues(:,2), 'g', timestep, lorenz_l2_preds(:,2),'--k', "LineWidth", 1)
title("Lorenz Dim 2 with ESN", "FontSize", 12)
ylim([-25, 25])
% xlabel('t')
ylabel('y')
nexttile
plot(timestep, lorenz_smoothl1_trues(:,2), 'g', timestep, lorenz_smoothl1_preds(:,2),'--k', "LineWidth", 1)
title("Lorenz Dim 2 with SmoothL1-ESN", "FontSize", 12)
ylim([-25, 25])
% xlabel('t')
% ylabel('y')
% lorenz dim 3
nexttile
plot(timestep, lorenz_l2_trues(:,3), 'b', timestep, lorenz_l2_preds(:,3),'--k', "LineWidth", 1)
title("Lorenz Dim 3 with ESN", "FontSize", 12)
ylim([3, 45])
xlabel('t')
ylabel('z')
nexttile
plot(timestep, lorenz_smoothl1_trues(:,3), 'b', timestep, lorenz_smoothl1_preds(:,3),'--k', "LineWidth", 1)
title("Lorenz Dim 3 with SmoothL1-ESN", "FontSize", 12)
ylim([3, 45])
xlabel('t')
% ylabel('z')