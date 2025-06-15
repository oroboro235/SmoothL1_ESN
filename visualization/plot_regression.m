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

% mackey-glass
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(2,1, "TileSpacing","compact");
nexttile
plot(timestep, mg_l2_trues, 'r', timestep, mg_l2_preds, '--k', "LineWidth", 1)
title("Mackey-Glass with ESN", "FontSize", 12)
nexttile
plot(timestep, mg_smoothl1_trues, 'r', timestep, mg_smoothl1_preds, '--k', "LineWidth", 1)
title("Mackey-Glass with SmoothL1-ESN", "FontSize", 12)

xlabel(t,'t')
ylabel(t,'x')
cb = legend('True', 'Predict');
% % cb.Layout.Tile = "north";
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])
exportgraphics(gcf, "mg_outputs.pdf",'ContentType','vector')


% lorenz
lorenz_l2_preds = table2array(readtable(dataset_name(2)+reg_name(1)+'preds.csv'));
lorenz_l2_trues = table2array(readtable(dataset_name(2)+reg_name(1)+'trues.csv'));

lorenz_smoothl1_preds = table2array(readtable(dataset_name(2)+reg_name(2)+'preds.csv'));
lorenz_smoothl1_trues = table2array(readtable(dataset_name(2)+reg_name(2)+'trues.csv'));

% dim 1
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(2, 1, "TileSpacing","compact");
nexttile
plot(timestep, lorenz_l2_trues(:,1), 'r', timestep, lorenz_l2_preds(:,1),'--k', "LineWidth", 1)
title("Lorenz Dim 1 with ESN", "FontSize", 12)
nexttile
plot(timestep, lorenz_smoothl1_trues(:,1), 'r', timestep, lorenz_smoothl1_preds(:,1),'--k', "LineWidth", 1)
title("Lorenz Dim 1 with SmoothL1-ESN", "FontSize", 12)
xlabel(t,'t')
ylabel(t,'x')
cb = legend('True', 'Predict');
% % cb.Layout.Tile = "north";
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])
exportgraphics(gcf, "lorenz_1_outputs.pdf",'ContentType','vector')

% dim 2
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(2, 1, "TileSpacing","compact");
nexttile
plot(timestep, lorenz_l2_trues(:,2), 'g', timestep, lorenz_l2_preds(:,2),'--k', "LineWidth", 1)
title("Lorenz Dim 2 with ESN", "FontSize", 12)
nexttile
plot(timestep, lorenz_smoothl1_trues(:,2), 'g', timestep, lorenz_smoothl1_preds(:,2),'--k', "LineWidth", 1)
title("Lorenz Dim 2 with SmoothL1-ESN", "FontSize", 12)
xlabel(t,'t')
ylabel(t,'y')
cb = legend('True', 'Predict');
% % cb.Layout.Tile = "north";
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])
exportgraphics(gcf, "lorenz_2_outputs.pdf",'ContentType','vector')


% dim 3
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(2, 1, "TileSpacing","compact");
nexttile
plot(timestep, lorenz_l2_trues(:,3), 'b', timestep, lorenz_l2_preds(:,3),'--k', "LineWidth", 1)
title("Lorenz Dim 3 with ESN", "FontSize", 12)
nexttile
plot(timestep, lorenz_smoothl1_trues(:,3), 'b', timestep, lorenz_smoothl1_preds(:,3),'--k', "LineWidth", 1)
title("Lorenz Dim 3 with SmoothL1-ESN", "FontSize", 12)
xlabel(t,'t')
ylabel(t,'z')
cb = legend('True', 'Predict');
% % cb.Layout.Tile = "north";
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])
exportgraphics(gcf, "lorenz_3_outputs.pdf",'ContentType','vector')

close all