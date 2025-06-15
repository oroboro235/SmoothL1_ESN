clear;clc;

dataset_name = ["mg_", "lorenz_"];
reg_name = ["l2_", "smoothl1_"];
timestep = 1:1:1000;


% lorenz
lorenz_l2_preds = table2array(readtable(dataset_name(2)+reg_name(1)+'preds.csv'));
lorenz_l2_trues = table2array(readtable(dataset_name(2)+reg_name(1)+'trues.csv'));

lorenz_smoothl1_preds = table2array(readtable(dataset_name(2)+reg_name(2)+'preds.csv'));
lorenz_smoothl1_trues = table2array(readtable(dataset_name(2)+reg_name(2)+'trues.csv'));

lorenz_l2_preds_x = lorenz_l2_preds(:, 1);
lorenz_l2_preds_y = lorenz_l2_preds(:, 2);
lorenz_l2_preds_z = lorenz_l2_preds(:, 3);

lorenz_l2_trues_x = lorenz_l2_trues(:, 1);
lorenz_l2_trues_y = lorenz_l2_trues(:, 2);
lorenz_l2_trues_z = lorenz_l2_trues(:, 3);

lorenz_smoothl1_preds_x = lorenz_smoothl1_preds(:, 1);
lorenz_smoothl1_preds_y = lorenz_smoothl1_preds(:, 2);
lorenz_smoothl1_preds_z = lorenz_smoothl1_preds(:, 3);

lorenz_smoothl1_trues_x = lorenz_smoothl1_trues(:, 1);
lorenz_smoothl1_trues_y = lorenz_smoothl1_trues(:, 2);
lorenz_smoothl1_trues_z = lorenz_smoothl1_trues(:, 3);

close all;
figure;
plot3(lorenz_l2_trues_x, lorenz_l2_trues_y, lorenz_l2_trues_z, '-r',lorenz_l2_preds_x, lorenz_l2_preds_y, lorenz_l2_preds_z, '--k', 'LineWidth', 1)
set(gca, 'XDir', 'reverse', 'YDir', 'reverse', 'TickDir', 'in')
view(gca, [-135 15])
axis tight;
grid on;
title('Lorenz with ESN')
xlabel('X')
ylabel('Y')
zlabel('Z')
cb = legend('True', 'Predict');
set(cb,'Position',[0.380502978715957 0.859523810623657 0.263392854721418 0.0464285703287238],...
    'Orientation','horizontal');
exportgraphics(gcf, "lorenz_l2.pdf",'ContentType','image')

close all;
figure;
plot3(lorenz_smoothl1_trues_x, lorenz_smoothl1_trues_y, lorenz_smoothl1_trues_z, '-r', lorenz_smoothl1_preds_x, lorenz_smoothl1_preds_y, lorenz_smoothl1_preds_z, '--k','LineWidth', 1)
set(gca, 'XDir', 'reverse', 'YDir', 'reverse', 'TickDir', 'in')
view(gca, [-135 15])
axis tight;
grid on;
title('Lorenz with SmoothL1-ESN')
xlabel('X')
ylabel('Y')
zlabel('Z')
cb = legend('True', 'Predict');
set(cb,'Position',[0.380502978715957 0.859523810623657 0.263392854721418 0.0464285703287238],...
    'Orientation','horizontal');
exportgraphics(gcf, "lorenz_smoothl1.pdf",'ContentType','image')
