clear;clc;

data_name = "lorenz_";
WO_name = ["WO_ridge" "WO_ols" "WO_lasso_smoothl1" "WO_lasso_cd" "WO_lasso_lars"];
test_y_name = "test_y";
outputs_name = ["test_pred_ridge" "test_pred_ols" "test_pred_lasso_smoothl1" "test_pred_lasso_cd" "test_pred_lasso_lars"];

ridge_weights = table2array(readtable(data_name+WO_name(1)+".csv"));
ridge_weights = ridge_weights(:);
ols_weights = table2array(readtable(data_name+WO_name(2)+".csv"));
ols_weights = ols_weights(:);
lasso_smoothl1_weights = table2array(readtable(data_name+WO_name(3)+".csv"));
lasso_smoothl1_weights = lasso_smoothl1_weights(:);
sparsity_smoothl1 = 1-(nnz(lasso_smoothl1_weights)/numel(lasso_smoothl1_weights));
lasso_cd_weights = table2array(readtable(data_name+WO_name(4)+".csv"));
lasso_cd_weights = lasso_cd_weights(:);
sparsity_cd = 1-(nnz(lasso_cd_weights)/numel(lasso_cd_weights));
lasso_lars_weights = table2array(readtable(data_name+WO_name(5)+".csv"));
lasso_lars_weights = lasso_lars_weights(:);
sparsity_lars = 1-(nnz(lasso_lars_weights)/numel(lasso_lars_weights));

test_y = table2array(readtable(data_name+test_y_name+".csv"));
ridge_outputs = table2array(readtable(data_name+outputs_name(1)+".csv"));
ols_outputs = table2array(readtable(data_name+outputs_name(2)+".csv"));
lasso_smoothl1_outputs = table2array(readtable(data_name+outputs_name(3)+".csv"));
lasso_cd_outputs = table2array(readtable(data_name+outputs_name(4)+".csv"));
lasso_lars_outputs = table2array(readtable(data_name+outputs_name(5)+".csv"));

% raw lorenz
raw_lorenz = table2array(readtable("lorenz_full.csv"));
raw_lorenz_plot = raw_lorenz(1000:8000, :);
raw_dim1 = raw_lorenz_plot(:, 1);
raw_dim2 = raw_lorenz_plot(:, 2);
raw_dim3 = raw_lorenz_plot(:, 3);
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,15,10])
t = tiledlayout(3, 1, "TileSpacing","compact");
nexttile
plot(raw_dim1,'r')
title("Dimension x")
ylabel("x")
xlim([1 length(raw_lorenz_plot)])
nexttile
plot(raw_dim2,'g')
title("Dimension y")
ylabel("y")
xlim([1 length(raw_lorenz_plot)])
nexttile
plot(raw_dim3, 'b')
title("Dimension z")
ylabel("z")
xlim([1 length(raw_lorenz_plot)])
xlabel(t, "t")
exportgraphics(gcf, "raw_lorenz_sep.pdf",'ContentType','vector')

% 2D lorenz
close all
figure
plot(raw_lorenz(1000:8000,1), raw_lorenz(1000:8000,3), 'Color','red')
% set(gca, 'XTickLabels', [])
% set(gca, 'YTickLabels', [])
xlabel("x")
ylabel("z")
% title("Lorenz attractor")

exportgraphics(gcf, "raw_lorenz_attractor.pdf",'ContentType','vector')



timestep = 1:1:length(test_y);

% dim 1
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(5, 1, "TileSpacing","compact");
nexttile
plot(timestep, test_y(:,1), 'k', timestep,ols_outputs(:,1),'r')
title("No regularization", "FontSize", 12)
nexttile
plot(timestep, test_y(:,1), 'k', timestep,ridge_outputs(:,1), 'r')
title("L2 regularization", "FontSize", 12)
nexttile
plot(timestep, test_y(:,1), 'k', timestep,lasso_cd_outputs(:,1), 'r')
title("L1 regularization - CD", "FontSize", 12)
nexttile
plot(timestep, test_y(:,1), 'k', timestep,lasso_lars_outputs(:,1), 'r')
title("L1 regularization - LARS", "FontSize", 12)
nexttile
plot(timestep, test_y(:,1), 'k', timestep,lasso_smoothl1_outputs(:,1), 'r')
title("L1 regularization - SmoothL1", "FontSize", 12)

xlabel(t,'t')
ylabel(t,'x')
cb = legend('True', 'Predict');
% % cb.Layout.Tile = "north";
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])


exportgraphics(gcf, "lorenz_outputs_dim1.pdf",'ContentType','vector')

% dim 2
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(5, 1, "TileSpacing","compact");
nexttile
plot(timestep, test_y(:,2), 'k', timestep,ols_outputs(:,2),'g')
title("No regularization", "FontSize", 12)
nexttile
plot(timestep, test_y(:,2), 'k', timestep,ridge_outputs(:,2), 'g')
title("L2 regularization", "FontSize", 12)
nexttile
plot(timestep, test_y(:,2), 'k', timestep,lasso_cd_outputs(:,2), 'g')
title("L1 regularization - CD", "FontSize", 12)
nexttile
plot(timestep, test_y(:,2), 'k', timestep,lasso_lars_outputs(:,2), 'g')
title("L1 regularization - LARS", "FontSize", 12)
nexttile
plot(timestep, test_y(:,2), 'k', timestep,lasso_smoothl1_outputs(:,2), 'g')
title("L1 regularization - SmoothL1", "FontSize", 12)

xlabel(t,'t')
ylabel(t,'y')
cb = legend('True', 'Predict');
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])

exportgraphics(gcf, "lorenz_outputs_dim2.pdf",'ContentType','vector')

% dim 3
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(5, 1, "TileSpacing","compact");
nexttile
plot(timestep, test_y(:,3), 'k', timestep,ols_outputs(:,3),'b')
title("No regularization", "FontSize", 12)
nexttile
plot(timestep, test_y(:,3), 'k', timestep,ridge_outputs(:,3),'b')
title("L2 regularization", "FontSize", 12)
nexttile
plot(timestep, test_y(:,3), 'k', timestep,lasso_cd_outputs(:,3),'b')
title("L1 regularization - CD", "FontSize", 12)
nexttile
plot(timestep, test_y(:,3), 'k', timestep,lasso_lars_outputs(:,3),'b')
title("L1 regularization - LARS", "FontSize", 12)
nexttile
plot(timestep, test_y(:,3), 'k', timestep,lasso_smoothl1_outputs(:,3),'b')
title("L1 regularization - SmoothL1", "FontSize", 12)

xlabel(t,'t')
ylabel(t,'z')
cb = legend('True', 'Predict');
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])

exportgraphics(gcf, "lorenz_outputs_dim3.pdf",'ContentType','vector')

% weights
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(5,1, "TileSpacing","compact");
nexttile
p1 = stem(ols_weights, 'filled',"MarkerSize",3);
title("No regularization", "FontSize", 12)
nexttile
p2 = stem(ridge_weights, 'filled',"MarkerSize",3);
title("L2 regularization (\lambda=1e-4)", "FontSize", 12)
nexttile
p3 = stem(lasso_cd_weights, 'filled',"MarkerSize",3);
title("L1 regularization - CD (\lambda=1e-4)", "FontSize", 12)
str = ['sparsity = ' num2str(sparsity_cd)];
text(15, -10, str)
nexttile
p4 = stem(lasso_lars_weights, 'filled',"MarkerSize",3);
title("L1 regularization - LARS (\lambda=1e-4)", "FontSize", 12)
str = ['sparsity = ' num2str(sparsity_lars)];
text(15, 5, str)
nexttile
p5 = stem(lasso_smoothl1_weights, 'filled',"MarkerSize",3);
title("L1 regularization - SmoothL1 (\lambda=1e-4)", "FontSize", 12)
str = ['sparsity = ' num2str(sparsity_smoothl1)];
text(15, 7, str)
xlabel(t,'weight index')
ylabel(t,'weight value')
exportgraphics(gcf, "lorenz_weights.pdf",'ContentType','vector')
