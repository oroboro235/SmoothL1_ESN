clear;clc;

data_name = "mg_";
WO_name = ["WO_ridge" "WO_ols" "WO_lasso_smoothl1" "WO_lasso_cd" "WO_lasso_lars"];
test_y_name = "test_y";
outputs_name = ["test_pred_ridge" "test_pred_ols" "test_pred_lasso_smoothl1" "test_pred_lasso_cd" "test_pred_lasso_lars"];



ridge_weights = table2array(readtable(data_name+WO_name(1)+".csv"));
ols_weights = table2array(readtable(data_name+WO_name(2)+".csv"));
lasso_smoothl1_weights = table2array(readtable(data_name+WO_name(3)+".csv"));
sparsity_smoothl1 = 1-(nnz(lasso_smoothl1_weights)/numel(lasso_smoothl1_weights));
lasso_cd_weights = table2array(readtable(data_name+WO_name(4)+".csv"));
sparsity_cd = 1-(nnz(lasso_cd_weights)/numel(lasso_cd_weights));
lasso_lars_weights = table2array(readtable(data_name+WO_name(5)+".csv"));
sparsity_lars = 1-(nnz(lasso_lars_weights)/numel(lasso_lars_weights));

test_y = table2array(readtable(data_name+test_y_name+".csv"));
ridge_outputs = table2array(readtable(data_name+outputs_name(1)+".csv"));
ols_outputs = table2array(readtable(data_name+outputs_name(2)+".csv"));
lasso_smoothl1_outputs = table2array(readtable(data_name+outputs_name(3)+".csv"));
lasso_cd_outputs = table2array(readtable(data_name+outputs_name(4)+".csv"));
lasso_lars_outputs = table2array(readtable(data_name+outputs_name(5)+".csv"));

raw_data = table2array(readtable("mg_t17.csv"));
close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,10,3])
plot(raw_data(1:1000))
% title("Mackey-Glass Series \tau=17")
xlabel("t")
ylabel("x")
exportgraphics(gcf, "mg_raw.pdf",'ContentType','vector')


timestep = 1:1:length(test_y);

close all
figure
set(gcf, 'Units','centimeters','Position',[10,5,20,15])
t = tiledlayout(5,1, "TileSpacing","compact");
nexttile
p1 = plot(timestep,test_y,'r', timestep,ols_outputs, 'b');
title("No regularization", "FontSize", 12)
nexttile
p2 = plot(timestep,test_y,'r', timestep,ridge_outputs, 'b');
title("L2 regularization", "FontSize", 12)
nexttile
p3 = plot(timestep,test_y,'r', timestep,lasso_cd_outputs, 'b');
title("L1 regularization - CD", "FontSize", 12)
nexttile
p4 = plot(timestep,test_y,'r', timestep,lasso_lars_outputs, 'b');
title("L1 regularization - LARS", "FontSize", 12)
nexttile
p5 = plot(timestep,test_y,'r', timestep,lasso_smoothl1_outputs, 'b');
title("L1 regularization - SmoothL1", "FontSize", 12)

xlabel(t,'t')
ylabel(t,'x')
cb = legend('True', 'Predict');
% % cb.Layout.Tile = "north";
cb.FontSize = 12;
cb.Orientation = "horizontal";
set(cb,'Position',[0.659410827786939 0.922409390126447 0.244708991437047 0.0440917096545878])

exportgraphics(gcf, "mg_outputs.pdf",'ContentType','vector')


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
text(15, 0.2, str)
nexttile
p4 = stem(lasso_lars_weights, 'filled',"MarkerSize",3);
title("L1 regularization - LARS (\lambda=1e-4)", "FontSize", 12)
str = ['sparsity = ' num2str(sparsity_lars)];
text(15, 0.4, str)
nexttile
p5 = stem(lasso_smoothl1_weights, 'filled',"MarkerSize",3);
title("L1 regularization - SmoothL1 (\lambda=1e-4)", "FontSize", 12)
str = ['sparsity = ' num2str(sparsity_smoothl1)];
text(15, -2, str)
xlabel(t,'weight index')
ylabel(t,'weight value')
exportgraphics(gcf, "mg_weights.pdf",'ContentType','vector')
