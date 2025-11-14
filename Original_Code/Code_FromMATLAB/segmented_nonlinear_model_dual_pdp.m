% 分段非线性交互项模型分析和双特征偏依赖图(PDP)
% 孕周数范围调整为30周，包含双特征PDP (BMI+孕周数)的3D可视化

clear; clc; close all;

%% 模型参数定义
beta0 = 0.096479;    % 截距项
beta1 = 0.013930;    % 孕周系数
beta2 = -0.006134;   % BMI系数
beta3 = 0.029171;    % 分段项系数
beta4 = -0.000690;   % 孕周²系数
beta5 = 0.000005;    % BMI²系数
beta6 = -0.000122;   % 分段²系数
beta7 = 0.000248;    % 孕周×BMI交互系数
beta8 = -0.000414;   % 分段×BMI交互系数

% 显示模型方程
fprintf('分段非线性交互项模型方程:\n');
fprintf('Y = %.6f + %.6f×孕周 + %.6f×BMI + %.6f×分段项\n', beta0, beta1, beta2, beta3);
fprintf('  + %.6f×孕周² + %.6f×BMI² + %.6f×分段²\n', beta4, beta5, beta6);
fprintf('  + %.6f×(孕周×BMI) + %.6f×(分段×BMI)\n', beta7, beta8);
fprintf('其中分段项 = max(0, 孕周-20)\n\n');

%% 数据范围设定
% 调整孕周数范围到30周
gestational_age_range = [12, 30];  % 孕周范围 (调整为30周)
bmi_range = [18, 35];              % BMI范围

% 基准值设定 (用于PDP计算)
baseline_gestational_age = 24;  % 基准孕周
baseline_bmi = 25;              % 基准BMI

fprintf('分析参数:\n');
fprintf('孕周范围: %.0f - %.0f 周\n', gestational_age_range(1), gestational_age_range(2));
fprintf('BMI范围: %.1f - %.1f kg/m²\n', bmi_range(1), bmi_range(2));
fprintf('基准孕周: %.0f 周\n', baseline_gestational_age);
fprintf('基准BMI: %.1f kg/m²\n\n', baseline_bmi);

%% 双特征偏依赖图 (PDP) - BMI和孕周数同时考虑
fprintf('正在计算双特征PDP...\n');

% 创建网格
gestational_age_grid = linspace(gestational_age_range(1), gestational_age_range(2), 20);
bmi_grid = linspace(bmi_range(1), bmi_range(2), 20);
[GA_mesh, BMI_mesh] = meshgrid(gestational_age_grid, bmi_grid);

% 计算双特征PDP
pdp_2d = zeros(size(GA_mesh));
for i = 1:size(GA_mesh, 1)
    for j = 1:size(GA_mesh, 2)
        ga = GA_mesh(i, j);
        bmi = BMI_mesh(i, j);
        pdp_2d(i, j) = predict_outcome(ga, bmi, beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
    end
end

fprintf('双特征PDP计算完成\n');

%% 3D可视化 - 双特征偏依赖图
fprintf('正在生成3D可视化图表...\n');

% 创建图形窗口
figure('Position', [100, 100, 1200, 500]);

% 子图1: 3D表面图
subplot(1, 2, 1);
surf(GA_mesh, BMI_mesh, pdp_2d, 'FaceAlpha', 0.8, 'EdgeColor', 'none');
hold on;

% 添加等高线投影到底面
contour3(GA_mesh, BMI_mesh, pdp_2d, 15, 'LineWidth', 1.5);

title('双特征PDP - 3D表面图', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('孕周 (周)', 'FontSize', 12);
ylabel('BMI (kg/m²)', 'FontSize', 12);
zlabel('预测结果', 'FontSize', 12);
colorbar;
colormap(jet);
grid on;
view(45, 30);

% 设置坐标轴范围
xlim(gestational_age_range);
ylim(bmi_range);

% 子图2: 等高线图 (俯视图)
subplot(1, 2, 2);
[C, h] = contourf(GA_mesh, BMI_mesh, pdp_2d, 20);
clabel(C, h, 'FontSize', 10);
hold on;

% 标记基准点
plot(baseline_gestational_age, baseline_bmi, 'ro', 'MarkerSize', 10, ...
     'MarkerFaceColor', 'red', 'LineWidth', 2);
text(baseline_gestational_age + 0.5, baseline_bmi + 0.5, ...
     sprintf('基准点\n(%.0f周, %.1f)', baseline_gestational_age, baseline_bmi), ...
     'FontSize', 10, 'FontWeight', 'bold');

title('双特征PDP - 等高线图', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('孕周 (周)', 'FontSize', 12);
ylabel('BMI (kg/m²)', 'FontSize', 12);
colorbar;
grid on;

% 设置坐标轴范围
xlim(gestational_age_range);
ylim(bmi_range);

% 调整整体布局
sgtitle(sprintf('分段非线性交互项模型 '), ...
        'FontSize', 16, 'FontWeight', 'bold');

fprintf('3D可视化完成\n');

%% 结果分析和统计信息
fprintf('\n=== 分析结果汇总 ===\n');

% 计算PDP统计信息
pdp_min = min(pdp_2d(:));
pdp_max = max(pdp_2d(:));
pdp_range = pdp_max - pdp_min;

fprintf('双特征PDP结果范围:\n');
fprintf('  最小值: %.6f\n', pdp_min);
fprintf('  最大值: %.6f\n', pdp_max);
fprintf('  变化范围: %.6f\n', pdp_range);

% 找出最高和最低预测值对应的条件
[max_row, max_col] = find(pdp_2d == pdp_max, 1);
[min_row, min_col] = find(pdp_2d == pdp_min, 1);

fprintf('\n最高预测值条件:\n');
fprintf('  孕周: %.1f周, BMI: %.1f kg/m², 预测值: %.6f\n', ...
        GA_mesh(max_row, max_col), BMI_mesh(max_row, max_col), pdp_max);

fprintf('最低预测值条件:\n');
fprintf('  孕周: %.1f周, BMI: %.1f kg/m², 预测值: %.6f\n', ...
        GA_mesh(min_row, min_col), BMI_mesh(min_row, min_col), pdp_min);

% 基准点预测值
baseline_prediction = predict_outcome(baseline_gestational_age, baseline_bmi, ...
                                    beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
fprintf('\n基准点预测值:\n');
fprintf('  孕周: %.0f周, BMI: %.1f kg/m², 预测值: %.6f\n', ...
        baseline_gestational_age, baseline_bmi, baseline_prediction);

%% 交互效应分析
fprintf('\n=== 交互效应分析 ===\n');

% 分析不同BMI水平下孕周的影响
low_bmi = 20;   % 低BMI
high_bmi = 30;  % 高BMI

fprintf('固定BMI分析孕周影响:\n');
for test_bmi = [low_bmi, high_bmi]
    pred_12w = predict_outcome(12, test_bmi, beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
    pred_30w = predict_outcome(30, test_bmi, beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
    change = pred_30w - pred_12w;

    fprintf('  BMI %.0f: 12周预测值 %.6f, 30周预测值 %.6f, 变化 %.6f\n', ...
            test_bmi, pred_12w, pred_30w, change);
end

% 分析不同孕周下BMI的影响
early_ga = 15;  % 早期孕周
late_ga = 28;   % 晚期孕周

fprintf('\n固定孕周分析BMI影响:\n');
for test_ga = [early_ga, late_ga]
    pred_low_bmi = predict_outcome(test_ga, low_bmi, beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
    pred_high_bmi = predict_outcome(test_ga, high_bmi, beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
    change = pred_high_bmi - pred_low_bmi;

    fprintf('  孕周 %.0f: BMI20预测值 %.6f, BMI30预测值 %.6f, 变化 %.6f\n', ...
            test_ga, pred_low_bmi, pred_high_bmi, change);
end

fprintf('\n分析完成！\n');
fprintf('注意：本分析已移除BMI单特征PDP，专注于双特征交互分析\n');


%% ===============================
%% 函数定义部分（放在文件末尾）
%% ===============================

function y = predict_outcome(gestational_age, bmi, beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8)
    % 分段非线性交互项模型预测函数
    % 输入参数:
    %   gestational_age: 孕周 (周)
    %   bmi: 体重指数 (kg/m²)
    %   beta0-beta8: 模型系数
    % 输出:
    %   y: 预测结果

    % 计算分段项 (孕周超过20周的部分)
    segment_term = max(0, gestational_age - 20);

    % 计算各项
    linear_ga = beta1 * gestational_age;                    % 孕周线性项
    linear_bmi = beta2 * bmi;                              % BMI线性项
    linear_segment = beta3 * segment_term;                  % 分段线性项

    quadratic_ga = beta4 * (gestational_age^2);           % 孕周二次项
    quadratic_bmi = beta5 * (bmi^2);                      % BMI二次项
    quadratic_segment = beta6 * (segment_term^2);          % 分段二次项

    interaction_ga_bmi = beta7 * (gestational_age * bmi);  % 孕周×BMI交互项
    interaction_segment_bmi = beta8 * (segment_term * bmi); % 分段×BMI交互项

    % 完整模型方程
    y = beta0 + linear_ga + linear_bmi + linear_segment + ...
        quadratic_ga + quadratic_bmi + quadratic_segment + ...
        interaction_ga_bmi + interaction_segment_bmi;
end
