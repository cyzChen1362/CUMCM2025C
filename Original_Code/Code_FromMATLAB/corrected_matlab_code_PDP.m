% ========================================================================
% 分段非线性交互项模型PDP分析 - MATLAB代码
% 修正版本：所有函数定义已移至文件末尾
% 孕周分析范围：12-30周
% ========================================================================
%
% 模型方程：
% Y = β₀ + β₁×孕周 + β₂×BMI + β₃×分段项 + β₄×孕周² + β₅×BMI² + β₆×分段² + β₇×(孕周×BMI) + β₈×(分段×BMI)
% 其中分段项 = max(0, 孕周-20)
%
% 参数值：
% β₀ = 0.096479, β₁ = 0.013930, β₂ = -0.006134, β₃ = 0.029171
% β₄ = -0.000690, β₅ = 0.000005, β₆ = -0.000122
% β₇ = 0.000248, β₈ = -0.000414
%
% ========================================================================

clear all; 
clc;

%% ========================================================================
%% 主脚本部分 - 分段非线性交互项模型PDP分析
%% ========================================================================

% 设置模型参数（分段非线性交互项模型）
beta = [0.096479, 0.013930, -0.006134, 0.029171, -0.000690, ...
        0.000005, -0.000122, 0.000248, -0.000414];

% 生成示例数据用于PDP分析
n = 200;              % 样本数量
rng(42);              % 设置随机种子以确保结果可重复

% 生成孕周数据 (12-30周)
weeks = 12 + 18 * rand(n, 1);

% 生成BMI数据 (18-35)
bmi = 18 + 17 * rand(n, 1);

% 计算Y染色体浓度
y_concentration = arrayfun(@(w, b) predict_model(w, b, beta), weeks, bmi);

% 添加噪声
y_concentration = y_concentration + 0.01 * randn(n, 1);

% 输出数据信息
fprintf('数据生成完成：%d个样本\n', n);
fprintf('孕周范围：%.1f - %.1f 周\n', min(weeks), max(weeks));
fprintf('BMI范围：%.1f - %.1f\n', min(bmi), max(bmi));
fprintf('Y染色体浓度范围：%.4f - %.4f\n', min(y_concentration), max(y_concentration));

%% ========================================================================
%% PDP分析 - 孕周数的部分依赖
%% ========================================================================

fprintf('\n正在计算孕周数的部分依赖图...\n');

% 设置孕周范围 (12-30周)
weeks_range = linspace(12, 30, 50);
pdp_weeks = zeros(length(weeks_range), 1);

% 计算每个孕周值的平均预测
for i = 1:length(weeks_range)
    predictions = arrayfun(@(b) predict_model(weeks_range(i), b, beta), bmi);
    pdp_weeks(i) = mean(predictions);
end

%% ========================================================================
%% PDP分析 - BMI的部分依赖
%% ========================================================================

fprintf('正在计算BMI的部分依赖图...\n');

% 设置BMI范围
bmi_range = linspace(18, 35, 50);
pdp_bmi = zeros(length(bmi_range), 1);

% 计算每个BMI值的平均预测
for i = 1:length(bmi_range)
    predictions = arrayfun(@(w) predict_model(w, bmi_range(i), beta), weeks);
    pdp_bmi(i) = mean(predictions);
end

%% ========================================================================
%% 可视化结果
%% ========================================================================

% 创建图形窗口
figure('Position', [100, 100, 1200, 500]);

% 子图1：孕周数的PDP
subplot(1, 2, 1);
plot(weeks_range, pdp_weeks, 'b-', 'LineWidth', 2);
hold on;

% 标记分段点 (孕周=20)
xline(20, 'r--', '分段点', 'LineWidth', 1.5, 'FontSize', 10);
grid on;

xlabel('孕周数 (周)', 'FontSize', 12);
ylabel('Y染色体浓度预测值', 'FontSize', 12);
title('孕周数的部分依赖图', 'FontSize', 14, 'FontWeight', 'bold');
legend('PDP曲线', '分段点(20周)', 'Location', 'best');

% 子图2：BMI的PDP
subplot(1, 2, 2);
plot(bmi_range, pdp_bmi, 'g-', 'LineWidth', 2);
grid on;

xlabel('BMI', 'FontSize', 12);
ylabel('Y染色体浓度预测值', 'FontSize', 12);
title('BMI的部分依赖图', 'FontSize', 14, 'FontWeight', 'bold');

% 调整子图间距
sgtitle('分段非线性交互项模型 - 部分依赖分析', 'FontSize', 16, 'FontWeight', 'bold');

%% ========================================================================
%% 统计分析和结果输出
%% ========================================================================

fprintf('\n=== PDP分析结果 ===\n');

% 孕周数PDP分析
weeks_slope_early = (pdp_weeks(find(weeks_range <= 20, 1, 'last')) - pdp_weeks(1)) / ...
                    (weeks_range(find(weeks_range <= 20, 1, 'last')) - weeks_range(1));

weeks_slope_late = (pdp_weeks(end) - pdp_weeks(find(weeks_range >= 20, 1))) / ...
                   (weeks_range(end) - weeks_range(find(weeks_range >= 20, 1)));

fprintf('孕周数影响分析：\n');
fprintf(' 早期斜率 (12-20周): %.6f\n', weeks_slope_early);
fprintf(' 后期斜率 (20-30周): %.6f\n', weeks_slope_late);
fprintf(' 分段效应: %.6f\n', weeks_slope_late - weeks_slope_early);

% BMI PDP分析
bmi_slope = (pdp_bmi(end) - pdp_bmi(1)) / (bmi_range(end) - bmi_range(1));
bmi_curvature = mean(diff(diff(pdp_bmi)));  % 二阶差分的均值

fprintf('\nBMI影响分析：\n');
fprintf(' 整体斜率: %.6f\n', bmi_slope);
fprintf(' 曲率指标: %.8f\n', bmi_curvature);

% 预测范围
fprintf('\n预测值范围：\n');
fprintf(' 孕周PDP范围: %.6f - %.6f\n', min(pdp_weeks), max(pdp_weeks));
fprintf(' BMI PDP范围: %.6f - %.6f\n', min(pdp_bmi), max(pdp_bmi));

% 模型参数输出
fprintf('\n使用的模型参数：\n');
param_names = {'β₀(截距)', 'β₁(孕周)', 'β₂(BMI)', 'β₃(分段项)', ...
               'β₄(孕周²)', 'β₅(BMI²)', 'β₆(分段²)', 'β₇(孕周×BMI)', 'β₈(分段×BMI)'};

for i = 1:length(beta)
    fprintf(' %s = %.6f\n', param_names{i}, beta(i));
end

fprintf('\n分析完成！\n');

%% ========================================================================
%% 函数定义区域 (所有函数定义必须在文件末尾)
%% ========================================================================

function y_pred = predict_model(weeks, bmi, beta)
    % 分段非线性交互项模型预测函数
    %
    % 输入：
    %   weeks: 孕周数
    %   bmi: BMI值
    %   beta: 模型参数向量 [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇, β₈]
    %
    % 输出：
    %   y_pred: 预测的Y染色体浓度
    %
    % 模型方程：
    %   Y = β₀ + β₁×孕周 + β₂×BMI + β₃×分段项 + β₄×孕周² + β₅×BMI²
    %     + β₆×分段² + β₇×(孕周×BMI) + β₈×(分段×BMI)
    %   其中分段项 = max(0, 孕周-20)

    % 计算分段项
    segment = max(0, weeks - 20);

    % 提取参数
    b0 = beta(1);   % 截距
    b1 = beta(2);   % 孕周线性项
    b2 = beta(3);   % BMI线性项
    b3 = beta(4);   % 分段线性项
    b4 = beta(5);   % 孕周平方项
    b5 = beta(6);   % BMI平方项
    b6 = beta(7);   % 分段平方项
    b7 = beta(8);   % 孕周×BMI交互项
    b8 = beta(9);   % 分段×BMI交互项

    % 计算预测值
    y_pred = b0 + b1*weeks + b2*bmi + b3*segment + ...
             b4*weeks^2 + b5*bmi^2 + b6*segment^2 + ...
             b7*(weeks*bmi) + b8*(segment*bmi);
end