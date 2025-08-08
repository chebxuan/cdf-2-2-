%% 量子CDF(2,2)变换与经典CDF(2,2)变换对比分析
% 处理4.2.03.tiff图像并分析结果差异
% 作者：基于用户提供的CDF(2,2)小波变换量子提升方案

clear; clc; close all;

fprintf('='*60); fprintf('\n');
fprintf('量子CDF(2,2)变换与经典CDF(2,2)变换对比分析\n');
fprintf('='*60); fprintf('\n');

%% 经典CDF(2,2)多层分解函数
function [A3, D3, D2, D1] = classical_cdf_transform_multilevel(signal, levels)
    % 经典CDF(2,2)多层小波分解
    % 输入: signal - 输入信号, levels - 分解层数
    % 输出: A3 - 第3层近似系数, D3, D2, D1 - 各层详细系数
    
    if nargin < 2
        levels = 3;
    end
    
    % 初始化输出
    D1 = [];
    D2 = [];
    D3 = [];
    
    % 第1层分解
    [A1, D1] = classical_cdf_transform(signal);
    
    if levels >= 2
        % 第2层分解（对A1进行分解）
        [A2, D2] = classical_cdf_transform(A1);
        
        if levels >= 3
            % 第3层分解（对A2进行分解）
            [A3, D3] = classical_cdf_transform(A2);
        else
            A3 = A2;
        end
    else
        A3 = A1;
    end
end

%% 量子模拟CDF(2,2)多层分解函数
function [A3, D3, D2, D1] = quantum_simulated_cdf_transform_multilevel(signal, levels)
    % 量子模拟CDF(2,2)多层小波分解
    % 输入: signal - 输入信号, levels - 分解层数
    % 输出: A3 - 第3层近似系数, D3, D2, D1 - 各层详细系数
    
    if nargin < 2
        levels = 3;
    end
    
    % 初始化输出
    D1 = [];
    D2 = [];
    D3 = [];
    
    % 第1层分解
    [A1, D1] = quantum_simulated_cdf_transform(signal);
    
    if levels >= 2
        % 第2层分解（对A1进行分解）
        [A2, D2] = quantum_simulated_cdf_transform(A1);
        
        if levels >= 3
            % 第3层分解（对A2进行分解）
            [A3, D3] = quantum_simulated_cdf_transform(A2);
        else
            A3 = A2;
        end
    else
        A3 = A1;
    end
end

%% 多层分解对比函数
function result = compare_multilevel_transforms(signal)
    % 对比经典和量子模拟多层变换
    % 经典多层变换
    tic;
    [A3_classical, D3_classical, D2_classical, D1_classical] = classical_cdf_transform_multilevel(signal, 3);
    classical_time = toc;
    
    % 量子模拟多层变换
    tic;
    [A3_quantum, D3_quantum, D2_quantum, D1_quantum] = quantum_simulated_cdf_transform_multilevel(signal, 3);
    quantum_time = toc;
    
    % 计算各层差异
    A3_diff = abs(A3_classical - A3_quantum);
    D3_diff = abs(D3_classical - D3_quantum);
    D2_diff = abs(D2_classical - D2_quantum);
    D1_diff = abs(D1_classical - D1_quantum);
    
    result = struct();
    result.classical = struct('A3', A3_classical, 'D3', D3_classical, 'D2', D2_classical, 'D1', D1_classical, 'time', classical_time);
    result.quantum = struct('A3', A3_quantum, 'D3', D3_quantum, 'D2', D2_quantum, 'D1', D1_quantum, 'time', quantum_time);
    result.differences = struct('A3_diff', A3_diff, 'D3_diff', D3_diff, 'D2_diff', D2_diff, 'D1_diff', D1_diff, ...
        'max_A3_diff', max(A3_diff), 'max_D3_diff', max(D3_diff), 'max_D2_diff', max(D2_diff), 'max_D1_diff', max(D1_diff), ...
        'mean_A3_diff', mean(A3_diff), 'mean_D3_diff', mean(D3_diff), 'mean_D2_diff', mean(D2_diff), 'mean_D1_diff', mean(D1_diff));
end

%% 可视化多层分解结果
function visualize_multilevel_results(result)
    % 可视化多层分解结果
    try
        figure('Position', [100, 100, 1400, 1000]);
        
        % 1. 第3层近似系数对比
        subplot(3, 3, 1);
        scatter(result.classical.A3, result.quantum.A3, 'b', 'filled', 'Alpha', 0.6);
        hold on;
        min_val = min([result.classical.A3, result.quantum.A3]);
        max_val = max([result.classical.A3, result.quantum.A3]);
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
        xlabel('经典A3');
        ylabel('量子模拟A3');
        title('第3层近似系数对比');
        legend('数据点', '理想线', 'Location', 'best');
        grid on;
        
        % 2. 第1层详细系数对比
        subplot(3, 3, 2);
        scatter(result.classical.D1, result.quantum.D1, 'g', 'filled', 'Alpha', 0.6);
        hold on;
        min_val = min([result.classical.D1, result.quantum.D1]);
        max_val = max([result.classical.D1, result.quantum.D1]);
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
        xlabel('经典D1');
        ylabel('量子模拟D1');
        title('第1层详细系数对比');
        legend('数据点', '理想线', 'Location', 'best');
        grid on;
        
        % 3. 第2层详细系数对比
        subplot(3, 3, 3);
        scatter(result.classical.D2, result.quantum.D2, 'm', 'filled', 'Alpha', 0.6);
        hold on;
        min_val = min([result.classical.D2, result.quantum.D2]);
        max_val = max([result.classical.D2, result.quantum.D2]);
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
        xlabel('经典D2');
        ylabel('量子模拟D2');
        title('第2层详细系数对比');
        legend('数据点', '理想线', 'Location', 'best');
        grid on;
        
        % 4. 第3层详细系数对比
        subplot(3, 3, 4);
        scatter(result.classical.D3, result.quantum.D3, 'c', 'filled', 'Alpha', 0.6);
        hold on;
        min_val = min([result.classical.D3, result.quantum.D3]);
        max_val = max([result.classical.D3, result.quantum.D3]);
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
        xlabel('经典D3');
        ylabel('量子模拟D3');
        title('第3层详细系数对比');
        legend('数据点', '理想线', 'Location', 'best');
        grid on;
        
        % 5. 各层差异分布
        subplot(3, 3, 5);
        histogram(result.differences.A3_diff, 20, 'FaceColor', 'blue', 'Alpha', 0.7);
        xlabel('差异值');
        ylabel('频次');
        title('A3差异分布');
        grid on;
        
        subplot(3, 3, 6);
        histogram(result.differences.D1_diff, 20, 'FaceColor', 'green', 'Alpha', 0.7);
        xlabel('差异值');
        ylabel('频次');
        title('D1差异分布');
        grid on;
        
        subplot(3, 3, 7);
        histogram(result.differences.D2_diff, 20, 'FaceColor', 'magenta', 'Alpha', 0.7);
        xlabel('差异值');
        ylabel('频次');
        title('D2差异分布');
        grid on;
        
        subplot(3, 3, 8);
        histogram(result.differences.D3_diff, 20, 'FaceColor', 'cyan', 'Alpha', 0.7);
        xlabel('差异值');
        ylabel('频次');
        title('D3差异分布');
        grid on;
        
        % 9. 性能对比
        subplot(3, 3, 9);
        bar([result.classical.time, result.quantum.time]);
        set(gca, 'XTickLabel', {'经典', '量子模拟'});
        ylabel('时间 (秒)');
        title('性能对比');
        grid on;
        
        sgtitle('量子CDF(2,2)三层分解与经典CDF(2,2)三层分解对比分析', 'FontSize', 14, 'FontWeight', 'bold');
        
        % 保存图像
        saveas(gcf, 'quantum_vs_classical_multilevel_comparison_matlab.png');
        fprintf('多层分解对比图已保存为 quantum_vs_classical_multilevel_comparison_matlab.png\n');
        
    catch ME
        fprintf('多层分解可视化失败: %s\n', ME.message);
    end
end

%% 经典CDF(2,2)小波变换函数
function [approx, detail] = classical_cdf_transform(signal)
    % 经典CDF(2,2)小波变换实现
    signal = double(signal);
    n = length(signal);
    
    % Step 1: Split
    even = signal(1:2:end);  % S(2i)
    odd = signal(2:2:end);   % S(2i+1)
    
    % Step 2: Predict - P(S) = 1/2[S(2i) + S(2i+2)]
    predict = zeros(size(odd));
    for i = 1:length(odd)
        if i == 1
            % 边界处理
            predict(i) = even(1);
        elseif i == length(odd)
            % 边界处理
            predict(i) = even(end);
        else
            predict(i) = 0.5 * (even(i) + even(i+1));
        end
    end
    
    % 计算详细系数 D(i) = S(2i+1) - P(S)
    detail = odd - predict;
    
    % Step 3: Update - W(D) = 1/4[D(i-1) + D(i)]
    update = zeros(size(even));
    for i = 1:length(even)
        if i == 1
            % 边界处理
            if ~isempty(detail)
                update(i) = 0.25 * detail(1);
            end
        elseif i == length(even)
            % 边界处理
            if i-1 <= length(detail)
                update(i) = 0.25 * detail(i-1);
            end
        else
            if i-1 <= length(detail) && i <= length(detail)
                update(i) = 0.25 * (detail(i-1) + detail(i));
            end
        end
    end
    
    % 计算近似系数 A(i) = S(2i) + W(D)
    approx = even + update;
end

%% 量子模拟CDF(2,2)变换函数
function [approx, detail] = quantum_simulated_cdf_transform(signal)
    % 量子模拟CDF(2,2)变换
    % 使用经典算法模拟量子计算过程
    signal = double(signal);
    n = length(signal);
    
    % 量子模拟：添加一些量子噪声和精度限制
    quantum_precision = 8;  % 量子比特精度
    max_quantum_value = 2^quantum_precision - 1;
    
    % Step 1: Split (量子分离)
    even = signal(1:2:end);
    odd = signal(2:2:end);
    
    % 量子精度限制
    even = min(max(even, 0), max_quantum_value);
    odd = min(max(odd, 0), max_quantum_value);
    
    % Step 2: Predict (量子预测)
    predict = zeros(size(odd));
    for i = 1:length(odd)
        if i == 1
            predict_val = even(1);
        elseif i == length(odd)
            predict_val = even(end);
        else
            predict_val = 0.5 * (even(i) + even(i+1));
        end
        
        % 量子精度限制
        predict_val = min(max(predict_val, 0), max_quantum_value);
        predict(i) = predict_val;
    end
    
    % 计算详细系数
    detail = odd - predict;
    
    % Step 3: Update (量子更新)
    update = zeros(size(even));
    for i = 1:length(even)
        if i == 1
            if ~isempty(detail)
                update_val = 0.25 * detail(1);
            else
                update_val = 0;
            end
        elseif i == length(even)
            if i-1 <= length(detail)
                update_val = 0.25 * detail(i-1);
            else
                update_val = 0;
            end
        else
            if i-1 <= length(detail) && i <= length(detail)
                update_val = 0.25 * (detail(i-1) + detail(i));
            else
                update_val = 0;
            end
        end
        
        % 量子精度限制
        update_val = min(max(update_val, 0), max_quantum_value);
        update(i) = update_val;
    end
    
    % 计算近似系数
    approx = even + update;
end

%% 图像加载和预处理函数
function image = load_and_preprocess_image(image_path)
    % 加载和预处理图像
    fprintf('尝试加载图像: %s\n', image_path);
    
    try
        % 尝试加载图像
        if exist(image_path, 'file')
            image = imread(image_path);
            
            % 转换为灰度图
            if size(image, 3) == 3
                image = rgb2gray(image);
            end
            
            fprintf('✓ 成功加载图像: %s\n', mat2str(size(image)));
            fprintf('像素值范围: [%d, %d]\n', min(image(:)), max(image(:)));
        else
            error('图像文件不存在');
        end
        
    catch ME
        fprintf('加载图像失败: %s\n', ME.message);
        % 创建测试图像
        image = uint8([
            1, 3, 2, 1;
            2, 6, 6, 6;
            5, 2, 7, 4;
            3, 1, 7, 2
        ]);
        fprintf('使用测试图像\n');
    end
end

%% 图像块处理函数
function [blocks, positions] = process_image_blocks(image, block_size)
    % 将图像分割为块并处理
    if nargin < 2
        block_size = [2, 2];
    end
    
    [h, w] = size(image);
    blocks = {};
    positions = {};
    block_count = 0;
    
    for i = 1:block_size(1):h
        for j = 1:block_size(2):w
            if i + block_size(1) - 1 <= h && j + block_size(2) - 1 <= w
                block = image(i:i+block_size(1)-1, j:j+block_size(2)-1);
                block_count = block_count + 1;
                blocks{block_count} = block(:)';  % 展平为行向量
                positions{block_count} = [i, j];
            end
        end
    end
end

%% 对比变换函数
function result = compare_transforms(signal)
    % 对比经典和量子模拟变换
    % 经典变换
    tic;
    [classical_approx, classical_detail] = classical_cdf_transform(signal);
    classical_time = toc;
    
    % 量子模拟变换
    tic;
    [quantum_approx, quantum_detail] = quantum_simulated_cdf_transform(signal);
    quantum_time = toc;
    
    % 计算差异
    approx_diff = abs(classical_approx - quantum_approx);
    detail_diff = abs(classical_detail - quantum_detail);
    
    result = struct();
    result.classical = struct('approx', classical_approx, 'detail', classical_detail, 'time', classical_time);
    result.quantum = struct('approx', quantum_approx, 'detail', quantum_detail, 'time', quantum_time);
    result.differences = struct('approx_diff', approx_diff, 'detail_diff', detail_diff, ...
        'max_approx_diff', max(approx_diff), 'max_detail_diff', max(detail_diff), ...
        'mean_approx_diff', mean(approx_diff), 'mean_detail_diff', mean(detail_diff));
end

%% 结果分析函数
function stats = analyze_results(all_results)
    % 分析结果
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('结果分析\n');
    fprintf('%s\n', repmat('=', 1, 60));
    
    % 收集所有差异数据
    all_approx_diffs = [];
    all_detail_diffs = [];
    total_classical_time = 0;
    total_quantum_time = 0;
    
    for i = 1:length(all_results)
        all_approx_diffs = [all_approx_diffs, all_results{i}.differences.approx_diff];
        all_detail_diffs = [all_detail_diffs, all_results{i}.differences.detail_diff];
        total_classical_time = total_classical_time + all_results{i}.classical.time;
        total_quantum_time = total_quantum_time + all_results{i}.quantum.time;
    end
    
    if ~isempty(all_approx_diffs)
        fprintf('近似系数差异统计:\n');
        fprintf('  最大差异: %.6f\n', max(all_approx_diffs));
        fprintf('  最小差异: %.6f\n', min(all_approx_diffs));
        fprintf('  平均差异: %.6f\n', mean(all_approx_diffs));
        fprintf('  标准差: %.6f\n', std(all_approx_diffs));
    end
    
    if ~isempty(all_detail_diffs)
        fprintf('详细系数差异统计:\n');
        fprintf('  最大差异: %.6f\n', max(all_detail_diffs));
        fprintf('  最小差异: %.6f\n', min(all_detail_diffs));
        fprintf('  平均差异: %.6f\n', mean(all_detail_diffs));
        fprintf('  标准差: %.6f\n', std(all_detail_diffs));
    end
    
    % 成功率分析
    success_threshold = 0.01;  % 1%的差异阈值
    approx_success = sum(all_approx_diffs <= success_threshold);
    detail_success = sum(all_detail_diffs <= success_threshold);
    
    fprintf('\n成功率分析 (差异 <= %.3f):\n', success_threshold);
    fprintf('  近似系数成功率: %d/%d (%.1f%%)\n', approx_success, length(all_approx_diffs), ...
        100*approx_success/length(all_approx_diffs));
    fprintf('  详细系数成功率: %d/%d (%.1f%%)\n', detail_success, length(all_detail_diffs), ...
        100*detail_success/length(all_detail_diffs));
    
    fprintf('\n性能对比:\n');
    fprintf('  经典变换总时间: %.6fs\n', total_classical_time);
    fprintf('  量子模拟总时间: %.6fs\n', total_quantum_time);
    fprintf('  时间比率: %.2fx\n', total_quantum_time/total_classical_time);
    
    % 判断量子版本是否成功
    approx_success_flag = mean(all_approx_diffs) <= success_threshold;
    detail_success_flag = mean(all_detail_diffs) <= success_threshold;
    
    fprintf('\n量子版本成功性评估:\n');
    if approx_success_flag && detail_success_flag
        fprintf('🎉 量子版本非常成功！差异很小，精度很高。\n');
    elseif approx_success_flag || detail_success_flag
        fprintf('✅ 量子版本基本成功，部分指标达到要求。\n');
    else
        fprintf('⚠️ 量子版本需要改进，差异较大。\n');
    end
    
    stats = struct();
    stats.approx_stats = struct('mean', mean(all_approx_diffs), ...
        'success_rate', 100*approx_success/length(all_approx_diffs));
    stats.detail_stats = struct('mean', mean(all_detail_diffs), ...
        'success_rate', 100*detail_success/length(all_detail_diffs));
    stats.performance = struct('classical_time', total_classical_time, ...
        'quantum_time', total_quantum_time);
end

%% 可视化对比结果
function visualize_comparison(all_results)
    % 可视化对比结果
    try
        % 收集数据
        classical_approx_all = [];
        quantum_approx_all = [];
        classical_detail_all = [];
        quantum_detail_all = [];
        
        for i = 1:length(all_results)
            classical_approx_all = [classical_approx_all, all_results{i}.classical.approx];
            quantum_approx_all = [quantum_approx_all, all_results{i}.quantum.approx];
            classical_detail_all = [classical_detail_all, all_results{i}.classical.detail];
            quantum_detail_all = [quantum_detail_all, all_results{i}.quantum.detail];
        end
        
        % 创建图形
        figure('Position', [100, 100, 1200, 800]);
        
        % 1. 近似系数对比
        subplot(2, 2, 1);
        scatter(classical_approx_all, quantum_approx_all, 'b', 'filled', 'Alpha', 0.6);
        hold on;
        min_val = min([classical_approx_all, quantum_approx_all]);
        max_val = max([classical_approx_all, quantum_approx_all]);
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
        xlabel('经典近似系数');
        ylabel('量子模拟近似系数');
        title('近似系数对比');
        legend('数据点', '理想线', 'Location', 'best');
        grid on;
        
        % 2. 详细系数对比
        subplot(2, 2, 2);
        scatter(classical_detail_all, quantum_detail_all, 'g', 'filled', 'Alpha', 0.6);
        hold on;
        min_val = min([classical_detail_all, quantum_detail_all]);
        max_val = max([classical_detail_all, quantum_detail_all]);
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
        xlabel('经典详细系数');
        ylabel('量子模拟详细系数');
        title('详细系数对比');
        legend('数据点', '理想线', 'Location', 'best');
        grid on;
        
        % 3. 差异分布
        approx_diffs = abs(classical_approx_all - quantum_approx_all);
        detail_diffs = abs(classical_detail_all - quantum_detail_all);
        
        subplot(2, 2, 3);
        histogram(approx_diffs, 20, 'FaceColor', 'blue', 'Alpha', 0.7);
        xlabel('差异值');
        ylabel('频次');
        title('近似系数差异分布');
        grid on;
        
        subplot(2, 2, 4);
        histogram(detail_diffs, 20, 'FaceColor', 'orange', 'Alpha', 0.7);
        xlabel('差异值');
        ylabel('频次');
        title('详细系数差异分布');
        grid on;
        
        sgtitle('量子CDF(2,2)变换与经典CDF(2,2)变换对比分析', 'FontSize', 14, 'FontWeight', 'bold');
        
        % 保存图像
        saveas(gcf, 'quantum_vs_classical_comparison_matlab.png');
        fprintf('对比图已保存为 quantum_vs_classical_comparison_matlab.png\n');
        
    catch ME
        fprintf('可视化失败: %s\n', ME.message);
    end
end

%% 主程序
% 加载图像
image_path = '4.2.03.tiff';
if ~exist(image_path, 'file')
    fprintf('图像文件 %s 不存在，使用测试图像\n', image_path);
    image_path = [];
end

image = load_and_preprocess_image(image_path);

% 处理图像块
[blocks, positions] = process_image_blocks(image);
fprintf('总共 %d 个2x2块\n', length(blocks));

% 分析每个块
all_results = {};

for i = 1:length(blocks)
    fprintf('\n--- 块 %d (位置 [%d, %d]) ---\n', i, positions{i}(1), positions{i}(2));
    fprintf('原始数据: [%s]\n', num2str(blocks{i}, '%.1f '));
    
    % 对比变换
    result = compare_transforms(blocks{i});
    
    fprintf('经典变换:\n');
    fprintf('  近似系数: [%s]\n', num2str(result.classical.approx, '%.3f '));
    fprintf('  详细系数: [%s]\n', num2str(result.classical.detail, '%.3f '));
    fprintf('  处理时间: %.6fs\n', result.classical.time);
    
    fprintf('量子模拟变换:\n');
    fprintf('  近似系数: [%s]\n', num2str(result.quantum.approx, '%.3f '));
    fprintf('  详细系数: [%s]\n', num2str(result.quantum.detail, '%.3f '));
    fprintf('  处理时间: %.6fs\n', result.quantum.time);
    
    fprintf('差异分析:\n');
    fprintf('  近似系数最大差异: %.6f\n', result.differences.max_approx_diff);
    fprintf('  详细系数最大差异: %.6f\n', result.differences.max_detail_diff);
    fprintf('  近似系数平均差异: %.6f\n', result.differences.mean_approx_diff);
    fprintf('  详细系数平均差异: %.6f\n', result.differences.mean_detail_diff);
    
    all_results{i} = result;
end

% 分析结果
stats = analyze_results(all_results);

% 可视化结果
visualize_comparison(all_results);

%% 三层分解测试
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('三层分解测试\n');
fprintf('%s\n', repmat('=', 1, 60));

% 创建测试一维信号
test_signal = [1, 3, 2, 1, 2, 6, 6, 6, 5, 2, 7, 4, 3, 1, 7, 2];
fprintf('测试一维信号 (长度 %d): [%s]\n', length(test_signal), num2str(test_signal, '%.1f '));

% 三层分解对比
multilevel_result = compare_multilevel_transforms(test_signal);

fprintf('\n三层分解结果:\n');
fprintf('经典版本:\n');
fprintf('  A3 (第3层近似): [%s]\n', num2str(multilevel_result.classical.A3, '%.3f '));
fprintf('  D1 (第1层详细): [%s]\n', num2str(multilevel_result.classical.D1, '%.3f '));
fprintf('  D2 (第2层详细): [%s]\n', num2str(multilevel_result.classical.D2, '%.3f '));
fprintf('  D3 (第3层详细): [%s]\n', num2str(multilevel_result.classical.D3, '%.3f '));
fprintf('  处理时间: %.6fs\n', multilevel_result.classical.time);

fprintf('量子模拟版本:\n');
fprintf('  A3 (第3层近似): [%s]\n', num2str(multilevel_result.quantum.A3, '%.3f '));
fprintf('  D1 (第1层详细): [%s]\n', num2str(multilevel_result.quantum.D1, '%.3f '));
fprintf('  D2 (第2层详细): [%s]\n', num2str(multilevel_result.quantum.D2, '%.3f '));
fprintf('  D3 (第3层详细): [%s]\n', num2str(multilevel_result.quantum.D3, '%.3f '));
fprintf('  处理时间: %.6fs\n', multilevel_result.quantum.time);

fprintf('三层分解差异分析:\n');
fprintf('  A3最大差异: %.6f, 平均差异: %.6f\n', multilevel_result.differences.max_A3_diff, multilevel_result.differences.mean_A3_diff);
fprintf('  D1最大差异: %.6f, 平均差异: %.6f\n', multilevel_result.differences.max_D1_diff, multilevel_result.differences.mean_D1_diff);
fprintf('  D2最大差异: %.6f, 平均差异: %.6f\n', multilevel_result.differences.max_D2_diff, multilevel_result.differences.mean_D2_diff);
fprintf('  D3最大差异: %.6f, 平均差异: %.6f\n', multilevel_result.differences.max_D3_diff, multilevel_result.differences.mean_D3_diff);

% 可视化三层分解结果
visualize_multilevel_results(multilevel_result);

fprintf('\n分析完成！\n');
fprintf('量子版本精度: %.1f%% (近似系数), %.1f%% (详细系数)\n', ...
    stats.approx_stats.success_rate, stats.detail_stats.success_rate);

%% 结论和建议
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('结论和建议\n');
fprintf('%s\n', repmat('=', 1, 60));

if stats.approx_stats.success_rate > 95 && stats.detail_stats.success_rate > 95
    fprintf('✓ 量子实现精度很高，可以用于实际应用\n');
elseif stats.approx_stats.success_rate > 80 && stats.detail_stats.success_rate > 80
    fprintf('✓ 量子实现精度良好，适合进一步优化\n');
else
    fprintf('⚠️ 建议优化量子实现，提高精度\n');
end

fprintf('\n主要发现:\n');
fprintf('1. 量子模拟与经典实现的核心算法相同\n');
fprintf('2. 主要差异来自量子精度限制和数值处理\n');
fprintf('3. 差异大小反映了量子实现的准确性\n');
fprintf('4. 成功率是评估量子版本成功的关键指标\n'); 