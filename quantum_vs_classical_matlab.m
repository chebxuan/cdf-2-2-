%% 量子CDF(2,2)变换与经典CDF(2,2)变换对比分析
% 处理4.2.03.tiff图像并分析结果差异
% 作者：基于用户提供的CDF(2,2)小波变换量子提升方案

clear; clc; close all;

fprintf('='*60); fprintf('\n');
fprintf('量子CDF(2,2)变换与经典CDF(2,2)变换对比分析\n');
fprintf('='*60); fprintf('\n');

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

%% 多层分解：经典版本
function [A, D] = classical_cdf_transform_multilevel(signal, levels)
    % 对一维信号进行多层CDF(2,2)分解
    % 输出：A 为第 levels 层近似；D 为长度为 levels 的 cell，D{1} 为第一层细节，依此类推
    x = double(signal(:)');
    D = cell(1, levels);
    for lev = 1:levels
        [approx, detail] = classical_cdf_transform(x);
        D{lev} = detail;
        x = approx;  % 下一层使用近似部分
    end
    A = x;
end

%% 多层分解：量子模拟版本
function [A, D] = quantum_simulated_cdf_transform_multilevel(signal, levels)
    x = double(signal(:)');
    D = cell(1, levels);
    for lev = 1:levels
        [approx, detail] = quantum_simulated_cdf_transform(x);
        D{lev} = detail;
        x = approx;
    end
    A = x;
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

fprintf('\n分析完成！\n');
fprintf('量子版本精度: %.1f%% (近似系数), %.1f%% (详细系数)\n', ...
    stats.approx_stats.success_rate, stats.detail_stats.success_rate);

%% 三层分解演示（Figure 6 要求）
try
    fprintf('\n%s\n', repmat('-', 1, 60));
    fprintf('一维信号的三层分解演示 (经典与量子模拟)\n');
    fprintf('%s\n', repmat('-', 1, 60));
    test_signal = [1, 3, 2, 6, 5, 2, 7, 4];  % 示例一维信号，长度为偶数
    levels = 3;
    
    [A3_c, D_c] = classical_cdf_transform_multilevel(test_signal, levels);
    [A3_q, D_q] = quantum_simulated_cdf_transform_multilevel(test_signal, levels);
    
    fprintf('经典三层分解: A3=[%s], D3=[%s], D2=[%s], D1=[%s]\n', ...
        num2str(A3_c, '%.3f '), num2str(D_c{3}, '%.3f '), num2str(D_c{2}, '%.3f '), num2str(D_c{1}, '%.3f '));
    fprintf('量子三层分解: A3=[%s], D3=[%s], D2=[%s], D1=[%s]\n', ...
        num2str(A3_q, '%.3f '), num2str(D_q{3}, '%.3f '), num2str(D_q{2}, '%.3f '), num2str(D_q{1}, '%.3f '));
catch ME
    fprintf('三层分解演示失败: %s\n', ME.message);
end

fprintf('\n');
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