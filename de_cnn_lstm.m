%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

d = 5;
res = xlsread('Z1.xlsx');

P_train = res(1: 800, 1: 5)';
T_train = res(1: 800, 7)';
M = size(P_train, 2);

P_test = res(801: end, 1: 5)';
T_test = res(801: end, 7)';
N = size(P_test, 2);


%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
P_train =  double(reshape(P_train, d, 1, 1, M));
P_test  =  double(reshape(P_test , d, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end


% 设置DE算法参数
F = 0.1;   % 变异因子
CR = 0.7;  % 交叉概率
pop_size = 3; % 种群大小
max_gen = 20; % 最大迭代次数

n = 2; % 优化参数个数
lb = [5, 1e-3];                 % 参数取值下界(LSTM层节点, 学习率)
ub = [50, 1e-2];                 % 参数取值上界(LSTM层节点, 学习率)
pop = repmat(lb, pop_size, 1) + rand(pop_size, n) .* repmat(ub-lb, pop_size, 1);

optimisedLog = zeros(20, 2);

% 进化主循环
for i = 1:max_gen
    fprintf('Generation %d start.', i)
    % 开始计时
    tic
    % 计算种群适应度值
    fitness = zeros(1, pop_size);
    for j = 1:pop_size
        fitness(j) = lstm_fitness(pop(j,:), p_train, t_train, p_test, ps_output, T_test, N, d);
    end
    
    % 打印当前代数和最优解信息
    [best_fitness, best_idx] = min(fitness);
    
    format long;
    fprintf('Generation %d: Best Fitness = %.4f, Best Solution = [%.0f, %.5f]\n', i, best_fitness, pop(best_idx,:));
    optimisedLog(i, :) = pop(best_idx,:);
%     disp(optimisedLog(i, :))
    t = toc;
    disp(['代码执行时间为：' num2str(t) ' 秒']);
    
    
    % 生成新的种群
    new_pop = zeros(pop_size, n);
    for j = 1:pop_size
        % 选择三个不同的父代个体
        idx = randperm(pop_size, 3);
        x1 = pop(idx(1), :);
        x2 = pop(idx(2), :);
        x3 = pop(idx(3), :);
        
        % 变异操作
        v = x1 + F * (x2 - x3);
        v = min(max(v, lb), ub);
        
        % 交叉操作
        jrand = randi(n);
        u = zeros(1, n);
        for k = 1:n
            if rand() <= CR || k == jrand
                u(k) = v(k);
            else
                u(k) = pop(j,k);
            end
        end
        
        % 更新种群
        if lstm_fitness(u, p_train, t_train, p_test, ps_output, T_test, N, d) < fitness(j)
            new_pop(j,:) = u;
        else
            new_pop(j,:) = pop(j,:);
        end
    end
    
    % 更新种群
    pop = new_pop;
end

% 计算最优解和适应度值
[best_fitness, best_idx] = min(fitness);
best_solution = pop(best_idx,:);
fprintf('Best Fitness = %.4f, Best Solution = [%d,%d,%.4f,%.4f]\n', best_fitness, best_solution);

% 将这个值保存到一个文件中
save('results/optimisedLog.mat', 'optimisedLog');


% CNN-LSTM 适应度函数
function fitness = lstm_fitness(x, p_train, t_train, p_test, ps_output, T_test, N, d)
    % 构建 LSTM 模型
    numUnits = int32(x(1));
    learnRate = x(2);
  
  %%  建立模型
    lgraph = layerGraph();                                                 % 建立空白网络结构
    tempLayers = [
        sequenceInputLayer([d, 1, 1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
        sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
    lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
    tempLayers = [
        %convolution2dLayer([2, 1], 16, 'Stride', [1, 1], "Name", "conv_1")
        convolution2dLayer([2, 1], 16, 'Padding', 'same', "Name", "conv_1") % 建立卷积层,卷积核大小[2, 1]，16个特征图
%         batchNormalizationLayer('Name', 'batch_1')
        reluLayer("Name", "relu_1")                                        % Relu 激活层

        maxPooling2dLayer([2, 1], 'Stride', [1, 1], 'Name', 'pool_1')

        convolution2dLayer([2, 1], 32, 'Padding', 'same', "Name", "conv_2") % 建立卷积层,卷积核大小[2, 1]，32个特征图
%         batchNormalizationLayer('Name', 'batch_2')
        reluLayer("Name", "relu_2")];                                      % Relu 激活层
    lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
    tempLayers = [
        sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
        flattenLayer("Name", "flatten")                                    % 网络铺平层
        lstmLayer(numUnits, "Name", "lstm", "OutputMode", "last")                 % LSTM层
        %dropoutLayer(0.05, 'Name', 'drop_1')
        fullyConnectedLayer(1, "Name", "fc")                          % 全连接层
        regressionLayer("Name", "regressionoutput")];                      % 回归层

    lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
    lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入
    lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                           % 折叠层输出 连接 反折叠层输入          
    lgraph = connectLayers(lgraph, "relu_2", "sequnfold/in");              % 激活层输出 连接 反折叠层输入
    
    options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 100, ...                 % 最大迭代次数
    'InitialLearnRate', learnRate, ...          % 初始学习learnRate
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Verbose', false, ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info, 0.001));
    
    % 加载数据并训练模型
%     load simple_lstm_data.mat;
%     net = trainNetwork(p_train, t_train, layers, options);
    net = trainNetwork(p_train, t_train, lgraph, options);
   
    
    %%  仿真预测
    t_sim2 = predict(net, p_test );

    %%  数据反归一化
    T_sim2 = mapminmax('reverse', t_sim2, ps_output);

    %%  均方根误差
    fitness = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
end

% 停止条件
function stop = stopIfAccuracyNotImproving(info, expectLoss)
% 如果连续 patience 次验证集准确率未提高，则停止训练
persistent minimum;
persistent nochange;

if info.State == "start"
    minimum = 1;
    nochange = 0;
end
if minimum > info.TrainingLoss
    minimum = info.TrainingLoss;
    nochange = 0;
else
   nochange = nochange + 1;   
end
    if info.TrainingLoss <= expectLoss
        stop = true;
    elseif nochange >= 20
        stop = true;
    else
        stop = false;
    end
end
