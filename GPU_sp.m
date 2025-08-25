% Start parallel pool
if isempty(gcp('nocreate'))
    parpool('local');
end

% Parameters
numS = 2:5;
numO = 3;
p = [0.85 0.9 0.95];

% Load data
load('flight_data_roc1.mat');
numFlights = 5335;
disp(numFlights);

% Preallocate results
deltaAll = cell(numFlights, 1);
hstAll = cell(numFlights, 1);

% Base HMM parameters
A2 = gpuArray([0.85, 0.15; 0.20, 0.80]);
B2 = gpuArray([0.20, 0.70, 0.10; 0.40, 0.10, 0.50]);

A3 = gpuArray([0.80, 0.15, 0.05; 0.10, 0.80, 0.10; 0.05, 0.15, 0.80]);
B3 = gpuArray([0.70, 0.20, 0.10; 0.20, 0.60, 0.20; 0.10, 0.20, 0.70]);

A4 = gpuArray([0.80, 0.10, 0.05, 0.05;
               0.10, 0.75, 0.10, 0.05;
               0.05, 0.10, 0.80, 0.05;
               0.05, 0.05, 0.10, 0.80]);
B4 = gpuArray([0.75, 0.20, 0.05;
               0.30, 0.50, 0.20;
               0.10, 0.20, 0.70;
               0.05, 0.20, 0.75]);

A5 = gpuArray([0.80, 0.10, 0.05, 0.03, 0.02;
               0.10, 0.75, 0.10, 0.03, 0.02;
               0.05, 0.10, 0.75, 0.08, 0.02;
               0.03, 0.03, 0.08, 0.80, 0.06;
               0.02, 0.02, 0.02, 0.06, 0.88]);
B5 = gpuArray([0.80, 0.15, 0.05;
               0.50, 0.40, 0.10;
               0.20, 0.60, 0.20;
               0.10, 0.40, 0.50;
               0.05, 0.20, 0.75]);


parfor f = 1:numFlights
    delta = zeros(length(p), length(numS));
    fieldName = sprintf('flight_%04d', f);
    obs = flight_data.(fieldName).ROC + 2;
    % Copy to CPU for hmmtrain (which does not support GPU)
    A2c = gather(A2); B2c = gather(B2);
    A3c = gather(A3); B3c = gather(B3);
    A4c = gather(A4); B4c = gather(B4);
    A5c = gather(A5); B5c = gather(B5);


    % Train models
    A2b = hmmtrain(obs, A2c, B2c, 'MaxIterations', 1000, 'Tolerance', 1e-4);
    A3b = hmmtrain(obs, A3c, B3c, 'MaxIterations', 1000, 'Tolerance', 1e-4);
    A4b = hmmtrain(obs, A4c, B4c, 'MaxIterations', 1000, 'Tolerance', 1e-4);
    A5b = hmmtrain(obs, A5c, B5c, 'MaxIterations', 1000, 'Tolerance', 1e-4);

    for i = 1:length(p)
        mask = rand(size(obs)) < p(i);
        obs_p = obs(mask);

        A2p = hmmtrain(obs_p, A2c, B2c, 'MaxIterations', 1000, 'Tolerance', 1e-4);
        A3p = hmmtrain(obs_p, A3c, B3c, 'MaxIterations', 1000, 'Tolerance', 1e-4);
        A4p = hmmtrain(obs_p, A4c, B4c, 'MaxIterations', 1000, 'Tolerance', 1e-4);
        A5p = hmmtrain(obs_p, A5c, B5c, 'MaxIterations', 1000, 'Tolerance', 1e-4);

        % Normalize transition matrices
        A2b = normalize_rows(A2b); A3b = normalize_rows(A3b);
        A4b = normalize_rows(A4b); A5b = normalize_rows(A5b);
        A2p = normalize_rows(A2p); A3p = normalize_rows(A3p);
        A4p = normalize_rows(A4p); A5p = normalize_rows(A5p);

        % Move to GPU for matrix ops
        A2b_gpu = gpuArray(A2b); A3b_gpu = gpuArray(A3b);
        A4b_gpu = gpuArray(A4b); A5b_gpu = gpuArray(A5b);

        A2c_gpu = A2b_gpu * p(i) / (eye(2, 'gpuArray') - (1 - p(i)) * A2b_gpu);
        A3c_gpu = A3b_gpu * p(i) / (eye(3, 'gpuArray') - (1 - p(i)) * A3b_gpu);
        A4c_gpu = A4b_gpu * p(i) / (eye(4, 'gpuArray') - (1 - p(i)) * A4b_gpu);
        A5c_gpu = A5b_gpu * p(i) / (eye(5, 'gpuArray') - (1 - p(i)) * A5b_gpu);

        A2c = gather(normalize_rows(A2c_gpu));
        A3c = gather(normalize_rows(A3c_gpu));
        A4c = gather(normalize_rows(A4c_gpu));
        A5c = gather(normalize_rows(A5c_gpu));

        delta(i, 1) = compare_matrices(A2p, A2c);
        delta(i, 2) = compare_matrices(A3p, A3c);
        delta(i, 3) = compare_matrices(A4p, A4c);
        delta(i, 4) = compare_matrices(A5p, A5c);
    end

    hSt = hidden_state(delta) + 1;
    deltaAll{f} = delta;
    hstAll{f} = hSt;
    fprintf("Flight %d/%d done\n", f, numFlights);
end

save('results_gpu_parallel.mat', 'deltaAll', 'hstAll');

%% Helper Functions

function A = normalize_rows(A)
    A = A ./ sum(A, 2);
end

function kldiv = compare_matrices(P, Q)
    M = numel(P);
    P = reshape(P, [M,1]);
    Q = reshape(Q, [M,1]);
    tf = P~=0 & Q~=0;
    kld = P(tf) .* log2(P(tf)./Q(tf));
    kldiv = sum(kld(~isnan(kld)));
end

function hState = hidden_state(M)
    A = M;
    precision = 4;
    tolerance = 10^(-precision);
    col_mins = min(A, [], 1);
    result_cols = [];

    for i = 1:length(col_mins)
        for j = i+1:length(col_mins)
            if abs(col_mins(i) - col_mins(j)) < tolerance
                result_cols = unique([result_cols, i, j]);
            end
        end
    end
    hState = result_cols;
end
