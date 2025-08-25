% Main Script: Parallel HMM Evaluation for 5335 Flights
numS = 2:5;            % Hidden states
numO = 3;              % Observation symbols
p = [0.85 0.9 0.95];   % Subsampling probabilities

load('flight_data_roc1.mat');
numFlights = 5335;

deltaAll = cell(numFlights, 1);
hstAll = cell(numFlights, 1);

% HMM Transition and Emission Matrices
A2 = [0.85, 0.15; 0.20, 0.80];
B2 = [0.20, 0.70, 0.10; 0.40, 0.10, 0.50];

A3 = [0.80, 0.15, 0.05; 0.10, 0.80, 0.10; 0.05, 0.15, 0.80];
B3 = [0.70, 0.20, 0.10; 0.20, 0.60, 0.20; 0.10, 0.20, 0.70];

A4 = [0.80, 0.10, 0.05, 0.05; 0.10, 0.75, 0.10, 0.05;
       0.05, 0.10, 0.80, 0.05; 0.05, 0.05, 0.10, 0.80];
B4 = [0.75, 0.20, 0.05; 0.30, 0.50, 0.20;
      0.10, 0.20, 0.70; 0.05, 0.20, 0.75];

A5 = [0.80, 0.10, 0.05, 0.03, 0.02; 0.10, 0.75, 0.10, 0.03, 0.02;
      0.05, 0.10, 0.75, 0.08, 0.02; 0.03, 0.03, 0.08, 0.80, 0.06;
      0.02, 0.02, 0.02, 0.06, 0.88];
B5 = [0.80, 0.15, 0.05; 0.50, 0.40, 0.10; 0.20, 0.60, 0.20;
      0.10, 0.40, 0.50; 0.05, 0.20, 0.75];

% Start parallel pool with max available (32)
if isempty(gcp('nocreate'))
    parpool('local', 32);
end

parfor f = 1:numFlights
    try
        fieldName = sprintf('flight_%04d', f);
        obs = flight_data.(fieldName).ROC + 2;

        rng('shuffle');
        [A2b, B2b] = hmmtrain(obs, A2, B2, 'MaxIterations', 1000, 'Tolerance', 1e-4);
        rng('shuffle');
        [A3b, B3b] = hmmtrain(obs, A3, B3, 'MaxIterations', 1000, 'Tolerance', 1e-4);
        rng('shuffle');
        [A4b, B4b] = hmmtrain(obs, A4, B4, 'MaxIterations', 1000, 'Tolerance', 1e-4);
        rng('shuffle');
        [A5b, B5b] = hmmtrain(obs, A5, B5, 'MaxIterations', 1000, 'Tolerance', 1e-4);

        delta = zeros(length(p), length(numS));

        for i = 1:length(p)
            mask = rand(size(obs)) < p(i);
            obs_p = obs(mask);

            rng('shuffle');
            [A2p, ~] = hmmtrain(obs_p, A2, B2, 'MaxIterations', 1000, 'Tolerance', 1e-4);
            rng('shuffle');
            [A3p, ~] = hmmtrain(obs_p, A3, B3, 'MaxIterations', 1000, 'Tolerance', 1e-4);
            rng('shuffle');
            [A4p, ~] = hmmtrain(obs_p, A4, B4, 'MaxIterations', 1000, 'Tolerance', 1e-4);
            rng('shuffle');
            [A5p, ~] = hmmtrain(obs_p, A5, B5, 'MaxIterations', 1000, 'Tolerance', 1e-4);

            A2p = normalize_matrix(A2p); A3p = normalize_matrix(A3p);
            A4p = normalize_matrix(A4p); A5p = normalize_matrix(A5p);
            A2b = normalize_matrix(A2b); A3b = normalize_matrix(A3b);
            A4b = normalize_matrix(A4b); A5b = normalize_matrix(A5b);

            A2c = A2b * p(i) / (1 - (1 - p(i)) * sum(diag(A2b)));
            A3c = A3b * p(i) / (1 - (1 - p(i)) * sum(diag(A3b)));
            A4c = A4b * p(i) / (1 - (1 - p(i)) * sum(diag(A4b)));
            A5c = A5b * p(i) / (1 - (1 - p(i)) * sum(diag(A5b)));

            A2c = normalize_matrix(A2c); A3c = normalize_matrix(A3c);
            A4c = normalize_matrix(A4c); A5c = normalize_matrix(A5c);

            delta(i,1) = compare_matrices(A2p, A2c);
            delta(i,2) = compare_matrices(A3p, A3c);
            delta(i,3) = compare_matrices(A4p, A4c);
            delta(i,4) = compare_matrices(A5p, A5c);
        end

        % Final best model index based on KL
        [~, hSt] = min(sum(delta,1));
        deltaAll{f} = delta;
        hstAll{f} = hSt + 1;

        fprintf("Flight %d/%d complete\n", f, numFlights);

    catch ME
        fprintf("Error in flight %d: %s\n", f, ME.message);
        deltaAll{f} = NaN;
        hstAll{f} = NaN;
    end
end

save('parallel_test_for_5335.mat', 'deltaAll', 'hstAll');

% ======================
% Helper Functions
% ======================
function M = normalize_matrix(M)
    M = M ./ sum(M, 2);
end

function kldiv = compare_matrices(P, Q)
    M = numel(P);
    P = reshape(P, [M,1]);
    Q = reshape(Q, [M,1]);
    tf = P~=0 & Q~=0;
    kld = P(tf) .* log2(P(tf)./Q(tf));
    kldiv = sum(kld(~isnan(kld)));
end
