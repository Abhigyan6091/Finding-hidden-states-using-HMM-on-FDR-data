function delta = compare_matrices(P, Q)
    P = P ./ sum(P, 2);
    Q = Q ./ sum(Q, 2);

    M = numel(P);
    P = reshape(P, [M, 1]);
    Q = reshape(Q, [M, 1]);

    tf = P ~= 0 & Q ~= 0;
    kld = P(tf) .* log2(P(tf) ./ Q(tf));
    kl_distance = sum(kld(~isnan(kld)));

    delta = kl_distance;
end




% Load data
load('flight_data_roc1.mat');  % Assumes ROC is a struct with 5335 entries

numFlights = length(flight_data);

numS2 = 2;
numS3 = 3;
numS4 = 4;
numS5 = 5;

numO = 3;



prob_seq = [0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95];

% Preallocate output arrays
deltaAll_flights = zeros(numFlights, length(prob_seq), 4);  % [flight x prob x stateModel]
optimalStates = zeros(numFlights, length(prob_seq));        % [flight x prob] - stores 2 to 5

% Define initial HMM parameters for 2–5 state models
pi_02 = [1, 0];
A_02 = [0.7, 0.3; 0.3, 0.7];
B_02 = [0.1, 0.8, 0.1; 0.8, 0.1, 0.1];

pi_03 = [1, 0, 0];
A_03 = [0.7, 0.3, 0.0; 0.2, 0.6, 0.2; 0.0, 0.3, 0.7];
B_03 = [0.1, 0.8, 0.1; 0.8, 0.1, 0.1; 0.1, 0.1, 0.8];

pi_04 = [1, 0, 0, 0];
A_04 = [0.7, 0.2, 0.1, 0.0; 0.2, 0.5, 0.2, 0.1; 0.1, 0.2, 0.5, 0.2; 0.0, 0.1, 0.2, 0.7];
B_04 = [0.1, 0.8, 0.1; 0.8, 0.1, 0.1; 0.8, 0.1, 0.1; 0.1, 0.1, 0.8];

pi_05 = [1, 0, 0, 0, 0];
A_05 = [0.6, 0.2, 0.1, 0.0, 0.0; 0.2, 0.5, 0.2, 0.1, 0.0; 0.1, 0.2, 0.4, 0.2, 0.1;
        0.0, 0.1, 0.2, 0.5, 0.2; 0.0, 0.0, 0.1, 0.2, 0.6];
B_05 = [0.8, 0.1, 0.1; 0.8, 0.1, 0.1; 0.1, 0.8, 0.1; 0.1, 0.1, 0.8; 0.1, 0.1, 0.8];








% Loop through all flights
for f = 1:numFlights
    fieldName = sprintf('flight_%04d', f);
    observations = flight_data.(fieldName).ROC;   % Adjust field name if needed


    % Train baseline HMMs using entire observation sequence
    [A_2bw, ~] = hmmtrain(observations, A_02, B_02);
    [A_3bw, ~] = hmmtrain(observations, A_03, B_03);
    [A_4bw, ~] = hmmtrain(observations, A_04, B_04);
    [A_5bw, ~] = hmmtrain(observations, A_05, B_05);

    I2 = eye(2); I3 = eye(3); I4 = eye(4); I5 = eye(5);

    for i = 1:length(prob_seq)
        p = prob_seq(i);
        mask = rand(size(observations)) < p;
        observations_p = observations(mask);

        % Train HMMs on masked observation sequences
        [A_2p, ~] = hmmtrain(observations_p, A_02, B_02);
        [A_3p, ~] = hmmtrain(observations_p, A_03, B_03);
        [A_4p, ~] = hmmtrain(observations_p, A_04, B_04);
        [A_5p, ~] = hmmtrain(observations_p, A_05, B_05);

        % Correct the transition matrices
        A_2c = A_2bw * p / (I2 - (1 - p) * A_2bw);
        A_3c = A_3bw * p / (I3 - (1 - p) * A_3bw);
        A_4c = A_4bw * p / (I4 - (1 - p) * A_4bw);
        A_5c = A_5bw * p / (I5 - (1 - p) * A_5bw);

        % Compute KL divergence for each model
        deltaAll_flights(f, i, 1) = compare_matrices(A_2p, A_2c);
        deltaAll_flights(f, i, 2) = compare_matrices(A_3p, A_3c);
        deltaAll_flights(f, i, 3) = compare_matrices(A_4p, A_4c);
        deltaAll_flights(f, i, 4) = compare_matrices(A_5p, A_5c);
    end

    % Track progress
    fprintf("Flight %d/%d processed\n", f, numFlights);
end

% Identify best model (least KL divergence) for each [flight, prob_seq]
for f = 1:numFlights
    for i = 1:length(prob_seq)
        kl_values = squeeze(deltaAll_flights(f, i, :));  % 4x1 vector
        [~, min_idx] = min(kl_values);
        optimalStates(f, i) = min_idx + 1;  % index 1→state 2, etc.
    end
end

% Save all outputs
save('deltaAll_results.mat', 'deltaAll_flights', 'optimalStates', 'prob_seq');
