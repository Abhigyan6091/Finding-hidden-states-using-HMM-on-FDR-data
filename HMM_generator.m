rng(42);  % For reproducibility

% Parameters
numStates = 6;
numSymbols = 9;
len = 10000;

% Define initial state distribution
startProb = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  % Must sum to 1

% Define transition matrix (rows sum to 1)
transMat = [0.99, 0.01, 0.0, 0.0, 0.0, 0.0; 0.0, 0.85, 0.15, 0.0, 0.0, 0.0; 0.0, 0.0, 0.97, 0.03, 0.0, 0.0; 0.0, 0.0, 0.05, 0.90, 0.05, 0.0; 0.0, 0.0, 0.0, 0.03, 0.95, 0.02; 0.25, 0.0, 0.0, 0.0, 0.0, 0.75];

% Define emission probability matrix
emissionMat =  [0.7, 0.1, 0.1,   0.1,   0,   0,   0,   0,   0;   
                0.1, 0.7, 0.1, 0.1,   0,   0,   0,   0,   0;      
                0,   0.1, 0.1, 0.3, 0.3,   0.1,   0.1,   0.0,   0;   
                0,   0.0, 0.1, 0.1, 0.3,   0.3,   0.1,   0.1,   0;   
                0,   0,   0,   0.0, 0.1, 0.1, 0.1,   0.1,   0.6;   
                0,   0,   0,   0,   0.1, 0.7, 0.1, 0.1,   0.0];  


% Initialize arrays
states = zeros(1, len);
observations = zeros(1, len);

% Sample initial state
states(1) = find(mnrnd(1, startProb));

% Sample initial observation
observations(1) = find(mnrnd(1, emissionMat(states(1), :)));

% Generate the rest of the sequence
for t = 2:len
    states(t) = find(mnrnd(1, transMat(states(t-1), :)));
    observations(t) = find(mnrnd(1, emissionMat(states(t), :)));
end

% Output as comma-separated strings
obs_str = strjoin(string(observations), ',');
states_str = strjoin(string(states), ',');

disp('Observation Sequence:');
disp(obs_str);

disp('Hidden State Sequence:');
disp(states_str);


histogram(observations, 'BinMethod', 'integers');
title('Distribution of Observation Symbols');
xlabel('Symbol');
ylabel('Frequency');
