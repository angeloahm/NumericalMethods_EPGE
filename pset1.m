%% Problem Set I - Numerical Methods (EPGE)
% Ângelo Avelar Hermeto Mendes

tic 

% Set parameters
rho = 0.95;         % autocorrelation
sigma = 0.007;      % SD
m = 3;              % scale parameter
mu = 0;             % drift
n = 9;              % gridpoints
T = 10000;          % number of periods

%% Exercise 1: Tauchen's Method
% Using function 'tauchen.m'
% grid: grid of the random variable
% P: transition matrix
[grid_tauchen, P_tauchen] = tauchen(n, m, mu, rho, sigma);

%% Exercise 2: Rouwenhorst Method
% Using function 'rouwenhorst.m'
% grid: grid of the random variable
% P: transition matrix
[grid_rouwen, P_rouwen] = rouwenhorst(n, mu, rho, sigma);

%% Exercise 3: Fitting an AR(1) using both methods
% We start simulating a continuous process (T=1000). Then we simulate the
% same process using Tauchen's and Rouwenhorst's methods. 
simulated_continuous = ones(1,T);
simulated_tauchen = ones(1,T);
simulated_rouwen = ones(1,T);

simulated_continuous(1) = 0;                           %set initial value for the AR(1) process

i0_tauchen = find(grid_tauchen==0);                    %where grid_tauchen=0?
i0_rouwen = find(grid_rouwen==0);                      %where grid_rouwen=0?

simulated_tauchen(1) = grid_tauchen(i0_tauchen);       %set initial value for the Tauchen simulation
simulated_rouwen(1) = grid_rouwen(i0_rouwen);          %set initial value for the Rouwenhorst simulation

%Start the loop using i0
state_tauchen = i0_tauchen;
state_rouwen = i0_rouwen;

for t=1:T-1
    %Shock
    shock = randn;
    
    %Simulating continous process
    simulated_continuous(1,t+1) = (1-rho)*mu+rho*simulated_continuous(t)+sigma*shock;

    % Map the normal random draw into the transition matrix
    cum_sum = cumsum(P_tauchen(state_tauchen,:));
    state_tauchen = sum(normcdf(shock) >= cum_sum)+1; 

    cum_sum = cumsum(P_rouwen(state_rouwen,:));
    state_rouwen = sum(normcdf(shock) >= cum_sum)+1; 
    
    simulated_tauchen(t+1) = grid_tauchen(state_tauchen);
    simulated_rouwen(t+1) = grid_rouwen(state_rouwen);
    
    %If we were not interested in comparing our discrete process to a
    %specific AR(1) we can use:
    %state_tauchen = randsample(n, 1, true, P_tauchen(state_tauchen,:));
    %state_rouwen = randsample(n, 1, true, P_rouwen(state_rouwen,:));
    
end

%MSE
mse_tauchen = (simulated_continuous-simulated_tauchen).^2 ./ T;
mse_rouwnen= (simulated_continuous-simulated_rouwen).^2 ./ T;

%Plot simulations
figure
plot(linspace(1,T,T),simulated_continuous, linspace(1,T,T), simulated_tauchen)
title('Tauchen Method')
xlabel('Periods')
legend('Continuous', 'Tauchen')

figure
plot(linspace(1,T,T),simulated_continuous, linspace(1,T,T), simulated_rouwen)
title('Rouwenhorst Method')
xlabel('Periods')
legend('Continuous', 'Rouwenhorst')

figure
plot(linspace(1,T,T),simulated_tauchen, linspace(1,T,T), simulated_rouwen)
title('Tauchen vs. Rouwenhorst')
xlabel('Periods')
legend('Tauchen', 'Rouwenhorst')

figure
plot(linspace(1,T,T),mse_tauchen, linspace(1,T,T), mse_rouwnen)
title('MSE')
xlabel('Periods')
legend('Tauchen', 'Rouwenhorst')

%% Exercise 4: Estimating the AR(1) process
% We use our simulated data to estimate an AR(1) process
mdl = arima(1,0,0);
estimate_tauchen = estimate(mdl, simulated_tauchen');
estimate_rouwen = estimate(mdl, simulated_rouwen');


%% Exercise 5: Repeat with rho = 0.99

rho_new = 0.99;
[grid_tauchen, P_tauchen] = tauchen(n, m, mu, rho_new, sigma);
[grid_rouwen, P_rouwen] = rouwenhorst(n, mu, rho_new, sigma);

new_simulated_continuous = ones(1,T);
new_simulated_tauchen = ones(1,T);
new_simulated_rouwen = ones(1,T);

new_simulated_continuous(1) = 0;                           %set initial value for the AR(1) process

i0_tauchen = find(grid_tauchen==0);                    %where grid_tauchen=0?
i0_rouwen = find(grid_rouwen==0);                      %where grid_rouwen=0?

new_simulated_tauchen(1) = grid_tauchen(i0_tauchen);       %set initial value for the Tauchen simulation
new_simulated_rouwen(1) = grid_rouwen(i0_rouwen);          %set initial value for the Rouwenhorst simulation

%Start the loop using i0
state_tauchen = i0_tauchen;
state_rouwen = i0_rouwen;

for t=1:T-1
    %Shock
    shock = randn;
    
    %Simulating continous process
    new_simulated_continuous(1,t+1) = (1-rho)*mu+rho*new_simulated_continuous(t)+sigma*shock;

    % Map the normal random draw into the transition matrix
    cum_sum = cumsum(P_tauchen(state_tauchen,:));
    state_tauchen = sum(normcdf(shock) >= cum_sum)+1; 

    cum_sum = cumsum(P_rouwen(state_rouwen,:));
    state_rouwen = sum(normcdf(shock) >= cum_sum)+1; 
    
    new_simulated_tauchen(t+1) = grid_tauchen(state_tauchen);
    new_simulated_rouwen(t+1) = grid_rouwen(state_rouwen);
end

%MSE
mse_tauchen = (new_simulated_continuous-new_simulated_tauchen).^2 ./ T;
mse_rouwnen= (new_simulated_continuous-new_simulated_rouwen).^2 ./ T;

%Plot simulations
figure
plot(linspace(1,T,T),new_simulated_continuous, linspace(1,T,T), new_simulated_tauchen)
title('Tauchen Method - rho=0.99')
xlabel('Periods')
legend('Continuous', 'Tauchen')

figure
plot(linspace(1,T,T),new_simulated_continuous, linspace(1,T,T), new_simulated_rouwen)
title('Rouwenhorst Method - rho=0.99')
xlabel('Periods')
legend('Continuous', 'Rouwenhorst')

figure
plot(linspace(1,T,T),new_simulated_tauchen, linspace(1,T,T), new_simulated_rouwen)
title('Tauchen vs. Rouwenhorst - rho=0.99')
xlabel('Periods')
legend('Tauchen', 'Rouwenhorst')

figure
plot(linspace(1,T,T),mse_tauchen, linspace(1,T,T), mse_rouwnen)
title('MSE - rho=0.99')
xlabel('Periods')
legend('Tauchen', 'Rouwenhorst')


toc
