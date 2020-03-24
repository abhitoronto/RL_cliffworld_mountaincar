% main_p2_mc_mpc: Main script for Problem 2.2 mountain car (MPC approach)
%
% --
% Control for Robotics
% AER1517 Spring 2020
% Programming Exercise 2
%
% --
% University of Toronto Institute for Aerospace Studies
% Dynamic Systems Lab
%
% Course Instructor:
% Angela Schoellig
% schoellig@utias.utoronto.ca
%
% Teaching Assistant:
% SiQi Zhou
% siqi.zhou@robotics.utias.utoronto.ca
%
% --
% Revision history
% [20.03.07, SZ]    first version

clear all;
close all;
clc;

%% General
% Add path
addpath(genpath(pwd));

% MPC parameters
n_lookahead = 100; % MPC prediction horizon
n_mpc_update = 1; % MPC update frequency
use_model = true;

% Cost function parameters
Q = diag([100, 0]); % not penalizing velocity
r = 0;

% Initial state
cur_state = [-pi/6; 0]; % [-pi/6; 0];
goal_state = [0.5; 0.05];
state_stack = cur_state;
input_stack = [];

% State and action bounds
pos_bounds = [-1.2, 0.5]; % state 1: position
vel_bounds = [-0.07, 0.07]; % state 2: velocity
acc_bounds = [-1, 1]; % action: acceleration

% Plotting parameters
linecolor = [1, 1, 1].*0.5;
fontcolor = [1, 1, 1].*0.5;
fontsize = 12;

% Max number of time steps to simulate
max_steps = 500;

% Standard deviation of simulated Gaussian measurement noise
noise = [1e-3; 1e-5];

% Result and plot directory
save_dir = './results/';
mkdir(save_dir);

%% Problem 2.2: (d) Solving mountain car problem with MPC
% State and action bounds
state_bound = [pos_bounds; vel_bounds];
action_bound = [acc_bounds];

% Struct used in simulation and visualization scripts
world.param.pos_bounds = pos_bounds;
world.param.vel_bounds = vel_bounds;
world.param.acc_bounds = acc_bounds;

% Action and state dimensions
dim_state = size(state_bound, 1);
dim_action = size(action_bound, 1);

% MPC implmentation
tic;
for k = 1:1:max_steps
    
    if mod(k, n_mpc_update) == 1 || n_mpc_update == 1
        fprintf('updating inputs...\n');
        % Get cost Hessian matrix
        S = get_cost(r, Q, n_lookahead);

        % Lower and upper bounds
        lb = [repmat(action_bound(1),n_lookahead,1); ...
            repmat(state_bound(:,1),n_lookahead,1)];
        ub = [repmat(action_bound(2),n_lookahead,1); ...
            repmat(state_bound(:,2)+[0;0],n_lookahead,1)];
        
        % Optimize state and action over prediction horizon
        if k <= 1
            % Solve nonlinear MPC at the first step
            if k == 1
                initial_guess = randn(n_lookahead*(dim_state+dim_action), 1);
            else
                initial_guess = x;
            end
            
            % Cost function
            sub_states = [repmat(0,n_lookahead,1); ...
                repmat(goal_state, n_lookahead,1)];
            fun = @(x) (x - sub_states)'*S*(x - sub_states);
    
            % Temporary variables used in 'dyncons'
            save('params', 'n_lookahead', 'dim_state', 'dim_action');
            save('cur_state', 'cur_state');
            
            % Solve nonlinear MPC
            % x is a vector containing the inputs and states over the
            % horizon [input,..., input, state', ..., state']^T
            options = optimoptions(@fmincon, 'MaxFunctionEvaluations', ...
                1e5, 'MaxIterations', 1e5, 'Display', 'iter');
            [x,fval] = fmincon(fun, initial_guess, [], [], [], [], ...
                lb, ub, @dyncons, options);
        else
            % ================== [TODO] QP Implementation =================
            % Problem 2.2: (d) Quadratic Program optimizing state and
            % action over prediction horizon
            
            % Feedback state used in MPC updates
            % 'cur_state' or 'cur_state_noisy'
            cur_state_mpc_update = cur_state;
            
            % Solve QP (e.g., using Matlab's quadprog function) 
            % Note 1: x is a vector containing the inputs and states over 
            %         the horizon [input,..., input, state', ..., state']^T
            % Note 2: The function 'get_lin_matrices' computes the
            %         Jacobians (A, B) evaluated at an operation point
            x_prev = cur_state_mpc_update;
            a_curr = [];
            states_curr = [];
            Aeq = zeros(n_lookahead*(dim_state), n_lookahead*(dim_state+dim_action));
            Aeq(:, n_lookahead*(dim_action)+1:end) = ...
                                   - eye(n_lookahead*(dim_state));
            % Populate Aeq
            for s = 1:n_lookahead
                % Get linearization points
                actions = n_lookahead*(dim_action);
                if s < n_lookahead
                    a_bar = cur_mpc_inputs(:, s);
                    if ~use_model
                        x_bar = cur_mpc_states(:, s);
                    end
                else
                    a_bar = 0;
                    if ~use_model
                        [x_bar, ~, ~] = ...
                        one_step_mc_model(world, x_prev, a_bar);
                    end
                end 
                if use_model
                    [x_bar, ~, ~] = ...
                        one_step_mc_model(world, x_prev, a_bar);
                end
                a_curr = [a_curr; a_bar];
                states_curr = [states_curr; x_bar];
                x_prev = x_bar;
                
                % Get linearized model for current state and action
                [A_, b_] = get_lin_matrices(x_bar, a_bar);
                
                % Populate Aeq
                if s > 1
                    mat = A_ * Aeq((s-2)*dim_state+1:(s-1)*dim_state, 1:(s-1)*dim_action );
                    Aeq((s-1)*dim_state+1:(s)*dim_state, 1:(s-1)*dim_action) = mat;
                end
                Aeq((s-1)*dim_state+1:(s)*dim_state, (s)*dim_action) = b_;
            end
            
            x = [a_curr; states_curr];
            
            beq = Aeq * x;
            f = -(S + S')*sub_states;
            H = S;
            
            x = quadprog(H,f,[],[],Aeq, beq, lb, ub);
            % =============================================================
        end

        % Separate inputs and states from the optimization variable x
        inputs = x(1:n_lookahead*dim_action);
        states_crossterms = x(n_lookahead*dim_action+1:end);
        position_indeces = 1:2:2*n_lookahead;
        velocity_indeces = position_indeces + 1;
        positions = states_crossterms(position_indeces);
        velocities = states_crossterms(velocity_indeces);
        
        % Variables if not running optimization at each time step
        cur_mpc_inputs = inputs';
        cur_mpc_states = [positions'; velocities'];
    end
    
    % Propagate
    action = cur_mpc_inputs(1);
    [cur_state, cur_state_noisy, ~, is_goal_state] = ...
        one_step_mc_model_noisy(world, cur_state, action, noise);
    
    % Remove first input
    cur_mpc_inputs(1) = [];
    cur_mpc_states(:,1) = [];
    
    % Save state and input
    state_stack = [state_stack, cur_state];
    input_stack = [input_stack, action];
    
    % Plot
    grey = [0.5, 0.5, 0.5];
    hdl = figure(1);
    hdl.Position(3) = 1155;
    clf;
    subplot(3,2,1);
	plot(state_stack(1,:), 'linewidth', 3); hold on;
    plot(k+1:k+length(cur_mpc_states(1,:)), cur_mpc_states(1,:), 'color', grey);
    ylabel('Car Position');
    subplot(3,2,3);
    plot(state_stack(2,:), 'linewidth', 3); hold on;
    plot(k+1:k+length(cur_mpc_states(2,:)), cur_mpc_states(2,:), 'color', grey);
    ylabel('Car Velocity');
    subplot(3,2,5);
    plot(input_stack(1,:), 'linewidth', 3); hold on;
    plot(k:k+length(cur_mpc_inputs)-1, cur_mpc_inputs, 'color', grey);
    xlabel('Discrete Time Index');
    ylabel('Acceleration Cmd');
    subplot(3,2,[2,4,6]);
	xvals = linspace(world.param.pos_bounds(1), world.param.pos_bounds(2));
	yvals = get_car_height(xvals);
	plot(xvals, yvals, 'color', linecolor, 'linewidth', 1.5); hold on;
    plot(cur_state(1), get_car_height(cur_state(1)), 'ro', 'linewidth', 2);
    xlabel('x Position');
    ylabel('y Position');
    pause(0.1);
    
    % Break if goal reached
    if is_goal_state
        fprintf('goal reached\n');
        break
    end
end
compute_time = toc;

% Visualization
plot_visualize = true;
plot_title = 'Model Predictive Control';
hdl = visualize_mc_solution_mpc(world, state_stack, input_stack, ...
    plot_visualize, plot_title, save_dir);

% Save results
save(strcat(save_dir, 'mpc_results.mat'), 'state_stack', 'input_stack');
