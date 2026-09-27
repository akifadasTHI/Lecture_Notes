    %% Rao-Blackwellized Particle Filter (RBPF) for 2D Positioning
    % Problem: Tracking a target with Nearly Constant Velocity (NCV)
    % Corrected: Proper Marginalized Kalman Filter for Linear Velocity

    clear; clc; close all;

    %% 1. Simulation Parameters
    dt = 0.1;               
    steps = 100;            
    t = (0:steps-1)*dt;

    % Process Noise Intensity
    q = 0.2; 
    % Discrete-time Process Noise Covariance Q (NCV model)
    % Q = [Q_pp, Q_pv; Q_vp, Q_vv]
    Q_1d = [dt^3/3, dt^2/2; dt^2/2, dt];
    Q_block = q * Q_1d;
    % Reorganizing into p and v blocks for RBPF
    Q_pp = diag([Q_block(1,1), Q_block(1,1)]);
    Q_pv = diag([Q_block(1,2), Q_block(1,2)]);
    Q_vp = Q_pv; 
    Q_vv = diag([Q_block(2,2), Q_block(2,2)]);

    % Measurement Noise Covariance R
    R = diag([0.5^2, 0.05^2]); % Range (m^2), Bearing (rad^2)

    %% 2. Models
    % Nonlinear Measurement Function
    h_func = @(p) [sqrt(p(1)^2 + p(2)^2); atan2(p(2), p(1))];

    %% 3. Generate Ground Truth
    x_true = zeros(4, steps);
    z_meas = zeros(2, steps);
    x_true(:,1) = [0; 2; 10; -1]; % [px; vx; py; vy]

    % Full Q for simulation
    Q_sim = [Q_block(1,1), Q_block(1,2), 0, 0;
            Q_block(2,1), Q_block(2,2), 0, 0;
            0, 0, Q_block(1,1), Q_block(1,2);
            0, 0, Q_block(2,1), Q_block(2,2)];

    for k = 2:steps
        F_sim = [1, dt, 0, 0; 0, 1, 0, 0; 0, 0, 1, dt; 0, 0, 0, 1];
        x_true(:,k) = F_sim * x_true(:,k-1) + mvnrnd(zeros(4,1), Q_sim)';
        z_meas(:,k) = h_func(x_true([1,3],k)) + mvnrnd(zeros(2,1), R)';
    end

    %% 4. RBPF Initialization
    Np = 300; % Number of particles
    particles = struct();

    for i = 1:Np
        % p = [px; py], v = [vx; vy]
        particles(i).p = [0.5; 9.5] + randn(2,1)*0.5; 
        particles(i).v = [1.5; -0.5] + randn(2,1)*0.2;
        particles(i).Pv = eye(2) * 0.1; % Covariance of velocity
        particles(i).w = 1/Np;
    end

    history = zeros(4, steps);

    %% 5. RBPF Main Loop
    for k = 1:steps
        z_k = z_meas(:,k);
        
        % --- Step 1: Prediction & Sampling ---
        for i = 1:Np
            p_prev = particles(i).p;
            v_prev = particles(i).v;
            Pv_prev = particles(i).Pv;
            
            % 1.1 Predict Linear State (Velocity) mean/cov
            % v_pred = v_prev, Pv_pred = Pv_prev + Q_vv
            v_pred = v_prev;
            Pv_pred = Pv_prev + Q_vv;
            
            % 1.2 Sample Non-linear State (Position) 
            % p_mean = p_prev + dt*v_prev, p_cov = dt^2*Pv_prev + Q_pp
            p_mean = p_prev + dt * v_prev;
            p_cov = (dt^2 * Pv_prev) + Q_pp;
            p_sampled = p_mean + chol(p_cov)' * randn(2,1);
            
            % 1.3 Update Weight based on measurement
            z_hat = h_func(p_sampled);
            innov = z_k - z_hat;
            innov(2) = atan2(sin(innov(2)), cos(innov(2))); % Wrap bearing
            w = mvnpdf(innov', [0,0], R);
            
            % 1.4 Marginalized Kalman Update for Velocity
            % The "measurement" for the velocity is the sampled position
            % y = p_sampled - p_prev = dt*v + noise
            y = p_sampled - p_prev;
            H = dt * eye(2);
            S = H * Pv_prev * H' + Q_pp;
            K = (Pv_prev * H' + Q_vp) / S; % Including cross-noise coupling
            
            particles(i).v = v_pred + K * (y - dt*v_prev);
            particles(i).Pv = Pv_pred - K * (H * Pv_prev + Q_pv');
            particles(i).p = p_sampled;
            particles(i).w = w;
        end
        
        % --- Step 2: Weight Normalization & Resampling ---
        weights = [particles.w];
        weights = weights / sum(weights);
        for i = 1:Np; particles(i).w = weights(i); end
        
        % Systematic Resampling
        if 1/sum(weights.^2) < Np/2
            edges = [0, cumsum(weights)];
            u = rand(1, Np)/Np + (0:Np-1)/Np;
            [~, idx] = histc(u, edges);
            particles = particles(idx);
            for i = 1:Np; particles(i).w = 1/Np; end
        end
        
        % --- Step 3: Estimate ---
        est_p = zeros(2,1); est_v = zeros(2,1);
        for i = 1:Np
            est_p = est_p + particles(i).p * particles(i).w;
            est_v = est_v + particles(i).v * particles(i).w;
        end
        history(:,k) = [est_p(1); est_v(1); est_p(2); est_v(2)];
    end

    %% 6. Results
    figure('Color','w','Position',[100 100 1100 450]);
    subplot(1,2,1);
    plot(x_true(1,:), x_true(3,:), 'k-', 'LineWidth', 2); hold on;
    plot(history(1,:), history(3,:), 'g--', 'LineWidth', 1.5);
    legend('Ground Truth', 'RBPF Estimate');
    title('2. Rao-Blackwellized Particle Filter Tracking');
    grid on; axis equal;

    subplot(1,2,2);
    err = sqrt((x_true(1,:)-history(1,:)).^2 + (x_true(3,:)-history(3,:)).^2);
    plot(t, err, 'r', 'LineWidth', 1.5);
    title(['Position RMSE: ', num2str(mean(err)), 'm']);
    xlabel('Time (s)'); grid on;