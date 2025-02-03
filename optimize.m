function [p] = optimize(model, output)
% Find the excitation signal that generates a mean muscle activation of 0.5,
% with minimal effort (defined as mean squared excitation).
% Two methods are used:
% (1) direct collocation
% (2) direct shooting, with square wave input parameters p = [umin,umax,duty_cycle]

    global problem
    rng(0)

    % default inputs
    if (nargin < 1)
        model = 'McLean2003';
    end
    if (nargin < 2)
        output = true;
    end


    % set some model parameters
    problem.model = model;
    problem.Tact   = 0.015;
    problem.Tdeact = 0.060;

    % options for the trajectory optimization
    optimizer = 'ipopt';  % choose 'ipopt', 'fmincon', or 'none' to skip optimizing
    
    % if ipopt was selected, but not installed, use fmincon instead
    if strcmp(optimizer,'ipopt') && isempty(which('ipopt'))
        disp('ipopt is not installed, using fmincon')
        optimizer = 'fmincon';
    end

    N = 200;     % number of collocation points
    T = 1.0;     % duration
    times = linspace(0,T,N);
    problem.N = N;
    problem.h = mean(diff(times));  % time step
    problem.method = 'trapezoidal';  % Choose 'euler' or 'trapezoidal'

    % we solve for a vector X which contains N samples of the
    % output x(t), followed by N samples of the control u(t)
    nX = 2*N;
    problem.ix = (1:N)';      % indices of x(t)
    problem.iu = N + (1:N)';  % indices of u(t)

    % initial guess and bounds
    X = rand(nX,1);
    lb = zeros(nX,1);
    ub = ones(nX,1);

    % determine sparsity structure of the constraints Jacobian
    problem.Jpattern = spalloc(N+2, nX, 10);
    for i = 1:10
        X = rand(nX,1);
        J = conjac(X);
        % add any nonzeros
        problem.Jpattern = double(problem.Jpattern | J~=0);
        % the following print can confirm that we had exactly the same NNZ number each time
        if (output)
            fprintf('nnz(Jpattern): %d\n', nnz(problem.Jpattern));
        end
    end

    % check the derivatives
    if (output)
        f = objfun(X);
        g = objgrad(X);
        c = confun(X);
        J = conjac(X);
        gnum = zeros(size(g));
        Jnum = zeros(size(J));
        hh = 1e-7;
        for i = 1:nX
            tmp = X(i);
            X(i) = X(i) + hh;
            fhh = objfun(X);
            gnum(i) = (fhh-f)/hh;
            chh = confun(X);
            Jnum(:,i) = (chh-c)/hh;
            X(i) = tmp;
        end
        [maxerr,row] = max(abs(g-gnum));
        fprintf("largest error in gradient is %f et %d (%f coded vs %f numerical)\n", ...
            maxerr,row,g(row),gnum(row));
        [~,col] = max(max(abs(J-Jnum)));
        [maxerr,row] = max(max(abs(J'-Jnum')));
        fprintf("largest error in Jacobian is %f at %d,%d (%f coded vs %f numerical)\n", ...
            full(maxerr),row,col,full(J(row,col)),full(Jnum(row,col)));
        disp('Hit ENTER to continue');
        pause
    end

    % solve with ipopt, if installed, otherwise use fmincon
    fprintf('Solving %s with collocation...\n', model)
    if strcmp(optimizer,'ipopt')
        % solve with IPOPT
        funcs.objective   = @objfun;
        funcs.gradient    = @objgrad;
        funcs.constraints = @confun;
        funcs.jacobian    = @conjac;
        funcs.jacobianstructure = @conjacStructure;
        options.cl = zeros(N+2,1);
        options.cu = zeros(N+2,1);
        options.lb = lb;
        options.ub = ub;
	    options.ipopt.hessian_approximation = 'limited-memory'; 
        options.ipopt.max_iter = 100000;
        if ~output
            options.ipopt.print_level = 0;
        end
        [X, info] = ipopt(X,funcs,options);
    elseif strcmp(optimizer,'fmincon')
        % solve with fmincon
        if output
            print_level = 'iter';
        else
            print_level = 'none';
        end
        options = optimoptions('fmincon','ConstraintTolerance',1e-10, ...
                                         'OptimalityTolerance',1e-10, ...
                                         'MaxFunctionEvaluations',100000, ...
                                         'MaxIterations',10000, ...
                                         'SpecifyObjectiveGradient',true, ...
                                         'SpecifyConstraintGradient',true, ...
                                         'Display',print_level);
        X = fmincon(@obj_fmincon,X,[],[],[],[],lb,ub,@con_fmincon,options);
    end

    % extract u(t) and x(t)
    u = X(problem.iu);
    x = X(problem.ix);

    % report the means, exclude the last point which is the start of the next cycle
    fprintf('    mean control-squared: %10.7f\n', mean(u(1:end-1).^2));
    fprintf('    mean activation:      %10.7f\n', mean(x(1:end-1)));
   
    % plot the optimal solution
    figure()
    set(gcf, 'Position', [387 573 853 305]);
    subplot(1,2,1)
    plot(times,[u x]);
    ylim([-0.2 1.2])
    xlabel('time (s)')
    legend('excitation','activation')
    title([problem.model ' collocation'])
    drawnow;

    % now solve the problem with a shooting method, and a parameterized
    % control input u(t): p = [umin,umax,dutycycle]
    fprintf('Solving %s with shooting...\n', model)

    % initial guess, and bounds, for the 3 parameters
    p = [0.1 ; 0.9; 0.5];  % umin, umax, dutycycle
    lb = zeros(3,1);
    ub = ones(3,1);

    % solve with fmincon
    if output
        print_level = 'iter';
    else
        print_level = 'none';
    end
    options = optimoptions('fmincon','ConstraintTolerance',1e-10, ...
                                     'OptimalityTolerance',1e-10, ...
                                     'MaxFunctionEvaluations',100000, ...
                                     'MaxIterations',10000, ...
                                     'Display',print_level);
    p = fmincon(@obj_shooting,p,[],[],[],[],lb,ub,@con_shooting,options);
    report = true;
    subplot(1,2,2);
    con_shooting(p,report);
end
%==================================================================================
function f = obj_shooting(p)
% evaluate the cost of the control u(t) with parameters p

    % extract parameters
    umin = p(1);
    umax = p(2);
    duty_cycle = p(3);

    % cost is the integral of u(t)^2
    f = duty_cycle*umax^2 + (1-duty_cycle)*umin^2;
end
%==================================================================================
function [c,ceq] = con_shooting(p, report)
% evaluate the task performance of control u(t) with parameters p

    global problem

    % by default, don't report any results
    if nargin < 2
        report = false;
    end

    % extract parameters
    umin = p(1);
    umax = p(2);
    duty_cycle = p(3);

    % inequality constraint: umox > umin
    c = umin - umax;  

    % then we need to satisfy an equality constraint: mean activation is 0.5
    % this is done by simulating the response

    % choose a very high square wave frequency (1000 Hz is good)
    f = 1000;  
    period = 1/f;

    % simulate for about 1 second, long enough that there is no effect of initial conditions
    duration = 1.0;
    ncycles = round(duration * f);
    xx = 0;  % initial activation state
    tt = 0;  % initial time value
    uu = 0;  % initial excitation input
    for i = 1:ncycles
        % simulate the activation phase of the cycle, with control input umax
        dur = duty_cycle * period; % duration of this phase
        [t,x] = ode45(@(t,x) odefun(x,umax), [0 dur], xx(end));
        % append results to the tt,xx,uu vectors
        % add a tiny amount to the first time value, otherwise interp1 (later) will complain about duplicate time points
        t(1) = t(1) + 1e-10;
        tt = [tt ; tt(end) + t];
        xx = [xx ; x]; 
        uu = [uu ; umax*ones(size(t))];

        % simulate the deactivation phase of the cycle, with control input umin
        dur = (1 - duty_cycle) * period; % duration of this phase
        [t,x] = ode45(@(t,x) odefun(x,umin), [0 dur], x(end));
        % append results to the tt,xx,uu vectors
        % add a tiny amount to the first time value, otherwise interp1 (later) will complain about duplicate time points
        t(1) = t(1) + 1e-10;
        tt = [tt ; tt(end) + t];
        xx = [xx ; x]; 
        uu = [uu ; umin*ones(size(t))];
    end
    duration = tt(end);  % this is the actual duration of the simulation

    % resample and calculate the means over the final cycle
    npoints = 1001;
    tnew = linspace(duration-period, duration, npoints);
    u = interp1(tt,uu,tnew);
    x = interp1(tt,xx,tnew);
    % exclude the last resampled point from the averages, since it's the start of the next cycle
    mean_u = mean(u(1:end-1));
    mean_usquared = mean(u(1:end-1).^2);
    mean_x = mean(x(1:end-1));
    % compare to the task requirement: mean activation is 0.5
    ceq = mean_x - 0.5;

    if (report)
        fprintf('    mean control-squared: %10.7f\n', mean_usquared)
        fprintf('    mean activation:      %10.7f\n', mean_x)
    
        % plot u(t) and a(t) for the final 10 simulated periods
        plot(tt,[uu xx]);
        xlabel('time (s)')
        tstart = max(0,duration-10*period);
        xlim([tstart duration])
        ylim([-0.2 1.2])
        title([problem.model ' shooting']);
        legend('excitation','activation')
        drawnow;
    end

end
%=================================================================
function [f,g] = obj_fmincon(X)
% objective function for fmincon optimizer
    f = objfun(X);
    if (nargout > 1)
        g = objgrad(X);
    end
end
%=================================================================
function f = objfun(X)
% optimization objective: minimize the mean squared control
    global problem

    f = mean(X(problem.iu(1:end-1)).^2);  % use the first N-1 points because N is the beginning of the next period

end
%==================================================================
function g = objgrad(X)
% gradient df/dX of objfun
    global problem

    % and the gradient:
    g = zeros(size(X));
    iu = problem.iu(1:end-1);  % indices to the controls in nodes 1 to N-1
    g(iu) = 2*X(iu)/numel(iu);

end
%=================================================================
function [c,ceq,Gc,Gceq] = con_fmincon(X)
% constraints for the fmincon optimizer
    % no inequality constraints
    c = [];
    Gc = [];
    % use the equality constraints coded in confun() and conjac()
    ceq = confun(X);
    if (nargout > 1)
        Gceq = conjac(X)';  % transpose because that is how fmincon wants it
    end
end
%=================================================================
function c = confun(X)
% equality constraints on the trajectory
    global problem

    N = problem.N;
    h = problem.h;

    % initialize the constraint vector and Jacobian matrix for the equality constraints
    Nc = (N-1) + 2 + 1; % N-1 for dynamics, 2 for periodicity, 1 for mean activation required
    c = zeros(Nc,1);

    % states and controls in nodes 1..N-1 and 2..N
    x1 = X(problem.ix(1:(N-1)));
    x2 = X(problem.ix(2:N));
    u1 = X(problem.iu(1:(N-1)));
    u2 = X(problem.iu(2:N));
    ic = 1:(N-1);

    % equality constraints to approximate the ODE dx/dt = f(x,u) at all time steps
    if strcmp(problem.method,'trapezoidal')
        % trapezoidal formula: (x2-x1)/h = 0.5*( f(x1,u1) + f(x2,u2) )
        f1 = odefun(x1,u1);
        f2 = odefun(x2,u2);
        c(ic) = (x2-x1)/h - 0.5*(f1+f2);
    else
        % backward Euler formula: (x2-x1)/h = f(x2,u2)
        f2 = odefun(x2,u2);
        c(ic) = (x2-x1)/h - f2;
    end

    % two constraints for periodicity: x(T)-x(0)=0 and u(T)-u(0)=0
    x0 = X(problem.ix(1));
    xT = X(problem.ix(end));
    u0 = X(problem.iu(1));
    uT = X(problem.iu(end));
    c(N)   = xT-x0;
    c(N+1) = uT-u0;

    % one constraint for the task requirement: produce a mean activation of 0.5
    c(N+2) = mean(X(1:(N-1))) - 0.5;  % only use the first N-1 nodes, because N is the start of the next period

end
%=================================================================
function J = conjac(X)
% Jacobian matrix for the constraints coded in confun(X)
    global problem

    N = problem.N;
    h = problem.h;
    J = problem.Jpattern;

    % indices for states and controls in nodes 1 and 2
    ix1 = 1;
    ix2 = 2;     
    iu1 = N+1;   
    iu2 = N+2;
    ic = 1;

    % equality constraints to approximate the ODE dx/dt = f(x,u) at all time steps
    for i = 1:N-1
        % extract states and controls for this pair of nodes (i and i+1)
        x1 = X(ix1);
        x2 = X(ix2);
        u1 = X(iu1);
        u2 = X(iu2);
        if strcmp(problem.method,'trapezoidal')
            % trapezoidal formula: (x2-x1)/h = 0.5*( f(x1,u1) + f(x2,u2) )
            [~,df1dx1,df1du1] = odefun(x1,u1);
            [~,df2dx2,df2du2] = odefun(x2,u2);
            % c(ic) = (x2-x1)/h - 0.5*(f1+f2);
            J(ic,ix1) = -1.0/h - 0.5*df1dx1;
            J(ic,ix2) =  1.0/h - 0.5*df2dx2;
            J(ic,iu1) = -0.5*df1du1;
            J(ic,iu2) = -0.5*df2du2;
        else
            % backward Euler formula: (x2-x1)/h = f(x2,u2)
            [~,df2dx2,df2du2] = odefun(x2,u2);
            % c(ic) = (x2-x1)/h - f2;
            J(ic,ix1) = -1.0/h;
            J(ic,ix2) = 1.0/h - df2dx2;
            J(ic,iu2) = -df2du2;
        end
        % advance the indices to the next node
        ix1 = ix1 + 1;
        ix2 = ix2 + 1;
        iu1 = iu1 + 1;
        iu2 = iu2 + 1;
        ic = ic + 1;
    end

    % two constraints for periodicity: x(T)-x(0)=0 and u(T)-u(0)=0
    % x0 = X(problem.ix(1));
    % xT = X(problem.ix(end));
    % u0 = X(problem.iu(1));
    % uT = X(problem.iu(end));
    % c(ic)   = xT-x0;
    % c(ic+1) = uT-u0;
    J(ic,   problem.ix(1))   = -1.0;
    J(ic,   problem.ix(end)) =  1.0;
    J(ic+1, problem.iu(1))   = -1.0;
    J(ic+1, problem.iu(end)) =  1.0;
    ic = ic + 2;

    % one constraint for the task requirement: produce a mean activation of 0.5
    % c(ic) = mean(X(1:(N-1))) - 0.5;  % only use the first N-1 nodes, because N is the start of the next period
    J(ic, 1:(N-1) ) = 1.0/(N-1);

end
%===================================
function J = conjacStructure()
    global problem
    J = problem.Jpattern;
end
%===================================================================
function [xdot, dxdot_dx, dxdot_du] = odefun(x,u)
% ODE function for activation state x with control input u
% also returns the derivatives, if requested
    global problem

    Tact   = problem.Tact;
    Tdeact = problem.Tdeact;

    % activation dynamics model from McLean et al., J Biomech Eng 2003
    if strcmp(problem.model, 'McLean2003')
        xdot = (u/Tact + (1-u)/Tdeact) .* (u - x);
    elseif strcmp(problem.model, 'McLean2003improved')
        % improved version (see activation-dynamics-report.pdf)
        b = 100;
        z = b*(u-x);
        f = 0.5 + 0.5*(z./sqrt(1+z.^2));  % this does not saturate as quickly as tanh and should work better for optimal control
        % rate constant is a weighted average of activation and deactivation rates
        R = f*(1/Tact) + (1-f)*(1/Tdeact);
        xdot = R .* (u - x);
    elseif strcmp(problem.model, 'DeGroote2016')
        b = 10;
        z = b*(u-x);
        f = 0.5*tanh(z);  
        % then equation (2) from De Groote et al., 2016
        xdot = ( 1/Tact./(0.5+1.5*x).*(f+0.5) + (0.5+1.5*x)./Tdeact.*(-f+0.5) ) .* (u-x);
    end

    % compute derivatives, if requested
    if (nargout > 1)
        if strcmp(problem.model, 'McLean2003')
            dxdot_dx = -(u/Tact + (1-u)/Tdeact);
            dxdot_du = (1/Tact - 1/Tdeact).*(u-x) + (u/Tact + (1-u)/Tdeact);
        elseif strcmp(problem.model, 'McLean2003improved')
            % improved version
            dz_dx = -b;
            dz_du = b;
            df_dz = 1./(2*(z.^2 + 1).^(1/2)) - z.^2./(2*(z.^2 + 1).^(3/2));
            df_dx = df_dz .* dz_dx;  % chain rule
            df_du = df_dz .* dz_du;  % chain rule
            dR_df = (1/Tact) - (1/Tdeact);
            dR_dx = dR_df .* df_dx;  % chain rule
            dR_du = dR_df .* df_du;  % chain rule
            dxdot_dx = dR_dx .* (u-x) - R;  % product rule
            dxdot_du = dR_du .* (u-x) + R;  % product rule
        elseif strcmp(problem.model, 'DeGroote2016')
            % z = b*(u-x);
            dz_dx = -b;
            dz_du = b;
            % f = 0.5*tanh(z);  
            df_dz = 1/2 - tanh(z)^2/2;
            df_dx = df_dz * dz_dx;
            df_du = df_dz * dz_du;
            y = 0.5+1.5*x;
            dy_dx = 1.5;
            R = 1/Tact/y*(f+0.5) + y/Tdeact*(-f+0.5);
            dR_dy = -1/Tact./y.^2*(f+0.5) + 1./Tdeact.*(-f+0.5);
            dR_df = 1/Tact./y - y/Tdeact;
            dR_dx = dR_dy .* dy_dx + dR_df .* df_dx;
            dR_du = dR_df .* df_du;
            % xdot = R .* (u-x);
            dxdot_dx = dR_dx .* (u-x) - R;
            dxdot_du = dR_du .* (u-x) + R;
        end
    end
end

