function optimize()
% Find the excitation signal that generates a mean muscle activation of 0.5,
% with minimal effort (defined as mean squared excitation).
% A direct collocation method is used.

    global problem
    close all
    rng(0)

    % set some model parameters
    problem.Tact   = 0.015;
    problem.Tdeact = 0.060;

    % options
    optimizer = 'ipopt';  % 'ipopt', 'fmincon', or 'none' to skip optimizing
    
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
        fprintf('nnz(Jpattern): %d\n', nnz(problem.Jpattern));
    end

    % check the derivatives
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

    % solve with ipopt, if installed, otherwise use fmincon
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
        [X, info] = ipopt(X,funcs,options);
    elseif strcmp(optimizer,'fmincon')
        % solve with fmincon
        options = optimoptions('fmincon','ConstraintTolerance',1e-10, ...
                                         'OptimalityTolerance',1e-10, ...
                                         'MaxFunctionEvaluations',100000, ...
                                         'MaxIterations',10000, ...
                                         'SpecifyObjectiveGradient',true, ...
                                         'SpecifyConstraintGradient',true, ...
                                         'Display','iter');
        X = fmincon(@obj_fmincon,X,[],[],[],[],lb,ub,@con_fmincon,options);
    end
    objOptimized = objfun(X);
    fprintf('objfun: %8.4f\n', objOptimized);

    % extract u(t) and x(t)
    u = X(problem.iu);
    x = X(problem.ix);

    % report the means, exclude the last point which is the start of the next cycle
    fprintf('Result of trajectory optimization:\n')
    fprintf('    mean control:         %7.4f\n', mean(u(1:end-1)));
    fprintf('    mean control-squared: %7.4f\n', mean(u(1:end-1).^2));
    fprintf('    mean activation:      %7.4f\n', mean(x(1:end-1)));
   
    % plot the optimal solution
    figure()
    plot(times,[u x]);
    ylim([-0.2 1.2])
    xlabel('time (s)')
    legend('excitation','activation')
    title('optimal control solution')

    % run a simulation with a hypothesized solution
    f = 1000;  % frequency (Hz) of the square wave
    period = 1/f;
    dutycycle = 0.2; % this duty cycle will produce a mean activation of exactly 0.5 with the McLean2003 model
    name = sprintf('%.0f Hz control input, %.3f duty cycle', f, dutycycle);
    fprintf('doing simulation with %s...\n', name);

    % simulate for about 5 seconds, to make sure there is no effect of startup transient
    duration = 5.0;
    ncycles = round(duration * f);
    aa = 0;  % initial activation state
    tt = 0;  % initial time value
    uu = 0;  % initial excitation input
    for i = 1:ncycles
        % simulate the activation phase of the cycle, with control input 1
        dur = dutycycle * period; % duration of this phase
        [t,a] = ode45(@(t,x) odefun(x,1), [0 dur], aa(end));
        % append results to the tt,aa,uu vectors
        % add a tiny amount to the first time value, otherwise interp1 (later) will complain about duplicate time points
        t(1) = t(1) + 1e-10;
        tt = [tt ; tt(end) + t];
        aa = [aa ; a]; 
        uu = [uu ; ones(size(a))];

        % simulate the deactivation phase of the cycle, with control input 0
        dur = (1 - dutycycle) * period; % duration of this phase
        [t,a] = ode45(@(t,x) odefun(x,0), [0 dur], aa(end));
        % append results to the tt,aa,uu vectors
        % add a tiny amount to the first time value, otherwise interp1 (later) will complain about duplicate time points
        t(1) = t(1) + 1e-10;
        tt = [tt ; tt(end) + t];
        aa = [aa ; a]; 
        uu = [uu ; zeros(size(a))];       
    end

    % plot u(t) and a(t) for the final 10 periods
    figure()
    plot(tt,[uu aa]);
    xlabel('time (s)')
    duration = tt(end);
    tstart = max(0,duration-10*period);
    xlim([tstart duration])
    ylim([-0.2 1.2])
    title(name);
    legend('excitation','activation')

    % resample and calculate the means over the final cycle
    npoints = 1001;
    tnew = linspace(duration-period, duration, npoints);
    u = interp1(tt,uu,tnew);
    a = interp1(tt,aa,tnew);
    % exclude the last resampled point from the averages, since it's the start of the next cycle
    mean_u = mean(u(1:end-1));
    mean_usquared = mean(u(1:end-1).^2);
    mean_a = mean(a(1:end-1));

    % report the means
    fprintf('Simulation result for %s\n', name)
    fprintf('    mean control:         %7.4f\n', mean_u)
    fprintf('    mean control-squared: %7.4f\n', mean_usquared)
    fprintf('    mean activation:      %7.4f\n', mean_a)

    % compare to the task requirement and the trajectory optimization solution
    if (abs(mean_a-0.5) > 1e-4)
        fprintf('This solution did NOT achieve a mean activation of 0.5\n')
    end
    if (mean_usquared < objOptimized)
        fprintf('This solution has a LOWER cost than what the trajectory optimization found!\n')
    else
        fprintf('This solution has a HIGHER cost than what the trajectory optimization found!\n')
    end

end
%=================================================================
function [f,g] = obj_fmincon(X)
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
    global problem

    % and the gradient:
    g = zeros(size(X));
    iu = problem.iu(1:end-1);  % indices to the controls in nodes 1 to N-1
    g(iu) = 2*X(iu)/numel(iu);

end
%=================================================================
function [c,ceq,Gc,Gceq] = con_fmincon(X)
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
            [f1,df1dx1,df1du1] = odefun(x1,u1);
            [f2,df2dx2,df2du2] = odefun(x2,u2);
            % c(ic) = (x2-x1)/h - 0.5*(f1+f2);
            J(ic,ix1) = -1.0/h - 0.5*df1dx1;
            J(ic,ix2) =  1.0/h - 0.5*df2dx2;
            J(ic,iu1) = -0.5*df1du1;
            J(ic,iu2) = -0.5*df2du2;
        else
            % backward Euler formula: (x2-x1)/h = f(x2,u2)
            [f2,df2dx2,df2du2] = odefun(x2,u2);
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
    x0 = X(problem.ix(1));
    xT = X(problem.ix(end));
    u0 = X(problem.iu(1));
    uT = X(problem.iu(end));
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
% ODE function with control input u
% also returns the derivatives, if requested
    global problem

    Tact   = problem.Tact;
    Tdeact = problem.Tdeact;

    % activation dynamics model from McLean et al., J Biomech Eng 2003
    % or a modified version
    xdot = (u/Tact + (1-u)/Tdeact) .* (u - x);

    if (nargout > 1)
        dxdot_dx = -(u/Tact + (1-u)/Tdeact);
        dxdot_du = (1/Tact - 1/Tdeact).*(u-x) + (u/Tact + (1-u)/Tdeact);
    end
end

