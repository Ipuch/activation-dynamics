function [mean_u, mean_usquared, mean_a] = actsim(model,umax,uperiod,dutycycle,makeplot)
% simulate activation dynamics with a square wave input
    global problem

    % if used without input, do one simulation with these settings:
    if nargin == 0
        problem.model = 'McLean2003';  % options are 'Mclean2003' or 'DeGroote2016Original' or 'DeGroote2016'
        problem.Tact = 0.015;
        problem.Tdeact = 0.060;
        problem.uperiod = 0.10;        % for excitation, use a square wave with this period
        problem.dutycycle = 0.5;
        problem.umax = 1.0;            
        close all 
        makeplot = true;
    else
        % use settings provided from input arguments
        problem.umax     = umax;
        problem.uperiod  = uperiod;
        problem.dutycycle = dutycycle;
        problem.model    = model;
    end
    name = sprintf('%s with %.3f Hz square wave input', problem.model, 1.0/problem.uperiod);

    % simulate for about 5 seconds, to make sure there is no effect of startup transient
    duration = 5.0;
    ncycles = round(duration/problem.uperiod);
    aa = 0;  % initial activation state
    tt = 0;  % initial time value
    uu = 0;  % initial excitation input
    for i = 1:ncycles
        % simulate the activation phase of the cycle, with excitation at umax
        problem.u = problem.umax;
        dur = problem.dutycycle*problem.uperiod; % duration of this phase
        [t,a] = ode45(@actdyn,[0 dur],aa(end));
        % append results to the tt,aa,uu vectors
        % add a tiny amount to the first time value, otherwise interp1 (later) will complain about duplicate time points
        t(1) = t(1) + 1e-10;
        tt = [tt ; tt(end) + t];
        aa = [aa ; a]; 
        uu = [uu ; problem.u*ones(size(a))];

        % simulate the deactivation phase of the cycle, with excitation at 0
        problem.u = 0;
        dur = (1 - problem.dutycycle)*problem.uperiod; % duration of this phase
        [t,a] = ode45(@actdyn,[0 dur],aa(end));
        % append results to the tt,aa,uu vectors
        % add a tiny amount to the first time value, otherwise interp1 (later) will complain about duplicate time points
        t(1) = t(1) + 1e-10;
        tt = [tt ; tt(end) + t];
        aa = [aa ; a]; 
        uu = [uu ; problem.u*ones(size(a))];
        
    end

    % plot u(t) and a(t) for the final 10 periods
    if (makeplot)
        plot(tt,[uu aa]);
        xlabel('time (s)')
        duration = tt(end);
        tstart = max(0,duration-10*problem.uperiod);
        xlim([tstart duration])
        ylim([-0.2 1.2])
        title(name)
        if (nargin==0)
            legend('excitation','activation')
        end
    end

    % resample and calculate the means over the final cycle
    npoints = 1001;
    tnew = linspace(tt(end)-problem.uperiod, tt(end), npoints);
    u = interp1(tt,uu,tnew);
    a = interp1(tt,aa,tnew);
    % exclude the last resampled point from the averages, since it's the start of the next cycle
    mean_u = mean(u(1:end-1));
    mean_usquared = mean(u(1:end-1).^2);
    mean_a = mean(a(1:end-1));

    if isnan(mean_u) keyboard, end;

    % report the means
    fprintf('Simulation result for %s\n', name)
    fprintf('    mean excitation:         %7.4f\n', mean_u)
    fprintf('    mean excitation-squared: %7.4f\n', mean_usquared)
    fprintf('    mean activation:         %7.4f\n', mean_a)

end
%================================================================
function [adot] = actdyn(t,a)
    global problem

    % time constants for activation and deactivation
    Tact   = problem.Tact;
    Tdeact = problem.Tdeact; 

    % control input u
    u = problem.u;

    if strcmp(problem.model, 'McLean2003')
        % activation dynamics model from McLean et al., J Biomech Eng 2003
        adot = (u/Tact + (1-u)/Tdeact) .* (u - a);
    elseif strcmp(problem.model, 'DeGroote2016Original')
        % equation (1) from De Groote et al 2016, original version as published
        b = 0.1;
        f = 0.5*tanh((u-a)*b);  
        % then equation (2) from De Groote et al., 2016
        adot = ( 1/Tact/(0.5+1.5*a)*(f+0.5) + (0.5+1.5*a)/Tdeact*(-f+0.5) ) * (u-a);
    elseif strcmp(problem.model, 'DeGroote2016')
        % equation (1) from De Groote et al 2016, but with b=10
        b = 10;
        f = 0.5*tanh((u-a)*b);  
        % then equation (2) from De Groote et al., 2016
        adot = ( 1/Tact/(0.5+1.5*a)*(f+0.5) + (0.5+1.5*a)/Tdeact*(-f+0.5) ) * (u-a);
    else
        error('unknown model: %s', settings.model)
    end

end
