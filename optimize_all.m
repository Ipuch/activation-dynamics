function optimize_all()

    close all

    % settingss
    output = false;   % change this to true if you want to see the optimization progress

    % solve the "0.5 mean activation" task for all models
    models = {'McLean2003', 'McLean2003improved', 'DeGroote2016'};
    % models = {'McLean2003improved'};
    for i = 1:numel(models)
        p{i} = optimize(models{i}, output);      
    end

    % print a table with the optimal square wave parameters
    fprintf('--------------------------------------------------------\n')
    fprintf('model                  umin       umax        duty_cycle\n')
    fprintf('--------------------------------------------------------\n')
    for i = 1:numel(models)
        fprintf('%20s %8.4f %10.4f %11.4f\n', models{i}, p{i});
    end
    fprintf('--------------------------------------------------------\n')
end