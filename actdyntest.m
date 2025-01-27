function actdyntest()
% simulate activation dynamics to show the "pumping" effect

    close all
    global problem

    % define the models to test
    % 'McLean2003' is McLean SG et al., J Biomech Eng 2003, eq. (3)
    % 'DeGroote2016' is De Groote F et al., Ann Biomed Eng 2016, eq. (1-2)
    models = {'McLean2003' 'DeGroote2016Original' 'DeGroote2016'};
    problem.Tact = 0.015;
    problem.Tdeact = 0.060;
    umax = 1.0;

    % excitation patterns are square waves with these frequencies (in Hz):
    frequencies = logspace(log10(1),log10(100),20);  % 1 to 100 Hz in 10 steps

    % run all simulations
    for imodel = 1:numel(models)
        model = models{imodel};
        figure();
        nfreq = numel(frequencies);
        isubplot=0;
        for ifreq = 1:nfreq
            frequency = frequencies(ifreq);
            uperiod = 1.0 / frequency;
            dutycycle = 0.5;
            % plot only three of the frequencies
            if (ifreq==1) || (ifreq==nfreq) || (ifreq==round(nfreq/2))
                isubplot = isubplot + 1;
                subplot(3,1,isubplot);
                makeplot = true;
                [~,~,meanact(ifreq,imodel)] = actsim(model, umax, uperiod, dutycycle, makeplot);
                if (ifreq==1)
                    legend('excitation','activation');
                end
            else
                % do a simulation but don't plot it
                makeplot = false;
                [~,~,meanact(ifreq,imodel)] = actsim(model, umax, uperiod, dutycycle, makeplot);
            end
        end
    end

    % plot mean activation as a function of input frequency
    figure
    semilogx(frequencies, meanact,'-o')
    legend(models)
    % ylim([0 1])
    xlabel('frequency (Hz)')
    ylabel('mean activation')
    title('response to square wave excitation')

end
