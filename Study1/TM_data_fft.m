% FFT analysis of TM data
% A.Aiello 10/26/2022

d= -FES01_preTM_0001.Force(8).Force(2,:);  %right belt APGRF
fs = FES01_preTM_0001.Force(8).Frequency;

d= -ME15_preCWS.Force(8).Force(2,:);
fs = ME15_preCWS.Force(8).Frequency;


t = 1/fs*[0:length(d)-1];
f = fs*[0:length(d)/2]/length(d);

figure
p1 = subplot(2,1,1); hold on
plot(t,d), xlabel('Time (s)'), xlim([0,t(end)])

Y = fft(d);
P2 = abs(Y/length(d));      %two-sided spectrum
P1 = P2(1:length(d)/2+1);
P1(2:end-1) = 2*P1(2:end-1);    %one-sided spectrum
[pk,lc] = findpeaks(P1,'MinPeakDistance',100,'MinPeakProminence',1.5);
p2 = subplot(2,1,2); hold on
plot(f,P1), xlabel('Frequency (Hz)'), xlim([0,20])
scatter(f(lc),pk,'rx','LineWidth',1.5)
frequency_peaks = f(lc)

freq = frequency_peaks(9);
pleg = scatter(p1,1/freq*[0:length(d)-1],zeros(1,length(d)),'k|','LineWidth',1.5);
legend(pleg,['Frequency ',num2str(freq)])


%%




figure
for n = 1:2

    if n == 1

        d= -FES01_preTM_0001.Force(8).Force(2,:);  %right belt APGRF
        fs = FES01_preTM_0001.Force(8).Frequency;

    elseif n == 2

        d = -ME15_preCWS.Force(8).Force(2,:);
        fs = ME15_preCWS.Force(8).Frequency;

    end

    t = 1/fs*(0:length(d)-1);
    f = fs*(0:length(d)/2)/length(d);

    Y = fft(d);
    P2 = abs(Y/length(d));      %two-sided spectrum
    P1 = P2(1:length(d)/2+1);
    P1(2:end-1) = 2*P1(2:end-1);    %one-sided spectrum
    [pk,lc] = findpeaks(P1,'MinPeakDistance',100,'MinPeakProminence',1.5);

    p2 = subplot(2,1,n); 
    hold on
    plot(f,P1), xlabel('Frequency (Hz)'), xlim([0,20])
    scatter(f(lc),pk,'rx','LineWidth',1.5)

end

subplot(2,1,1)
title('Ashlyns Data')
subplot(2,1,2)
title('Andys Data')

