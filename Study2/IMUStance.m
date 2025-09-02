function dataOut = IMUStance(dataIn, sub, day, con, side)

% Parse out input variable
thigh = dataIn(:,1:6);
shank = dataIn(:,7:12);
foot = dataIn(:,13:18);

% Get vertical acc of the foot for stance trimming
footX = foot(:,1);

% Find negative peaks; must be >30% of the max negative peak
[pk_neg, loc_neg] = findpeaks(-footX, 'MinPeakDistance', 10,...
    'MinPeakHeight', max(-footX)*0.3);

% Find positive peaks
[pk_pos, loc_pos] = findpeaks(footX, 'MinPeakDistance', 1,...
    'MinPeakHeight', 1);

%% Exploit the pattern in foot vertical acc data
% to determine initial contact (IC) and toe off (TO)

% Stance phase starts w/ IC, so if there's any TOs before the first IC, get
% rid of them

b=1; % Counting variable

% For all the positive peaks
for a = 1:length(loc_pos)
    
    % If there's a positive peak before the first negative peak
    if loc_pos(a) < loc_neg(1)
        
        % Adjust counter
        b=b+1;
        
    end
    
end

% Trim vector of positive peaks to start after the first negative peak
loc_pos = loc_pos(b:end);

% Stance phase ends w/ TO, so if there are any ICs after the last TO, get
% rid of them

b = 0; % Counting variable
flip_neg = flip(loc_neg); % Flip the neg peak array so that the loop runs through in reverse order

% For each negative peak (in reverse order)
for a = 1:length(flip_neg)
    
    % If there's a negative peak after the last positive peak
    if flip_neg(a) >= loc_pos(end)
        
        % Adjust counter
        b=b+1;
        
    end
    
end

% Trim vector of negative peaks to end after the last positive peak
loc_neg = loc_neg(1:end-b);

%%
b = 1; % Counting variable
negdist = 65; % Minimum number of points b/w negative peaks
posdist = 50; % Minimum number of points b/w negative and positive peaks

% For each negative peak
for a = 2:length(loc_neg)
    
    
    % If more than a certain number of points pass b/w this negative peak
    % and the next one, this peak is the IC
    if loc_neg(a) >= (loc_neg(a-1) + negdist)
        
        cut(b,1) = loc_neg(a-1); % IC
        
        % Find the first positive peak after IC, at least 25 points away
        here = find(loc_pos >= loc_neg(a-1) + posdist, 1);
        
        % Mark that positive peak as TO
        cut(b,2) = loc_pos(here);
        
        
        
        % Some checks based on visual inspection
        if a > 2 && b > 2
            
            % If the last TO is the same as this TO, skip
            if cut(b,2) == cut(b-1,2)
                
                cut(b,:) = [];
                
            end
            
            % If this IC comes before the last TO, skip last IC and TO
            if cut(b,1) <= cut(b-1,2)
                
                cut(b-1,:) = [];

            end
            
        end
        
        b=b+1; % Adjust counting variable
        
    end
    
    
    
end

% Because 'cut' ends up w/ zeros sometimes and I can't figure out why
cut = [nonzeros(cut(:,1)) nonzeros(cut(:,2))];

%% Graph check

fh = figure('visible', 'off');

plot(footX)
hold on
title(strcat(sub, day, con, side, 'Foot Vert Acc'))
for a = 1:length(cut)
    
    plot(cut(a,1), footX(cut(a,1)), 'rx', 'markersize', 10) % IC
    plot(cut(a,2), footX(cut(a,2)), 'gx', 'markersize', 10) % TO

end

print('-bestfit', strcat(sub, day, con, side), '-dpdf')
close gcf


%% Trim to stance phase
for a = 1:size(cut,1)
    
    % Get IC and TO for this stride
    IC = cut(a,1);
    TO = cut(a,2);
    
    % Cut segment data to this stride
    Fcut = foot(IC:TO, :);
    Scut = shank(IC:TO, :);
    Tcut = thigh(IC:TO, :);
    
    % Resample to 100 points and assign foot data to strut
    df.foot.Acc.X(:,a) = resample(Fcut(:,1), 100, length(Fcut));
    df.foot.Acc.Y(:,a) = resample(Fcut(:,2), 100, length(Fcut));
    df.foot.Acc.Z(:,a) = resample(Fcut(:,3), 100, length(Fcut));
    
    df.foot.Gyro.X(:,a) = resample(Fcut(:,4), 100, length(Fcut));
    df.foot.Gyro.Y(:,a) = resample(Fcut(:,5), 100, length(Fcut));
    df.foot.Gyro.Z(:,a) = resample(Fcut(:,6), 100, length(Fcut));
    
    % Resample to 100 points and assign shank data to strut
    df.shank.Acc.X(:,a) = resample(Scut(:,1), 100, length(Scut));
    df.shank.Acc.Y(:,a) = resample(Scut(:,2), 100, length(Scut));
    df.shank.Acc.Z(:,a) = resample(Scut(:,3), 100, length(Scut));
    
    df.shank.Gyro.X(:,a) = resample(Scut(:,4), 100, length(Scut));
    df.shank.Gyro.Y(:,a) = resample(Scut(:,5), 100, length(Scut));
    df.shank.Gyro.Z(:,a) = resample(Scut(:,6), 100, length(Scut));
    
    % Resample to 100 points and assign thigh data to strut
    df.thigh.Acc.X(:,a) = resample(Tcut(:,1), 100, length(Tcut));
    df.thigh.Acc.Y(:,a) = resample(Tcut(:,2), 100, length(Tcut));
    df.thigh.Acc.Z(:,a) = resample(Tcut(:,3), 100, length(Tcut));
    
    df.thigh.Gyro.X(:,a) = resample(Tcut(:,4), 100, length(Tcut));
    df.thigh.Gyro.Y(:,a) = resample(Tcut(:,5), 100, length(Tcut));
    df.thigh.Gyro.Z(:,a) = resample(Tcut(:,6), 100, length(Tcut));
    
    
    
end


%% Output data
dataOut = df;


end