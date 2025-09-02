function dataOut = IMUStance(dataIn)

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

b = 1; % Counting variable
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


% ICs should be right (will check visually), but the TOs are tricky due to
% multiple peaks; so we're going to choose the second peak following the IC
% to be the TO for that stride (again, w/ visual inspection)

% For each IC
for a = 1:length(loc_neg)
    
    % Mark IC as start of stance
    cut(a,1) = loc_neg(a);
    
    % Find the first positive peak after IC, at least 25 points away
    here = find(loc_pos >= loc_neg(a) + 50, 1);
    
    % Mark that positive peak as TO
    cut(a,2) = loc_pos(here);
        


end


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