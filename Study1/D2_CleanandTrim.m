clear
close all
clc
%#ok<*SAGROW>
%#ok<*BDSCA>

%% Go to main folder

cd 'C:\Users\amalv\Desktop\Tinkerings\MaxEffort\MATLAB'

% Load IMU strut
load('IMU.mat')

% Load 'All' strut so we can save to it
load('All.mat')


%% Filter and cut to stance phase

% Assign things
Subject = IMU.ID;
% Subject = ["ME10", "ME15"];
Side = ["Paretic", "Nonparetic"];
Day = ["NS", "S"];
Condition = ["Pre", "Post"];
Segment = ["Thigh", "Shank", "Foot"];
Trial = ["Static", "Walk"];



% For each subject
for i = 1:length(Subject)
    
    % For each day (No Suit, Suit)
    for j = 1:length(Day)
        
        % For each condition (Pre, Post)
        for k = 1:length(Condition)
            
            % For each side (Paretic, Nonparetic)
            for l = 1:length(Side)
                
                
                sub = Subject(i);
                day = Day(j);
                con = Condition(k);
                side = Side(l);
                tri = Trial(2); % only use walking trials
                
                % Skip trials that don't have IMU data
                if sub == "ME03" && day == "NS" ||...
                        sub == "ME04" && day == "S" && con == "Post"||...
                        sub == "ME06" && day == "NS"||...
                        sub == "ME06" && day == "S" && con == "Post" ||...
                        sub == "ME07" && day == "NS" ||...
                        sub == "ME10" && day == "NS" && con == "Post" ||...
                        sub == "ME14" && day == "S" && con == "Post" ||...
                        sub == "ME15" && day == "S" && con == "Post"
                    
                    break
                    
                end
                
                % Combine acc and gyro data from all segments
                thigh = IMU.(sub).(day).(con).(side).Thigh.(tri).data(2:end,2:7);
                shank = IMU.(sub).(day).(con).(side).Shank.(tri).data(2:end,2:7);
                foot = IMU.(sub).(day).(con).(side).Foot.(tri).data(2:end,2:7);
                
                % Cut to final 30 seconds
                Fs = 100; % collection frequency
                trim = 30; % number of seconds being trimed
                last30 = trim*Fs; % number of data points
                
                thigh = thigh((end-last30):end, :);
                shank = shank((end-last30):end, :);
                foot = foot((end-last30):end, :);
                
                
                %% Flip necessary data
                % Right side IMUs will have their Y (Anterior/Posterior) and Z
                % (Medial/Lateral) flipped to match the lab coordinate system;
                % left side IMUs already match the lab
                
                % Check which side is paretic
                paretic = All.Paretic(i);
                
                % If paretic == Right, flip paretic side
                if paretic == "Right" && side == "Paretic"
                    
                    thigh(:,[2,3,5,6]) = -thigh(:,[2,3,5,6]);
                    shank(:,[2,3,5,6]) = -shank(:,[2,3,5,6]);
                    foot(:,[2,3,5,6]) = -foot(:,[2,3,5,6]);
                    
                % If paretic == Left, flip nonparetic side
                elseif paretic == "Left" && side == "Nonparetic"
                    
                    thigh(:,[2,3,5,6]) = -thigh(:,[2,3,5,6]);
                    shank(:,[2,3,5,6]) = -shank(:,[2,3,5,6]);
                    foot(:,[2,3,5,6]) = -foot(:,[2,3,5,6]);
                    
                end
                
                % Combine data into single matrix
                dataIn = horzcat(thigh, shank, foot);
                
                % Filter data using a lowpass, 2nd order
                % butterworth filter at 10 Hz
                nth = 2; % order
                Wn = (1/10); % filter frequency
                dataMid = IMUFilter(dataIn, nth, Wn); % filtered data
                % above uses custom function 'IMUFilter'
                
                % Trim data to stance phase
                dataOut = IMUStance(dataMid, sub, day, con ,side); % stance phase data
                % above uses custom function 'IMUStance'
                
                % Parse out accelerometer and gyro data
                All.(sub).(day).(con).(side).IMU.Thigh = ...
                    dataOut.thigh;
                All.(sub).(day).(con).(side).IMU.Shank = ...
                    dataOut.shank;
                All.(sub).(day).(con).(side).IMU.Foot = ...
                    dataOut.foot;
                
                % Get rid of extra fields
%                 fields = {'data', 'textdata', 'colheaders'};
%                 All.(sub).(day).(con).(side) = ...
%                     rmfield(All.(sub).(day).(con).(side), fields);
                
                
                
                clear sub day con side tri thigh shank foot paretic dataIN...
                    Fs trim last30 nth Wn dataMid dataOut
                
                
            end
            
        end
        
    end
    
end

%%
%%% save to avoid losing progress to another random restart...
%%% ... or me just being an idiot
save('IMU.mat', 'IMU')
save('All.mat', 'All')