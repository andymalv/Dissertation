clear all
close all
clc
%#ok<*SAGROW>
%#ok<*BDSCA>
%#ok<*VUNUS>

%%

load('All.mat')

Subject = All.ID;
Side = ["Paretic", "Nonparetic"];
Day = ["NS", "S"];
Condition = ["Pre", "Post"];
Segment = ["Thigh", "Shank", "Foot"];
Trial = ["Walk"];

%% Create subject specific models

% Table headers
tblhead = ["ThighAccX", "ThighAccY", "ThighAccZ", "ThighGyroX", "ThighGyroY", "ThighGyroZ",...
    "ShankAccX", "ShankAccY", "ShankAccZ", "ShankGyroX", "ShankGyroY", "ShankGyroZ",...
    "FootAccX", "FootAccY", "FootAccZ", "FootGyroX", "FootGyroY", "FootGyroZ"];

% Initialize strut variables
for i = 1:length(Subject)

    for j = 1:length(Side)

        for m = 1:length(Segment)

            sub = Subject(i);
            side = Side(j);
            seg = Segment(m);

            All.(sub).Features.(side).(seg).AccX = [];
            All.(sub).Features.(side).(seg).AccY = [];
            All.(sub).Features.(side).(seg).AccZ = [];
            All.(sub).Features.(side).(seg).GyroX = [];
            All.(sub).Features.(side).(seg).GyroY = [];
            All.(sub).Features.(side).(seg).GyroZ = [];



        end

        All.(sub).Responses.(side).APGRF = [];
        All.(sub).Responses.(side).BrakeMag = [];
        All.(sub).Responses.(side).BrakeImpulse = [];
        All.(sub).Responses.(side).PropMag = [];
        All.(sub).Responses.(side).PropImpulse = [];

    end

end

%%
% Create strut w/ all features
for i = 1:length(Subject)

    for j = 1:length(Day)

        for k = 1:length(Condition)

            % Use try/catch to skip trials that dont' have IMU data
            try

                for l = 1:length(Side)

                    for m = 1:length(Segment)

                        sub = Subject(i);
                        day = Day(j);
                        con = Condition(k);
                        side = Side(l);
                        seg = Segment(m);

                        % Grab IMU data (features)
                        AccX = All.(sub).(day).(con).(side).IMU.(seg).Acc.X;
                        AccY = All.(sub).(day).(con).(side).IMU.(seg).Acc.Y;
                        AccZ = All.(sub).(day).(con).(side).IMU.(seg).Acc.Z;
                        GyroX = All.(sub).(day).(con).(side).IMU.(seg).Gyro.X;
                        GyroY = All.(sub).(day).(con).(side).IMU.(seg).Gyro.Y;
                        GyroZ = All.(sub).(day).(con).(side).IMU.(seg).Gyro.Z;

                        % Add IMU data to subject specific structs
                        All.(sub).Features.(side).(seg).AccX =...
                            horzcat(All.(sub).Features.(side).(seg).AccX,...
                            AccX);
                        All.(sub).Features.(side).(seg).AccY =...
                            horzcat(All.(sub).Features.(side).(seg).AccY,...
                            AccY);
                        All.(sub).Features.(side).(seg).AccZ =...
                            horzcat(All.(sub).Features.(side).(seg).AccZ,...
                            AccZ);

                        All.(sub).Features.(side).(seg).GyroX =...
                            horzcat(All.(sub).Features.(side).(seg).GyroX,...
                            GyroX);
                        All.(sub).Features.(side).(seg).GyroY =...
                            horzcat(All.(sub).Features.(side).(seg).GyroY,...
                            GyroY);
                        All.(sub).Features.(side).(seg).GyroZ =...
                            horzcat(All.(sub).Features.(side).(seg).GyroZ,...
                            GyroZ);

                        % Clear relevant variables after use
                        clear AccX AccY AccZ GyroX GyroY GyroZ Brake Prop

                    end

                    % Grab GRF metrics (responses)
                    BrakeMag = All.(sub).(day).(con).(side).GRF.metrics.brake.peak_mag.raw;
                    BrakeImp = All.(sub).(day).(con).(side).GRF.metrics.brake.impulse.raw;
                    PropMag = All.(sub).(day).(con).(side).GRF.metrics.prop.peak_mag.raw;
                    PropImp = All.(sub).(day).(con).(side).GRF.metrics.prop.impulse.raw;

                    % Grab APGRF (responses)
                    APGRF = All.(sub).(day).(con).(side).GRF.Y;

                    % Add metrics to subject specific struct
                    All.(sub).Responses.(side).BrakeMag = ...
                        vertcat(All.(sub).Responses.(side).BrakeMag, BrakeMag);
                    All.(sub).Responses.(side).BrakeImpulse = ...
                        vertcat(All.(sub).Responses.(side).BrakeImpulse, BrakeImp);
                    All.(sub).Responses.(side).PropMag = ...
                        vertcat(All.(sub).Responses.(side).PropMag, PropMag);
                    All.(sub).Responses.(side).PropImpulse = ...
                        vertcat(All.(sub).Responses.(side).PropImpulse, PropImp);
                    All.(sub).Responses.(side).APGRF = ...
                        horzcat(All.(sub).Responses.(side).APGRF, APGRF);

                    % Clear relevant variables after use
                    clear BrakeMag BrakeImp PropMag PropImp BrakeTime PropTime

                end

            catch

                % If there is an error - most likely to due w/ not having
                % IMU data for this trial - display message
                disp(strcat(sub, " does not have IMU data for the ",...
                    day, " ", con, " condition."))

            end % for try/catch



        end

    end

end


%% Rearrange data for ease of use

Subject = All.ID;
Side = ["Paretic", "Nonparetic"];
Day = ["NS", "S"];
Condition = ["Pre", "Post"];
Segment = ["Thigh", "Shank", "Foot"];

% For each subject
for i = 1:length(Subject)

    % For each side (Paretic, Nonparetic)
    for l = 1:length(Side)

        % Assign things
        sub = Subject(i);
        side = Side(l);

        allsgm = []; % Initialize variable

        % For each segment (Thigh, Shank, Foot)
        for m = 1:length(Segment)

            % Assign things
            seg = Segment(m);

            % Set up variable names for table columns
            VarNames = cellstr({strcat(sub, side, seg, "AccX"), ...
                strcat(sub, side, seg, "AccY"), strcat(sub, side, seg, "AccZ"),...
                strcat(sub, side, seg, "GyroX"), strcat(sub, side, seg, "GyroY"),...
                strcat(sub, side, seg, "GyroZ")});

            % Get data for this segment in this direction and assign
            % names
            sgm = struct2table(All.(sub).Features.(side).(seg));
            sgm.Properties.VariableNames = VarNames;

            % Combine this w/ other segments
            allsgm = horzcat(allsgm, sgm);

            % Clear variables after use
            clear sgm VarNames

        end

        % Assign data from both directions to struct
        All.(sub).Features.(side).Combined = allsgm;

        % Clear variables after use
        clear allsgm

    end


end










