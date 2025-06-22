clear
close all
clc
%#ok<*SAGROW>
%#ok<*BDSCA>

%% Set up

% Go to main folder
cd 'C:\Users\amalv\Desktop\Tinkerings\MaxEffort\MATLAB'

% Get ID and paretic side information
%%% this was done manually and put in a .csv file; just reading that
df = table2struct(readtable('Subject_Info.csv'));

IMU.ID(:,1) = string({df.ID});
IMU.Paretic(:,1) = string({df.PareticSide});
IMU.Mass(:,1) = [df.Weight];

% Assign things
Subject = IMU.ID;
% Subject = ["ME04", "ME10", "ME15"];
Side = ["Paretic", "Nonparetic"];
Day = ["NS", "S"];
Condition = ["Pre", "Post"];

%% Read in Ascii files

%%% reads in each file from ID list

Lthigh = "00B41CE8.txt";
Lshank = "00B41CE5.txt";
Lfoot = "00B41CE4.txt";
Rthigh = "00B41CB0.txt";
Rshank = "00B41CA6.txt";
Rfoot = "00B41CDA.txt";

for i = 1:length(Subject)
    
    % Get subject ID
    sub = Subject(i);
    
    % Assign file path where data is located
    folder.Data = strcat('C:\Users\amalv\Desktop\Tinkerings\MaxEffort\Data\');
    
    % Create file names
    left.thigh.pre.walk = strcat(sub, '_preCWS-000_', Lthigh);
    left.shank.pre.walk = strcat(sub, '_preCWS-000_', Lshank);
    left.foot.pre.walk = strcat(sub, '_preCWS-000_', Lfoot);
    right.thigh.pre.walk = strcat(sub, '_preCWS-000_', Rthigh);
    right.shank.pre.walk = strcat(sub, '_preCWS-000_', Rshank);
    right.foot.pre.walk = strcat(sub, '_preCWS-000_', Rfoot);
    
    left.thigh.post.walk = strcat(sub, '_postCWS-000_', Lthigh);
    left.shank.post.walk = strcat(sub, '_postCWS-000_', Lshank);
    left.foot.post.walk = strcat(sub, '_postCWS-000_', Lfoot);
    right.thigh.post.walk = strcat(sub, '_postCWS-000_', Rthigh);
    right.shank.post.walk = strcat(sub, '_postCWS-000_', Rshank);
    right.foot.post.walk = strcat(sub, '_postCWS-000_', Rfoot);
    
    
    %% Set up file paths
    folder.NS = strcat(folder.Data, sub, '\No Suit\IMUs\2 Exported Data');
    
    folder.S = strcat(folder.Data, sub, '\Suit\IMUs\2 Exported Data');
    
    %% Check which side is paretic
    paretic = IMU.Paretic(i);
    
    % For each day (No Suit, Suit)
    for j = 1:length(Day)
        
        % For each condition (Pre, Post)
        for k = 1:length(Condition)
            
            % Assign day and cond
            day = Day(j);
            con = Condition(k);
            
            % Skip trials that don't have IMU data
            if sub == "ME03" && day == "NS" ||...
                    sub == "ME06" && day == "NS"||...
                    sub == "ME06" && day == "S" && con == "Pre" ||...
                    sub == "ME07" && day == "NS" ||...
                    sub == "ME10" && day == "NS" && con == "Post" ||...
                    sub == "ME14" && day == "S" && con == "Post" ||...
                    sub == "ME15" && day == "S" && con == "Post"
                
                break
                
            end
            
            % Go to folder
            cd(folder.(day));
            
            %% Import files
            
            if paretic == 'Right'
                
                % Paretic Thigh
                IMU.(sub).(day).Pre.Paretic.Thigh.Walk = ...
                    importdata(right.thigh.pre.walk);
                IMU.(sub).(day).Post.Paretic.Thigh.Walk = ...
                    importdata(right.thigh.post.walk);
                
                % Paretic Shank
                IMU.(sub).(day).Pre.Paretic.Shank.Walk = ...
                    importdata(right.shank.pre.walk);
                IMU.(sub).(day).Post.Paretic.Shank.Walk = ...
                    importdata(right.shank.post.walk);
                
                % Paretic Foot
                IMU.(sub).(day).Pre.Paretic.Foot.Walk = ...
                    importdata(right.foot.pre.walk);
                IMU.(sub).(day).Post.Paretic.Foot.Walk = ...
                    importdata(right.foot.post.walk);
                
                % Nonparetic Thigh
                IMU.(sub).(day).Pre.Nonparetic.Thigh.Walk = ...
                    importdata(left.thigh.pre.walk);
                IMU.(sub).(day).Post.Nonparetic.Thigh.Walk = ...
                    importdata(left.thigh.post.walk);
                
                % Nonparetic Shank
                IMU.(sub).(day).Pre.Nonparetic.Shank.Walk = ...
                    importdata(left.shank.pre.walk);
                IMU.(sub).(day).Post.Nonparetic.Shank.Walk = ...
                    importdata(left.shank.post.walk);
                
                % Nonparetic Foot
                IMU.(sub).(day).Pre.Nonparetic.Foot.Walk = ...
                    importdata(left.foot.pre.walk);
                IMU.(sub).(day).Post.Nonparetic.Foot.Walk = ...
                    importdata(left.foot.post.walk);
                
                
            else
                
                % Nonparetic Thigh
                IMU.(sub).(day).Pre.Nonparetic.Thigh.Walk = ...
                    importdata(right.thigh.pre.walk);
                IMU.(sub).(day).Post.Nonparetic.Thigh.Walk = ...
                    importdata(right.thigh.post.walk);
                
                % Nonparetic Shank
                IMU.(sub).(day).Pre.Nonparetic.Shank.Walk = ...
                    importdata(right.shank.pre.walk);
                IMU.(sub).(day).Post.Nonparetic.Shank.Walk = ...
                    importdata(right.shank.post.walk);
                
                % Nonparetic Foot
                IMU.(sub).(day).Pre.Nonparetic.Foot.Walk = ...
                    importdata(right.foot.pre.walk);
                IMU.(sub).(day).Post.Nonparetic.Foot.Walk = ...
                    importdata(right.foot.post.walk);
                
                
                % Paretic Thigh
                IMU.(sub).(day).Pre.Paretic.Thigh.Walk = ...
                    importdata(left.thigh.pre.walk);
                IMU.(sub).(day).Post.Paretic.Thigh.Walk = ...
                    importdata(left.thigh.post.walk);
                
                % Paretic Shank
                IMU.(sub).(day).Pre.Paretic.Shank.Walk = ...
                    importdata(left.shank.pre.walk);
                IMU.(sub).(day).Post.Paretic.Shank.Walk = ...
                    importdata(left.shank.post.walk);
                
                % Paretic Foot
                IMU.(sub).(day).Pre.Paretic.Foot.Walk = ...
                    importdata(left.foot.pre.walk);
                IMU.(sub).(day).Post.Paretic.Foot.Walk = ...
                    importdata(left.foot.post.walk);
                
            end
            
            
        end
        
    end
    
    clear left right sub sns prpo folder paretic
    
end

%% Return to main folder

cd 'C:\Users\amalv\Desktop\Tinkerings\MaxEffort\MATLAB'


%%
%%% save to avoid losing progress to another random restart...
%%% ... or me just being an idiot
save('IMU.mat', 'IMU')
















