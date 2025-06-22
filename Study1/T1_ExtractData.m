clear
close all
clc
%#ok<*SAGROW>
%#ok<*BDSCA>

%%

cd 'C:\Users\amalv\Desktop\Tinkerings\MaxEffort\MATLAB'

% Get ID and paretic side information
%%% this was done manually and put in a .csv file; just reading that
df = table2struct(readtable('Subject_Info.csv'));

All.ID(:,1) = string({df.ID});
All.Paretic(:,1) = string({df.PareticSide});
All.Mass(:,1) = [df.Weight];

Subject = All.ID;
Side = ["Paretic", "Nonparetic"];
Day = ["NS", "S"];
Condition = ["Pre", "Post"];

% n = length(Subject);
% m = length(Side);
% o = length(Day);
% p = length(Condition);


%% Read in Ascii files

%%% reads in each file from ID list

% folder.Work = 'C:\Users\amalv\Desktop\Tinkerings\MaxEffort\MATLAB';


for i = 1:length(Subject)
    
    sub = Subject(i);
    folder.Data = 'C:\Users\amalv\Desktop\Tinkerings\MaxEffort\Data\';
    
    % Create file names
    file1 = strcat(sub, '_Left.txt');
    file2 = strcat(sub, '_Right.txt');
    
    % Set up file paths
    folder.NS.Pre = strcat(folder.Data, sub, '\No Suit\4 Exported Data\Pre');
    folder.NS.Post = strcat(folder.Data, sub, '\No Suit\4 Exported Data\Post');
    
    folder.S.Pre = strcat(folder.Data, sub, '\Suit\4 Exported Data\Pre');
    folder.S.Post = strcat(folder.Data, sub, '\Suit\4 Exported Data\Post');
    
    % Check which side is paretic
    paretic = All.Paretic(i);
    
    for j = 1:length(Day)

        for k = 1:length(Condition)
            
            day = Day(j);
            cond = Condition(k);
            
            cd(folder.(day).(cond));
            
            % Import files
            left = importdata(file1);
            right = importdata(file2);
            
            if paretic == 'Right'
                
                All.(sub).(day).(cond).Paretic = right;
                All.(sub).(day).(cond).Nonparetic = left;
                
            else
                
                All.(sub).(day).(cond).Paretic = left;
                All.(sub).(day).(cond).Nonparetic = right;
                
            end
            

        end
     
    end

    clear file1 file2 left right sub sns prpo folder paretic
    
end


cd 'C:\Users\amalv\Desktop\Tinkerings\MaxEffort\MATLAB'


%% Extract and Normalize GRFs

% g = 9.8;

for i = 1:length(Subject) % subject
    
    sub = Subject(i);
    
    for j = 1:length(Day) % day
        
        day = Day(j);
        
        for k = 1:length(Condition) % condition
            
            con = Condition(k);
            
            for l = 1:length(Side) % side
                
                side = Side(l);
                
                % Get data and replace NaNs w/ zeros
                blah = All.(sub).(day).(con).(side).data(:,2:end);
                blah(isnan(blah)) = 0;
                
                % Initialize counter variable
                m = 1;
                
                % For each stride (set of x, y and z)
                for n = 1:3:size(blah,2)
                    
                    % Get single stride
                    stride = blah(:,n:(n+2)); 
                    
                    % Remove zeros from arrays
                    x = nonzeros(stride(:,1));
                    y = nonzeros(stride(:,2));
                    z = nonzeros(stride(:,3));
                    
                    % Resample to 100 points
                    X = resample(x,100,length(x));
                    Y = resample(y,100,length(y));
                    Z = resample(z,100,length(z));
                    
                    % Move to structure
                    All.(sub).(day).(con).(side).GRF.X(:,m) = X*100;
                    All.(sub).(day).(con).(side).GRF.Y(:,m)  = Y*100;
                    All.(sub).(day).(con).(side).GRF.Z(:,m)  = Z*100;
                    
                    % Add to counter
                    m = m+1;
                    
                end
                
                clear stride m x y z X Y Z
                
            end
            
            clear side blah
            
        end
        
        clear con
        
    end
    
    clear day
    
end

clear sub g


%%
%%% save to avoid losing progress to another random restart...
%%% ... or me just being an idiot
% save('All.mat', 'All')















