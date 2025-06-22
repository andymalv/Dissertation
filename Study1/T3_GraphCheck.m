clear
close all
clc
%#ok<*SAGROW>

tic
%% Graph Check

% Load file w/ data
load('All.mat')

% Get number of subjects
Subject = All.ID;
Side = ["Paretic", "Nonparetic"];
Day = ["NS", "S"];
Condition = ["Pre", "Post"];

% Initialize labels
label.y = {'% Body Weight'};
label.x = {'% Stance'};
label.title.sub = {'Paretic', 'Nonparetic'};


%% Plot

% Plot APGRF curves w/ point metrics
for i = 1:length(Subject)
    
    sub = Subject(i);
    side = Side(1);
    
    % Open figure window
    fh = figure('visible', 'off');
    
    % Initialize titles
    label.title.main = sub; % main figure title
    sgtitle(label.title.main);
    hold on
    
    %% For each day and condition
    for j = 1:length(Day)
        
        day = Day(j);
        
        for k = 1:length(Condition)
            
            con = Condition(k);
            
            % Get subject data
            obj = All.(sub).(day).(con).(side).GRF.Y;
            
            % No suit, pre
            if j == 1 && k == 1
                
                m = 1;
                
                sph(m) = subplot(2,2,1); 
                set(gca, 'ColorOrder', 'factory');
                plot(obj)
                title('No Suit, Pre')
                xlabel(label.x)
                ylabel(label.y)
                grid on
                hold on
             
            % No suit, post    
            elseif j == 1 && k == 2
                
                m = 2;
                
                sph(m) = subplot(2,2,2);
                set(gca, 'ColorOrder', 'factory');
                plot(obj)
                title('No Suit, Post')
                xlabel(label.x)
                ylabel(label.y)
                grid on
                hold on
            
            % Suit, pre
            elseif j == 2 && k == 1
                
                m = 3;
                
                sph(m) = subplot(2,2,3);
                set(gca, 'ColorOrder', 'factory');
                plot(obj)
                title('Suit, Pre')
                xlabel(label.x)
                ylabel(label.y)
                grid on
                hold on
            
            % Suit, post
            elseif j == 2 && k == 2
                
                m = 4;
                
                sph(m) = subplot(2,2,4);
                set(gca, 'ColorOrder', 'factory');
                plot(obj)
                title('Suit, Post')
                xlabel(label.x)
                ylabel(label.y)
                grid on
                hold on
                
            end
            
            % Initialize y axis limits
            ymin = 0;
            ymax = 0;
            
            %% Mark peak braking and propulsion
            for n = 1:size(obj,2)
                
                % Mark peak braking
                x.brake = All.(sub).(day).(con).(side).GRF.metrics.brake.peak_time.raw(n);
                y.brake = All.(sub).(day).(con).(side).GRF.metrics.brake.peak_mag.raw(n);
                plot(x.brake, y.brake, 'bx', 'markersize', 10)
                
                % Mark peak propulsion
                x.prop = All.(sub).(day).(con).(side).GRF.metrics.prop.peak_time.raw(n);
                y.prop = All.(sub).(day).(con).(side).GRF.metrics.prop.peak_mag.raw(n);
                plot(x.prop, y.prop, 'rx', 'markersize', 10)
                
                % Mark propulsion onset
                x.onset = All.(sub).(day).(con).(side).GRF.metrics.prop.onset.raw(n);
                y.onset = 0;
                plot(x.onset, y.onset, 'gx', 'markersize', 10)
                
                % Get y axis limits
                if y.brake <= ymin
                    
                    ymin = y.brake;
                    
                end
                
                if y.prop >= ymax
                    
                    ymax = y.prop;
                    
                end
                
            end
            
            % Adjust all the axes
            linkaxes(sph, 'y')
            ax = axis;
            axis([ax(1:2) (ymin-5) (ymax+5)])
            
            %% Clear variables
            clear obj x y
            
        end
        
    end
    
    %% Save figure, then close
    print('-bestfit', label.title.main, '-dpdf')
    close(gcf)
    
    %% Clear all vairables
    clear obj x y label.title label.save xmin xmax ymin ymax fh
    
end

clc

%%
toc





