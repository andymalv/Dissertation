clear all
close all
clc
%#ok<*SAGROW>
%#ok<*BDSCA>
%#ok<*VUNUS>
%#ok<*NASGU> 


%%

load('All.mat')

%% Subject specific models

Subject = ["ME04", "ME10", "ME14", "ME15"];
Side = ["Paretic", "Nonparetic"];
Metric = ["BrakeMag", "BrakeImpulse", "PropMag", "PropImpulse"];
Segment = ["Thigh", "Shank", "Foot"];

for i = 1:length(Subject)

    for j = 1:length(Side)

        for k = 1:length(Metric)

            % Assign things
            side = Side(j);
            met = Metric(k);
            pick = strcat(side,met);

            % Assign more things
            y = resp.(pick);
            xall = feat;

            % Feature selection
            [idx, scores] = fsrftest(xall, y);
            p = exp(-scores);
            sig = find(p < 0.05);
            x = xall(:, sig);

            % Determine test data
            testper = 0.7; % Percentage of data to be used for training
            n = length(y); % Total number of data
            testamnt = round(n * testper); % Total number of training data
            a = sort(randperm(n, testamnt))'; % Random selection of training data

            % Find which data weren't selected as training data
            b = [];

            for m = 1:n

                if isempty(find(a == m, 1))

                    b = vertcat(b, m);

                end

            end

            % Assign training and test variables
            xtrain = x(a,:);
            xtest = x(b,:);
            ytrain = y(a,:);
            ytest = y(b,:);

            % Train SVM model
            model = fitrsvm(xtrain, ytrain);

            % If model doesn't converge, standardize the variables
            if model.ConvergenceInfo.Converged ~= 1

                model = fitrsvm(xtrain, ytrain, 'Standardize', true);

            end

            % Get predicted values based on model and test data
            ypred = predict(model, xtest);

            % Calculate performance metrics
            RMSE = rmse(ypred, ytest);
            MAE = mae(ypred - ytest);
            MAPE = mape(ypred, ytest);

            [R, p] = corrcoef(ytest, ypred);
            R2 = R(1,2)^2;
            [r2, ~] = rsquare(ytest, ypred);

            ybar = mean(ytest);
            SStot = sum((ytest - ybar).^2);
            SSres = sum((ytest - ypred).^2);
            R_2 = 1 - (SSres/SStot);

            %% Save everything

            Models.(sub).(side).(met).model = model;
            Models.(sub).(side).(met).Predictors = model.ExpandedPredictorNames;
            Models.(sub).(side).(met).RMSE = RMSE;
            Models.(sub).(side).(met).MAE = MAE;
            Models.(sub).(side).(met).MAPE = MAPE;
            Models.(sub).(side).(met).R2 = array2table([r2 R_2 R2 p(2)]);

            % Display message for reasons
            disp(strcat(sub, " ", side, " ", met, " model complete."))

            % Clear variables after use
            clear pick idx scores p sig x y testper n testamnt a b xtrain...
                ytrain xtest ytest model ypred RMSE MAE MAPE R p R2 r2...
                R_2 ybar SStot SSres

        end

    end

    % Clear variables after use
    clear resp feat xall sub side met

end


%% General Model

% Assign things
Subject = All.ID;
Side = ["Paretic", "Nonparetic"];
Metric = ["BrakeMag", "BrakeImpulse", "PropMag", "PropImpulse"];

% Get all the features and responses together
presp = [];
npresp = [];
pfeat = [];
npfeat = [];

for i = 1:length(Subject)

    sub = Subject(i);

    presp = horzcat(presp, All.(sub).Responses.Paretic.APGRF);
    npresp = horzcat(npresp, All.(sub).Responses.Nonparetic.APGRF);

    pfeat = horzcat(pfeat, All.(sub).Features.Paretic.Combined);
    npfeat = horzcat(npfeat, All.(sub).Features.Nonparetic.Combined);

end


%%

% Train and test


% GEt number of strides
n = size(presp,2);

resp = horzcat(presp, npresp); %#ok<*NASGU> 
feat = horzcat(pfeat, npfeat);

% Assign more things
y = table(presp);
xall = feat;

y = All.ME04.Responses.Paretic.APGRF(:,1);
xall = All.ME04.Features.Paretic.Combined;

% Feature selection
[idx, scores] = fsrmrmr(xall, y);
p = exp(-scores);
sig = find(p < 0.05);
x = xall;%(:, sig);

% Determine test data
testper = 0.7; % Percentage of data to be used for training
n = length(y); % Total number of data
testamnt = round(n * testper); % Total number of training data
a = sort(randperm(n, testamnt))'; % Random selection of training data

% Find which data weren't selected as training data
b = [];

for m = 1:n

    if isempty(find(a == m, 1))

        b = vertcat(b, m);

    end

end

% Assign training and test variables
xtrain = x(:,a);
xtest = x(:,b);
ytrain = y(:,a);
ytest = y(:,b);

% Train SVM model
model = fitrsvm(xtrain, ytrain);

% If model doesn't converge, standardize the variables
if model.ConvergenceInfo.Converged ~= 1

    model = fitrsvm(xtrain, ytrain, 'Standardize', true);

end

% Get predicted values based on model and test data
ypred = predict(model, xtest);

% Calculate performance metrics
RMSE = rmse(ypred, ytest);
MAE = mae(ypred - ytest);
MAPE = mape(ypred, ytest);

[R, p] = corrcoef(ytest, ypred);
R2 = R(1,2)^2;
[r2, ~] = rsquare(ytest, ypred);

ybar = mean(ytest);
SStot = sum((ytest - ybar).^2);
SSres = sum((ytest - ypred).^2);
R_2 = 1 - (SSres/SStot);

%% Save everything

Models.General.(side).(met).model = model;
Models.General.(side).(met).Predictors = model.ExpandedPredictorNames;
Models.General.(side).(met).RMSE = RMSE;
Models.General.(side).(met).MAE = MAE;
Models.General.(side).(met).MAPE = MAPE;
Models.General.(side).(met).R2 = array2table([r2 R_2 R2 p(2)]);

% Display message for reasons
disp(strcat(side, " ", met, " model complete."))

% Clear variables after use
clear pick idx scores p sig x y testper n testamnt a b xtrain...
    ytrain xtest ytest model ypred RMSE MAE MAPE R p R2 r2...
    R_2 ybar SStot SSres



% Clear variables after use
clear resp feat xall sub side met




















