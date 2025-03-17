% Program to predict the acoustic field in 1-D uniform duct with linear
% temperature gradient along the axial direction

clc;
clear all;

%% Generate training data
freq = 500;             % Frequency
gamma = 1.4;            % Specific heat ratio
R = 287;                % Universal gas constant
T0 = 1600;              % Temperature at the inlet
m = -800;               % Slope of T = T0+mx

omega = 2*pi*freq;      % Angular frequency

x = [0 1];              % Domain

L = x(2)-x(1);          % Length of the domain


x0BC1 = x(1);           % Boundary coordinates
x0BC2 = x(2);

u0BC1 = 1;              % Boundary values
u0BC2 = -1;

X0 = [x0BC1 x0BC2];
U0 = [u0BC1 u0BC2];

numInternalCollocationPoints = 1000;                    % No. of collocation points
    
pointSet = sobolset(1,"Skip",1);                        % Base-2 digital sequence that fills space in a highly uniform manner
points = net(pointSet,numInternalCollocationPoints);    % Generates quasirandom point set


dataX = (x(2)-x(1))*points+x(1);                % Creates random x-data points between x1 and x2

%% Define deep learning model
numLayers = 5;
numNeurons = 90;
maxFuncEvaluations = 100;
maxIterations = 100;

parameters = buildNet(numLayers,numNeurons);

%% Specify optimization options
options = optimoptions("fmincon", ...
    HessianApproximation="lbfgs", ...
    MaxIterations=maxIterations, ...
    MaxFunctionEvaluations=maxFuncEvaluations, ...
    OptimalityTolerance=1e-5, ...
    SpecifyObjectiveGradient=true, ...
    Display='iter');
%% Train network for acoustic pressure
start = tic;

[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = extractdata(parametersV);

%% Convert the variables into deep-learning variables
X = dlarray(dataX,"BC");
X0 = dlarray(X0,"CB");
U0 = dlarray(U0,"CB");

objFun = @(parameters) objectiveFunction(parameters,X,X0,U0,parameterNames,parameterSizes,omega,gamma,m,R,T0);

parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);

parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

toc(start)

%% Evaluate model accuracy
numPredictions = 500;                           % No.of test points
XTest = linspace(x(1),x(2),numPredictions);

dlXTest = dlarray(XTest,'CB');
U = model(parameters,dlXTest);
W1_test = (X0(2)-dlXTest)/(X0(2)-X0(1));
W2_test = (dlXTest-X0(1))/(X0(2)-X0(1));
phi_eqv_test = W1_test.*W2_test;
dlUPred = W1_test*U0(1)+W2_test*U0(2)+phi_eqv_test.*U;

% Calcualte true values.
UTest = uniformduct_linear_temp_bvp(XTest,omega,m,gamma,R,extractdata(U0),T0);
% Calculate error.
err = norm(extractdata(dlUPred) - UTest) / norm(UTest);

f1 = figure;

% Plot predictions.
plot(XTest,extractdata(dlUPred),'-','LineWidth',2);
% ylim([-1.1, 1.1])

% Plot true values.
hold on
plot(XTest, UTest, '--','LineWidth',2)
hold off

xlabel('x','FontSize',14,'FontWeight','bold')
ylabel('p','FontSize',14,'FontWeight','bold')
title("Frequency = " + freq + " Hz;" + " Relative Error = " + gather(err));

legend('Predicted','True')

