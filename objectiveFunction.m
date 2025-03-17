function [loss,gradientsV] = objectiveFunction(parametersV,X,X0,U0,parameterNames,parameterSizes,omega,gamma,m,R,T0)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model loss and gradients.
[loss,gradients] = dlfeval(@modelLoss,parameters,X,X0,U0,omega,gamma,m,R,T0);

% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end
