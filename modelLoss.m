function [loss,gradients] = modelLoss(parameters,X,X0,U0,omega,gamma,m,R,T0)
% Make predictions with the initial conditions.
dTdx = m;
T = T0+m*X;
Q = (omega^2)./(gamma*R*T);

U = model(parameters,X);
W1 = (X0(2)-X)/(X0(2)-X0(1));
W2 = (X-X0(1))/(X0(2)-X0(1));

phi_eqv = W1.*W2;

% Trial neural network
G = W1*U0(1)+W2*U0(2)+phi_eqv.*U;

% Calculate derivatives with respect to X.
Gx = dlgradient(sum(G,'all'),X,'EnableHigherDerivatives',true);

% Calculate second-order derivatives with respect to X.
Gxx = dlgradient(sum(Gx,'all'),X,'EnableHigherDerivatives',true);

% Calculate loss.
f = Gxx+(dTdx./T).*Gx+Q.*G;

zeroTarget = zeros(size(f),"like",f);
loss = l2loss(f, zeroTarget);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
