function p = uniformduct_linear_temp_bvp(xtest,omega,m,gamma,R,U0,T0)
solinit = bvpinit(xtest,@guess);
sol = bvp4c(@(x,y)bvpfcn(x,y,omega,m,gamma,R,U0,T0), @(ya,yb)bcfcn(ya,yb,omega,m,gamma,R,U0,T0), solinit);
p = sol.y(1,:);
end

function g = guess(x) % initial guess for y and y'
g = [sin(x);cos(x)];
end

function res = bcfcn(ya,yb,omega,m,gamma,R,U0,T0)
res = [ya(1)-U0(1); yb(1)-U0(2)];
end

function dydx = bvpfcn(x,y,omega,m,gamma,R,U0,T0)
dTdx = m;
T = T0+m*x;
Q = (omega^2)./(gamma*R*T);
dydx = [y(2); -(dTdx./T).*y(2)-Q.*y(1)];
end