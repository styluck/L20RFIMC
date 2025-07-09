 %% ***************************************************************
%  filename: funV
%
%  Compute the gradient of the loss function f
%
%  f(U,V) = 0.5*||P_Omega(AUV'B'-M)||^2 
%
%  gradV = [A'*P_Omega(AUV'B'-M)*B]'*U 
%
%% ****************************************
%%  2025-01-24 by TaoTing
%% ****************************************

function [Loss,gradV] = funV(U,V,b,A,B,nzidx,PX)

UVt = U*V';        AXB =  A*UVt*B';

yk = AXB(nzidx) - b;

Loss = 0.5*norm(yk)^2;

if nargout>=2

    PX(nzidx) = yk;
  
    gradV =  (A'*PX*B)'*U;  

end

end
