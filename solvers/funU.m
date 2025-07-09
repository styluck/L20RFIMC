 %% ***************************************************************
%  filename: funU
%
%  Compute the gradient of the loss function f
%
%  f(U,V) = 0.5*||P_Omega(AUV'B')-b||^2 
%
%  grad_U f(U,V) = A'*P_Omega(AUV'B'-M)*B*V
%
%% ****************************************
%%  2025-01-24 by TaoTing
%% ****************************************
function  [Loss,gradU] = funU(U,V,b,A,B,nzidx,PX)

UVt = U*V';        AXB =  A*UVt*B';

yk = AXB(nzidx) - b;

Loss = 0.5*norm(yk)^2;

if nargout>=2
    
  PX(nzidx) = yk;
  
  gradU =  A'*(PX*(B*V));  

end

end
