 %% ***************************************************************
%  filename: funUV
%
%  Compute the gradient of the loss function f
%
%  f(U,V) = 0.5*||P_Omega(AUV'B')-b||^2 
%
%  gradU f(U,V) = A'*P_Omega(AUV'B'-M)*B*V 
%
%  gradV = [A'*P_Omega(AUV'B'-M)*B]'*U 
%
%% ****************************************
%%  2025-01-24 by TaoTing
%% ****************************************

function  [X,AXB,Loss,gradU,gradV] = funUV(U,V,b,A,B,nzidx,PX)

X = U*V';     AXB = A*X*B';     yk = AXB(nzidx)-b;  

Loss = 0.5*norm(yk)^2;

if nargout>=3

    PX(nzidx) = yk;

    gradU =  A'*(PX*(B*V));   

    gradV = (A'*PX*B)'*U; 
    
end