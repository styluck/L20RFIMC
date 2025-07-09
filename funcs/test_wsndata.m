function [ERR_wmc,ERR_rmc,ERR_imc] = test_wsndata(name,recoveryCriterion,sr,exp_times,isSaving)

%%


isDisp =0;
p_num = numel(sr);

% Preallocate result matrices
re_pamm = zeros(p_num, exp_times);
re_pammse = zeros(p_num, exp_times);
time_pamm = zeros(p_num, exp_times); % Matrix to store computational times for PAMM
time_pammsc = zeros(p_num, exp_times); % Matrix to store computational times for PAMM_SC

load('intel_hum.mat');
Rawdata = Mdata;


Mdata=Mdata(:,1:288);
[m,n]=size(Mdata);
A = eye(m);
B = dctmtx(288)';
B = B(:,1:30);


for i = 1:p_num    
    p = sr(i);
    
    parfor j = 1:exp_times
        
        offset = randsample(288*9,1);
        Mdata = Rawdata(:,1+offset:288+offset);
        Omega_idx = rand(m,n);
        Omega = Omega_idx>1-p;
        Obs = Mdata.*Omega;
        
        
        try
            L = DirtyIMC(Obs,find(Omega),A,B,0.1,0.1,10,10);
            err = norm((L-Mdata),'fro')/norm(Mdata,'fro');
            ERR_dirtyImc(i,j) = err;
            if err < recoveryCriterion
                SUC_dirtyImc(i,j) = 1;
            end 
        catch ErrorInfo
            disp(ErrorInfo);
        end
            
        fprintf('sample rate = %f, seq = %d \n',p,j);
    end   
end


%% plotings



figure('numbertitle','off','name','SuccessRate_vs_p');
plot(sr(:),100.0*sum(SUC_dirtyImc,2)/exp_times,'-.k','MarkerSize',8,'linewidth',1);
hold on;
plot(sr(:),100.0*sum(SUC_imc,2)/exp_times,'-m', 'MarkerSize',8,'linewidth',1);
plot(sr(:),100.0*sum(SUC_rmc,2)/exp_times,'-xr','MarkerSize',8,'linewidth',1);
plot(sr(:),100.0*sum(SUC_wmc,2)/exp_times,'--b','MarkerSize',8,'linewidth',1);
plot(sr(:),100.0*sum(SUC_mc, 2)/exp_times,'-+g','MarkerSize',8,'linewidth',1);
xlim([sr(1) sr(end)]);
legend('DirtyIMC','IMC','RMC','WMC','MC','Location','SouthEast');
xlabel({'$p$'},'Interpreter','latex','fontsize',12)
ylabel({'Success rate(%)'},'fontsize',12)

if isSaving
    %print('-depsc',['figures\syndata_suc_',name,'.eps']);
    print(gcf,'-dpng',['figures\wsndata_suc_',name,'.png']);
end

figure('numbertitle','off','name','Err_vs_p');
plot(sr(:),mean(ERR_dirtyImc,2),'-.k','MarkerSize',8,'linewidth',1);
hold on;
plot(sr(:),mean(ERR_imc,2),'-m','MarkerSize',8,'linewidth',1);
plot(sr(:),mean(ERR_rmc,2),'-xr','MarkerSize',8,'linewidth',1);
plot(sr(:),mean(ERR_wmc,2),'--b','MarkerSize',8,'linewidth',1);
plot(sr(:),mean(ERR_mc, 2),'-+g','MarkerSize',8,'linewidth',1);
xlim([sr(1) sr(end)]);
legend('DirtyIMC','IMC','RMC','WMC','MC','Location','SouthEast');
xlabel({'$p$'},'Interpreter','latex','fontsize',12)
ylabel({'Recovery Error'},'fontsize',12)

if isSaving
    %print('-depsc',['figures\syndata_err_',name,'.eps']);
    print(gcf,'-dpng',['figures\wsndata_err_',name,'.png']);
end

if isSaving
    save(['savings\wsndata_',name,'.mat'],'exp_times','sr','SUC_mc','SUC_rmc','SUC_wmc','SUC_imc','SUC_dirtyImc','ERR_mc','ERR_rmc','ERR_wmc','ERR_imc','ERR_dirtyImc','iter_num');
end




figure('numbertitle','off','name','Err_vs_p');
plot(sr(:),log10(mean(ERR_dirtyImc,2)),'-.k','MarkerSize',8,'linewidth',1);
hold on;
plot(sr(:),log10(mean(ERR_imc,2)),'-m','MarkerSize',8,'linewidth',1);
plot(sr(:),log10(mean(ERR_rmc,2)),'-xr','MarkerSize',8,'linewidth',1);
plot(sr(:),log10(mean(ERR_wmc,2)),'--b','MarkerSize',8,'linewidth',1);
plot(sr(:),log10(mean(ERR_mc, 2)),'-+g','MarkerSize',8,'linewidth',1);
xlim([sr(1) sr(end)]);
legend('DirtyIMC','IMC','RMC','WMC','MC','Location','SouthEast');
xlabel({'$p$'},'Interpreter','latex','fontsize',12)
ylabel({'Recovery Error'},'fontsize',12)
