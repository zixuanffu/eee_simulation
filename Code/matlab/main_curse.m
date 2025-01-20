%%
% Logistic many moments model to show curse of dimensionality
%
% Section 3.1.3 of Kaji, Manresa, and Pouliot (2023) "An Adversarial
% Approach to Structural Estimation"
%
% Produces Figures 4 and 5.
%
% Written by Tetsuya Kaji


%%
% Setup
%

% To save the source code in the data
%
code = fileread('main_curse.m');

% For replication
%
rng default

% Sample sizes
%
n = 200;
m = 200;
Mom = [1 3 5 7 9 11]; % Numbers of moments

% Parameters
%
th0 = 0;    % mean of logistic for P0
s0 = 1;     % scale of logistic for P0

% Simulation size
%
S = 500;

% Transformation
%
T = @(th,x) th+x;

% Bootstrap size for optimal weight calculation
%
B = 200;


%%
% Analytic calculation
%

% Likelihood
%
log_p0 = @(x) loglogisticpdf(x,th0,s0);
log_p = @(th,x) loglogisticpdf(x,th,s0);

% Score and Hessian
%
ld = @(th,x) (-1+2*logisticcdf(x,th,s0))./s0;
ldd = @(th,x) -2*logisticcdf(x-th,0,s0).*logisticcdf(-(x-th),0,s0)./s0.^2;

% Miscellaneous intermediate functions
%
log_D = @(th,x) log_p0(x)-logaddexp(log_p0(x),log_p(th,x));     % log(D)
log_UmD = @(th,x) log_p(th,x)-logaddexp(log_p0(x),log_p(th,x)); % log(1-D)
p0 = @(x) exp(log_p0(x));           % p_0
p = @(th,x) exp(log_p(th,x));       % p_theta
D = @(th,x) exp(log_D(th,x));       % Oracle D

% The middle term of the sandwich formula in Theorem 3
%
% It computes the version that does not assume "n/m -> 0" (Assumption 2),
% which is given in Theorem 3 of arXiv:2007.06169v2.
%
meat = @(x) 4*exp(log_D(th0,x)+log_UmD(th0,x)).*ld(th0,x).^2;
Meat_1 = integral(@(x) meat(x).*p(th0,x),-Inf,Inf,'AbsTol',1e-20);
Meat_2 = integral(@(x) meat(x).*p0(x),-Inf,Inf,'AbsTol',1e-20);
V = Meat_1+(n/m).*Meat_2;

% The wing terms of the sandwich formula of Theorem 3
%
bun = @(x) 2*(D(th0,x).*ld(th0,x).^2+(ldd(th0,x)+ld(th0,x).^2).*log_UmD(th0,x));
I_tilde = integral(@(x) bun(x).*p(th0,x),-Inf,Inf,'AbsTol',1e-20);

% Asymptotic variance of Adv
%
V_Adv = I_tilde\V/I_tilde;

% Asymptotic standard deviation of MLE
%
V_MLE = -1/integral(@(x) ldd(th0,x).*p0(x),-Inf,Inf);


%%
% Orthogonality
%

K = 100;
th_grid = s0*linspace(-1.4,1.4,K)';
h_grid = th_grid*sqrt(n);
LL_grid = zeros(K,1);
oD_grid = zeros(K,1);
SMM_grid = zeros(K,numel(Mom));
owSMM_grid = zeros(K,numel(Mom));
cD_grid = zeros(K,numel(Mom));
SMM0_grid = zeros(1,numel(Mom));
owSMM0_grid = zeros(1,numel(Mom));
cD0_grid = zeros(1,numel(Mom));

% Real data
%
X = logisticrnd(th0,s0,n,1);

% Latent data
%
Z = logisticrnd(0,s0,m,1);

% Label for logistic regression
%
L = [ones(n,1);2*ones(m,1)]; % labels for logistic regression; 1 is the base category

tic
for k = 1 : K

    th = th_grid(k);

    % Log likelihood ratio
    %
    LL_grid(k) = -mean(log_p(th,X));

    % Oracle Adv objective function
    %
    oD_grid(k) = mean(log_D(th,X)) + mean(log_UmD(th,T(th,Z)));

end

% For level normalization
%
LL0_grid = -mean(log_p(th0,X));
oD0_grid = mean(log_D(th0,X)) + mean(log_UmD(th,T(th0,Z)));

for i = 1 : numel(Mom)

    % Real moment for SMM
    %
    moment_r = mean(X.^(1:Mom(i)),1);

    % Feasible optimal weight (simple estimation)
    %
    opW = cov(X.^(1:Mom(i)));

    for k = 1 : K

        th = th_grid(k);

        % Unweighted SMM objective function
        %
        SMM_grid(k,i) = norm(moment_r-mean(T(th,Z).^(1:Mom(i)),1))^2;

        % Optimally-weighted SMM objective function
        %
        owSMM_grid(k,i) = innerp(moment_r-mean(T(th,Z).^(1:Mom(i)),1),opW);

        % AdvL objective function
        %
        cD_grid(k,i) = loss([X;T(th,Z)].^(0:Mom(i))*mnrfit([X;T(th,Z)].^(1:Mom(i)),L),n);

    end

    % For level normalization
    %
    SMM0_grid(i) = norm(moment_r-mean(T(th0,Z).^(1:Mom(i)),1))^2;
    owSMM0_grid(i) = innerp(moment_r-mean(T(th0,Z).^(1:Mom(i)),1),opW);
    cD0_grid(i) = loss([X;T(th0,Z)].^(0:Mom(i))*mnrfit([X;T(th0,Z)].^(1:Mom(i)),L),n);
end

% Scale adjustment of SMM with owSMM
%
SMM0_grid = SMM0_grid./std(SMM_grid,1);
SMM_grid = SMM_grid./std(SMM_grid,1);
owSMM0_grid = owSMM0_grid./std(owSMM_grid,1);
owSMM_grid = owSMM_grid./std(owSMM_grid,1);
toc


%%
% Figures
%

% Figure 4a
%
figure(4)
subplot(1,3,1)
i = 2;
plot(th_grid, [cD_grid(:,i) oD_grid])
ylim([-1.39 -1.31])
yyaxis right
plot(th_grid, LL_grid/2)
ylim([0.972 1.052])
xlim([-0.7 0.7])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('Loss with %d moments', Mom(i)))
legend(...
    '${\bf M}_\theta(\hat{D}_\theta)$',...
    '${\bf M}_\theta(D_\theta)$',...
    '${\bf L}_\theta$',...
    'Interpreter','latex')
legend('boxoff')

% Figure 4b
%
figure(4)
subplot(1,3,2)
i = 4;
plot(th_grid, [cD_grid(:,i) oD_grid])
ylim([-1.39 -1.31])
yyaxis right
plot(th_grid, LL_grid/2)
ylim([0.972 1.052])
xlim([-0.7 0.7])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('Loss with %d moments', Mom(i)))
legend(...
    '${\bf M}_\theta(\hat{D}_\theta)$',...
    '${\bf M}_\theta(D_\theta)$',...
    '${\bf L}_\theta$',...
    'Interpreter','latex')
legend('boxoff')

% Figure 4c
%
figure(4)
subplot(1,3,3)
i = 6;
plot(th_grid, [cD_grid(:,i) oD_grid])
ylim([-1.39 -1.19])
yyaxis right
plot(th_grid, LL_grid/2)
ylim([0.972 1.172])
xlim([-1.4 1.4])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('Loss with %d moments', Mom(i)))
legend(...
    '${\bf M}_\theta(\hat{D}_\theta)$',...
    '${\bf M}_\theta(D_\theta)$',...
    '${\bf L}_\theta$',...
    'Interpreter','latex')
legend('boxoff')


% Export csv for Figure 4
%
writetable(array2table([th_grid LL_grid/2 SMM_grid owSMM_grid cD_grid oD_grid],'VariableNames',...
    {'theta' 'LL/2' 'SMM(1)' 'SMM(3)' 'SMM(5)' 'SMM(7)' 'SMM(9)' 'SMM(11)'...
    'owSMM(1)' 'owSMM(3)' 'owSMM(5)' 'owSMM(7)' 'owSMM(9)' 'owSMM(11)'...
    'cD(1)' 'cD(3)' 'cD(5)' 'cD(7)' 'cD(9)' 'cD(11)' 'oD'}),...
    'curse_orthogonality.csv')


%%
% Estimation
%

% Memory allocation
%
th_MLE = zeros(S,1);
th_SMM = zeros(S,numel(Mom));
th_owSMM = zeros(S,numel(Mom));
th_AdvL = zeros(S,numel(Mom));

% Initialize progress tracker
%
fID = fopen('main_curse.txt','w');
fprintf(fID,'main_curse''s routine started at %s\n',datetime('now'));
fprintf(fID,[...
    '---------+----------+----------------+--------+-------------\n' ...
    '   Step  |  Passed  |       At       | To go  |    Until    \n' ...
    '---------+----------+----------------+--------+-------------\n']);
fclose(fID);

tic
for s = 1 : S

    % Initial theta to try
    %
    th00 = (-1)^binornd(1,0.5);

    % Real data
    %
    X = logisticrnd(th0, s0, n, 1);

    % Latent data
    %
    Z = logisticrnd(0, s0, m, 1);

    % MLE
    %
    th_MLE(s) = fminsearch(@(th) -sum(loglogisticpdf(X,th,s0)),th00);

    for i = 1 : numel(Mom)

        % Real moment for SMM
        %
        moment_r = mean(X.^(1:Mom(i)),1);

        % Feasible optimal weight (simple estimation)
        %
        opW = cov(X.^(1:Mom(i)));

        % SMM
        %
        th_SMM(s,i) = fminsearch(@(th) norm(moment_r-mean((th+Z).^(1:Mom(i)),1)),th00);

        % optimally-weighted SMM
        %
        th_owSMM(s,i) = fminsearch(@(th) innerp(moment_r-mean((th+Z).^(1:Mom(i)),1),opW),th00);

        % Adversarial estimator with a logistic discriminator
        %
        th_AdvL(s,i) = fminsearch(@(th) loss([X;th+Z].^(0:Mom(i))*mnrfit([X;th+Z].^(1:Mom(i)),L),n),th00);

    end

    % Update progress tracker
    %
    if mod(s,10) == 0
        sec_psst = toc;  % seconds since tic
        h_psst = floor(sec_psst/60/60);
        m_psst = floor(mod(sec_psst,60*60)/60);
        s_psst = floor(mod(sec_psst,60));
        sec_togo = (S-s)/s*sec_psst;
        h_togo = floor(sec_togo/60/60);
        m_togo = floor(mod(sec_togo,60*60)/60);
        s_togo = floor(mod(sec_togo,60));
        tnow = datetime('now');
        tdone = tnow+seconds(sec_togo);
        fID = fopen('main_curse.txt','a+');
        fprintf(fID,' %3d/%3d |%3d:%02d:%02d | %s | %3d:%02d | %s\n',s,S,h_psst,m_psst,s_psst,datetime(tnow,'Format','MM/dd HH:mm:ss'),h_togo,m_togo,datetime(tdone,'Format','MM/dd HH:mm'));
        fclose(fID);
    end
end

% Finalize progress tracker
%
fID = fopen('main_curse.txt','a+');
fprintf(fID,'---------+----------+----------------+--------+-------------\n');
fclose(fID);

toc


%%
% Figures
%

% Figure 5a
%
figure(5)
subplot(2,3,1)
i = 2;
bins = linspace(-0.7,0.7,24);
histogram(th_owSMM(:,i),bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 5])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('SMM with %d moments', Mom(i)))
legend(sprintf('SMM(%d) (%.2f)', Mom(i), sqrt(n)*std(th_owSMM(:,i))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

% Figure 5b
%
figure(5)
subplot(2,3,2)
i = 4;
bins = linspace(-2.8,2.8,24);
histogram(th_owSMM(:,i),bins,'Normalization','pdf')
hold on
bins = linspace(-0.7,0.7,24);
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 5])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('SMM with %d moments', Mom(i)))
legend(sprintf('SMM(%d) (%.1f)', Mom(i), sqrt(n)*std(th_owSMM(:,i))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

% Figure 5c
%
figure(5)
subplot(2,3,3)
i = 6;
bins = linspace(-3.5,3.5,24);
histogram(th_owSMM(:,i),bins,'Normalization','pdf')
hold on
bins = linspace(-0.7,0.7,24);
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 5])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('SMM with %d moments', Mom(i)))
legend(sprintf('SMM(%d) (%.1f)', Mom(i), sqrt(n)*std(th_owSMM(:,i))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

% Figure 5d
%
figure(5)
subplot(2,3,4)
i = 2;
bins = linspace(-0.7,0.7,24);
histogram(th_AdvL(:,i),bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 5])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('Adv with %d moments', Mom(i)))
legend(sprintf('Adv(%d) (%.2f)', Mom(i), sqrt(n)*std(th_AdvL(:,i))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

% Figure 5e
%
figure(5)
subplot(2,3,5)
i = 4;
bins = linspace(-0.7,0.7,24);
histogram(th_AdvL(:,i),bins,'Normalization','pdf')
hold on
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 5])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('Adv with %d moments', Mom(i)))
legend(sprintf('Adv(%d) (%.2f)', Mom(i), sqrt(n)*std(th_AdvL(:,i))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')

% Figure 5f
%
figure(5)
subplot(2,3,6)
i = 6;
bins = linspace(-1.4,1.4,24);
histogram(th_AdvL(:,i),bins,'Normalization','pdf')
hold on
bins = linspace(-0.7,0.7,24);
histogram(th_MLE,bins,'Normalization','pdf')
hold off
ylim([0 5])
xlabel('$\theta$','Interpreter','latex')
title(sprintf('Adv with %d moments', Mom(i)))
legend(sprintf('Adv(%d) (%.2f)', Mom(i), sqrt(n)*std(th_AdvL(:,i))), ...
       sprintf('MLE (%.2f)', sqrt(n)*std(th_MLE)))
legend('boxoff')


% Export csv for Figure 5
%
writetable(array2table([th_MLE th_SMM th_owSMM th_AdvL],'VariableNames',...
    {'MLE' 'SMM(1)' 'SMM(3)' 'SMM(5)' 'SMM(7)' 'SMM(9)' 'SMM(11)'...
    'owSMM(1)' 'owSMM(3)' 'owSMM(5)' 'owSMM(7)' 'owSMM(9)' 'owSMM(11)'...
    'AdvL(1)' 'AdvL(3)' 'AdvL(5)' 'AdvL(7)' 'AdvL(9)' 'AdvL(11)'}),...
    'curse_estimator.csv')


%%
% Save Matlab workspace and figures
%

seed = rng;
save(sprintf('main_curse_%s.mat',datetime('now','format','yyyyMMddHHmmss')))

saveas(figure(4),'Figure4.png')
saveas(figure(5),'Figure5.png')


%%
% Functions
%

% For logistic discriminator, compute loss from the index xb.
%
function v = loss(xb,n)
    v = -mean(softplus(-xb(1:n)))-mean(softplus(xb(n+1:end)));
end

% Inner product of a row vector x.
%
function ip = innerp(x,A)
    ip = x/A*x';
end
