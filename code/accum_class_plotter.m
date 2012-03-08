% Do accumulation classifier plots for paper
clear;
load('csum_figure_data.mat');
figure();
% Subject k6b...
subplot(1,3,1);
hold on;
plot(mean(C1_csums_k6b),'b-');
plot(mean(C1_csums_k6b)+std(C1_csums_k6b),'b:');
plot(mean(C1_csums_k6b)-std(C1_csums_k6b),'b:');
plot(mean(C2_csums_k6b),'r-');
plot(mean(C2_csums_k6b)+std(C2_csums_k6b),'r:');
plot(mean(C2_csums_k6b)-std(C2_csums_k6b),'r:');
% Subject k3b...
subplot(1,3,2);
hold on;
plot(mean(C1_csums_k3b),'b-');
plot(mean(C1_csums_k3b)+std(C1_csums_k3b),'b:');
plot(mean(C1_csums_k3b)-std(C1_csums_k3b),'b:');
plot(mean(C2_csums_k3b),'r-');
plot(mean(C2_csums_k3b)+std(C2_csums_k3b),'r:');
plot(mean(C2_csums_k3b)-std(C2_csums_k3b),'r:');
% Subject l1b...
subplot(1,3,3);
hold on;
plot(mean(C1_csums_l1b),'b-');
plot(mean(C1_csums_l1b)+std(C1_csums_l1b),'b:');
plot(mean(C1_csums_l1b)-std(C1_csums_l1b),'b:');
plot(mean(C2_csums_l1b),'r-');
plot(mean(C2_csums_l1b)+std(C2_csums_l1b),'r:');
plot(mean(C2_csums_l1b)-std(C2_csums_l1b),'r:');
