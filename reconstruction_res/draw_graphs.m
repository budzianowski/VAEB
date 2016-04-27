continuous = [6.81307601929 6.84845234758; 3.61092782021 3.97013417255; 3.27499914169 3.73249050824];
discrete = [37.1403999329 33.0772819519; 19.0405158997 12.7276439667; 14.7736902237 8.26831054688];

bar(discrete);
legend('AE', 'VAE');
set(gca,'XTickLabel',{'2','10', '20'})
title('MSE for various VAE and AE')
xlabel('Latent space size')
ylabel('MSE')
print('../res/mnist_mse.pdf','-dpdf')
