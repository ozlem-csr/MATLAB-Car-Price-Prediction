clc;
clear;
close all;


veri=readtable("Galeri_Verileri.csv");
disp(veri)


size(veri)
veri.Properties.VariableNames
summary(veri)


veri.aracYasi=2026-veri.modelYili;
disp(veri)


sum(ismissing(veri))


veri.vitesNo=double(strcmp(veri.vitesTuru,"Otomatik"));
veri.yakitNo=double(strcmp(veri.yakitTuru,"Dizel"));
veri.durumNo=double(strcmp(veri.galeriDurumu,"Kiralık"));
disp(veri)


veri.modelYiliNorm=normalize(veri.modelYili,"range");
veri.kilometreNorm=normalize(veri.kilometre,"range");
veri.aracYasiNorm=normalize(veri.aracYasi,"range");
disp(veri)


veri.aracGrubu=strings(height(veri),1);
veri.aracGrubu(veri.aracUcreti <= 1000000) ="Ekonomik";
veri.aracGrubu(veri.aracUcreti > 1000000 & veri.aracUcreti <= 2000000) ="Orta";
veri.aracGrubu(veri.aracUcreti > 2000000) ="Luks";
disp(veri)


girisVerisi = veri(:,["kilometreNorm","aracYasiNorm","vitesNo","yakitNo","durumNo"]);
cikisVerisi = veri.aracUcreti;

cv = cvpartition(height(veri),"HoldOut",0.2);

girisEgitim = girisVerisi(training(cv),:);
girisTest = girisVerisi(test(cv),:);

cikisEgitim = cikisVerisi(training(cv));
cikisTest = cikisVerisi(test(cv));




%Lineer Regresyon

egitimVerisi = girisEgitim;
egitimVerisi.aracUcreti = cikisEgitim;

model = fitlm(egitimVerisi,'aracUcreti ~ kilometreNorm + aracYasiNorm + vitesNo + yakitNo + durumNo');

tahmin = predict(model,girisTest);

hata=cikisTest-tahmin;
MAE=mean(abs(hata))
MSE=mean(hata.^2)
RMSE=sqrt(MSE)

figure
plot(cikisTest,'bo')
hold on
plot(tahmin,'r*')
grid on
legend('Gercek Fiyat','Tahmin')
title('Lineer Regresyon Sonuclari')
xlabel('Test Verileri')
ylabel('Arac Ucreti')





%Siniflandirma algoritmalari icin kullanilir: decision tree, knn, svm

girisSinif = veri(:,["kilometreNorm","aracYasiNorm","vitesNo","yakitNo","durumNo"]);
cikisSinif = veri.aracGrubu;

cv = cvpartition(height(veri),"HoldOut",0.2);

girisEgitim = girisSinif(training(cv),:);
girisTest = girisSinif(test(cv),:);

cikisEgitim = cikisSinif(training(cv));
cikisTest = cikisSinif(test(cv));





%Decision Tree

model = fitctree(girisEgitim,cikisEgitim);

tahmin = predict(model,girisTest);

%disp(tahmin)

tahmin = string(tahmin);

figure
confusionchart(cikisTest,tahmin)

dogru = sum(cikisTest==tahmin);
accuracy = (dogru/length(cikisTest))*100

C = confusionmat(cikisTest,tahmin);

precision = mean(diag(C)./sum(C,1)');

recall = mean(diag(C)./sum(C,2));

f1 = 2*(precision*recall)/(precision+recall)

precision

recall








% KNN

model = fitcknn(girisEgitim,cikisEgitim);

tahmin = predict(model,girisTest);

%disp(tahmin)

tahmin = string(tahmin);

figure
confusionchart(cikisTest,tahmin)

dogru = sum(cikisTest==tahmin);
accuracy = (dogru/length(cikisTest))*100

C = confusionmat(cikisTest,tahmin);

precision = mean(diag(C)./sum(C,1)');

recall = mean(diag(C)./sum(C,2));

f1 = 2*(precision*recall)/(precision+recall)

precision

recall






% SVM

model = fitcecoc(girisEgitim,cikisEgitim);

tahmin = predict(model,girisTest);

%disp(tahmin)

tahmin = string(tahmin);

figure
confusionchart(cikisTest,tahmin)

dogru = sum(cikisTest==tahmin);
accuracy = (dogru/length(cikisTest))*100

C = confusionmat(cikisTest,tahmin);

precision = mean(diag(C)./sum(C,1)');

recall = mean(diag(C)./sum(C,2));

f1 = 2*(precision*recall)/(precision+recall)

precision

recall








%GMM

X=[veri.kilometreNorm veri.aracYasiNorm];

gm=fitgmdist(X,3);
yeni=random(gm,1);

disp("Yeni uretilen arac bilgisi:")
disp(yeni)

figure
scatter(X(:,1),X(:,2),100,'filled')
hold on

scatter(yeni(:,1),yeni(:,2),100,'r','filled')

legend('Orijinal Veri','Yeni Kayit')

xlabel('Kilometre')
ylabel('Arac Yasi')

title('GMM Sonucu')

grid on



