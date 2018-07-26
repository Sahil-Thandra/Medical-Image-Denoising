files1=dir('D:\DX\Original\*.pgm');
files2=dir('D:\MMM\Original\*.pgm');
pgm1files=dir('D:\A\*.pgm');
pgm2files=dir('D:\B\*.pgm');

ssim_tot=0;
for idx =001:400
ref=strcat('D:\DX\Original\',files1(idx).name);
im1=imread(im);
act =strcat('D:\A\',pgm1files(idx).name);
im2=imread(im);
[ssimval, ssimmap] = ssim(im2,im1);
ssim_tot = ssimval + ssim_tot;
end
ssim_mean = ssim_tot/400;
fprintf('The SSIM value is %0.4f.\n',ssim_mean);

ssim_tot=0;
for idx =001:322
ref=strcat('D:\MMM\Original\',files2(idx).name);
im1=imread(im);
act =strcat('D:\B\',pgm2files(idx).name);
im2=imread(im);
[ssimval, ssimmap] = ssim(im2,im1);
ssim_tot = ssimval + ssim_tot;
end
ssim_mean = ssim_tot/322;
fprintf('The SSIM value is %0.4f.\n',ssim_mean);


