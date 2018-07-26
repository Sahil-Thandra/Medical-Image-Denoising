
files=dir('D:\Noisy\*.jpg');
jpg1files=dir('D:\A\*.jpg');
jpg2files=dir('D:\B\*.jpg');
n=numel(files);
for idx =001:400
im=strcat('D:\Noisy\',files(idx).name);
im1=imread(im);

I = im1;I=double(I);
I = I - min(I(:));
I = I / max(I(:));

%// Add noise to image
v = 0.1*var(I(:));
I_noisy = imnoise(I, 'gaussian', 0, v);
I_noisy=255.*I_noisy;

I_noisy = I_noisy - min(I_noisy(:));
I_noisy = I_noisy / max(I_noisy(:));
 
imx=strcat('D:\A\',jpg1files(idx).name);
imy=strcat('D:\B\',jpg2files(idx).name);


imwrite(I,imx);
imwrite(I_noisy,imy);
end

