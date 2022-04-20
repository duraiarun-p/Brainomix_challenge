clc;clear;close all;
%%
inputImageFileName='/home/arun/Documents/PyWSPrecision/Brainomix/Images/vol_02.nii.gz';
V = niftiread(inputImageFileName);
% V=rescale(V,0,1);
Vhline=smooth(double(V(:,256,1)));
Vhline1=smooth(double(V(256,:,1)));

[N,edges] = histcounts(V);
edges(1)=[];
[c,i]=max(N);
Vback=edges(i);
% figure(90);plot(edges,N)
% 
if Vback>-2000
    lung_analysis(V);
else
    disp('Second Type of function');
    lung_analysis_2(V);
end
function lung_analysis_2(V)    
siz=size(V);
% V=rescale(V,0,1);

Vbodymask=false(siz);

Vbody=int16(zeros(siz));

Vlungsmask=false(siz);

Vlung=int16(zeros(siz));

Vvesselmask=false(siz);


Vmax=max(V(:));
Vmin=min(V(:));
Vran=range(V(:));

% BodyThresh=-500;

LungThresh=Vmin+2724;
BodyThresh=Vmin+2524;
VesselThresh=Vmin+2324;
% VesselThresh=-700;


for i=1:siz(3)
sl=V(:,:,i);
siz=size(V);
Vhline=smooth(double(sl(:,256)));
Vhline1=smooth(double(sl(256,:)));

% Vstart=0.5*(Vhline(1)+Vhline(2));
Vstart=min([Vhline(1),Vhline1(1)]);
V(V==Vmin)=Vstart;

sl=V(:,:,i);
%Body segmentation
sl1=sl<BodyThresh;

sl1=imclose(sl1,8);
%sl1=imfill(sl1,8,'holes');

wholemask=sl1; % For reuse in other operations

sl1=imcomplement(sl1);
sl1=imfill(sl1,8,'holes');

Vbodymask(:,:,i)=sl1;

sl_n_1=int16(immultiply(sl1,sl));
Vbody(:,:,i)=sl_n_1;

%Lung segmentation
sl2=sl_n_1<LungThresh;
% sl2=imclose(sl2,8);
% sl2=imfill(sl2,8,'holes');
SE = strel("disk",3,4);
sl2=imerode(sl2,SE);
% sl2=immultiply(imcomplement(sl1),imcomplement(sl2));
sl2=imdilate(sl2,SE);
sl2=imfill(sl2,8,'holes');

Vlungsmask(:,:,i)=sl2;
sl_n_2=int16(immultiply(sl2,sl_n_1));
Vlung(:,:,i)=sl_n_2; % Volume can be calculated here


% Vessel segmentation
sl31=sl_n_2==0;
sl31=imcomplement(sl31);
sl32=sl_n_2>VesselThresh;
sl3=immultiply(sl31,sl32);
SE = strel("diamond",2);
sl3=imerode(sl3,SE);
sl3=imfill(sl3,8,'holes');
Vvesselmask(:,:,i)=sl3;
% Vvesselmask(:,:,i)=immultiply(sl31,sl32);

end

figure(1);sliceViewer(V);title('Vol');
figure(2);sliceViewer(Vbodymask);title('bodymask');
figure(3);sliceViewer(Vbody);title('masked vol');
figure(4);sliceViewer(Vlungsmask);title('lungmask');
figure(5);sliceViewer(Vlung);title('lung');
figure(6);sliceViewer(Vvesselmask);title('vesselmask'); 
end

function lung_analysis(V)
Vhline=smooth(double(V(:,256,1)));
Vhline1=smooth(double(V(256,:,1)));

Vmax=max(V(:));
Vmin=min(V(:));
Vran=range(V(:));

LungThresh=Vmin+865;
BodyThresh=Vmin+265;
VesselThresh=Vmin+924;

% LungThresh=-100;
% BodyThresh=-700;
% VesselThresh=-600;


% BodyThresh=Vmin+(0.094369299*Vran);
% VesselThresh=Vmin+(0.1258257*Vran);
% LungThresh=Vmin+(0.283107896*Vran);

siz=size(V);
% V=rescale(V,0,1);

Vbodymask=false(siz);

Vbody=int16(zeros(siz));

Vlungsmask=false(siz);

Vlung=int16(zeros(siz));

Vvesselmask=false(siz);

%% Body segmentation
% Thresh1=min(V(:))+(0.30*range(V(:)));
% Thresh1=min(V(:))+(0.28310789556*range(V(:)));
% Thresh1=min(V(:))+(0.28310789556*range(V(:)));

for i=1:siz(3)
sl=V(:,:,i);
% Thresh1=min(sl(:))+(0.30*range(sl(:)));
% Thresh1=min(sl(:))+(0.28310789556*range(sl(:)));
%Body segmentation
sl1=sl<BodyThresh;

sl1=imclose(sl1,8);
sl1=imfill(sl1,8,'holes');

wholemask=sl1; % For reuse in other operations

sl1=imcomplement(sl1);
sl1=imfill(sl1,8,'holes');

Vbodymask(:,:,i)=sl1;
% Vbody(:,:,i)=immultiply(sl1,sl);
% sl_n_1=int16(sl1*double(sl));
sl_n_1=int16(immultiply(sl1,sl));
Vbody(:,:,i)=sl_n_1;

%Lung segmentation
sl2=sl_n_1<LungThresh;
sl2=imclose(sl2,8);
sl2=imfill(sl2,8,'holes');
SE = strel("disk",7,8);
sl2=imerode(sl2,SE);
% sl2=immultiply(imcomplement(sl1),imcomplement(sl2));


Vlungsmask(:,:,i)=sl2;
sl_n_2=int16(immultiply(sl2,sl_n_1));
Vlung(:,:,i)=sl_n_2; % Volume can be calculated here

% Vessel segmentation
sl31=sl_n_2~=0;
sl32=sl_n_2>VesselThresh;
Vvesselmask(:,:,i)=immultiply(sl31,sl32);


end


figure(1);sliceViewer(V);title('Vol');
figure(2);sliceViewer(Vbodymask);title('bodymask');
figure(3);sliceViewer(Vbody);title('masked vol');
figure(4);sliceViewer(Vlungsmask);title('lungmask');
figure(5);sliceViewer(Vlung);title('lung');
figure(6);sliceViewer(Vvesselmask);title('vesselmask');  




end
%%
