isUseGPU=true; %true; Please set isUseGPU=true, when you use GPUs.

%% load pre-trained CNNs (you can train your own CNNs instead of using ours)
netname='cub200';
model='vgg16';
load(['../pretrained_cnns/',model,'_',netname,'.mat'],'net');
mkdir(['./mat/',netname]);

%% setup matconvnet
system('wget -O ../matconvnet.zip --no-check-certificate https://github.com/vlfeat/matconvnet/archive/v1.0-beta24.zip');
system('unzip ../matconvnet.zip -d ../');
system('rm ../matconvnet.zip');


%% compile matconvnet
cd ../matconvnet-1.0-beta24/
addpath(genpath('matlab'));
vl_compilenn('enableGpu', isUseGPU); %% Please check setting options in http://www.vlfeat.org/matconvnet/install/ before running this, so that you can revise the command according to your system.
cd ../code


%% download images
system('wget -O ../CUB.tgz --no-check-certificate http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz');
system('tar -xvzf ../CUB.tgz -C ../data/');
system('rm ../CUB.tgz');


%% extract images
conf.data.catedir='../data/CUB_200_2011/';
conf.data.imgdir='../data/CUB_200_2011/images/';
conf.data.readCode='./data_input/data_input_CUB-200-2011/';
addpath(genpath(conf.data.readCode));
addpath(genpath('tool'));
objset=readAnnotation(netname,conf);
objNum=numel(objset);
tarSize=net.meta.normalization.imageSize(1:2);
images=zeros(tarSize(1),tarSize(2),3,objNum,'uint8');
for objID=1:objNum
    I=getI(objset(objID),conf,tarSize,false);
    images(:,:,:,objID)=uint8(I);
end
images_neg=getNegativeImages(tarSize);
save(['./mat/',netname,'/images.mat'],'images','images_neg');
clear images;


%% extract CNN features
addpath(genpath('./tool'));
conf=getNetParameters('vgg16',net); % this function returns structural configurations of VGG16, ResNets, and VAEGAN.
addpath(genpath(conf.convnet.codedir));
net=vl_simplenn_tidy(net);
if(isUseGPU)
    net=vl_simplenn_move(net,'gpu');
end
vl_setupnn;
load(['./mat/',netname,'/images.mat'],'images','images_neg');
[batch_f,batch_f_flip,stat_all]=getRoughCNN(net,conf,images,images_neg);
clear images;
save(['./mat/',netname,'/roughCNN.mat'],'batch_f','batch_f_flip','conf','stat_all');
