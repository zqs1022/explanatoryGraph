function computeStability_cub()
netname='cub200';
selectedPatternNum=167;
patchNumPerPattern=20;
partList=[1,6,14];

addpath(genpath('./tool'));
load(['./mat/',netname,'/roughCNN.mat'],'batch_f','conf');
load(['./mat/',netname,'/model.mat'],'model');
load(['./mat/',netname,'/images.mat'],'images');
len=norm([size(images,1),size(images,2)]);
clear images;
conf.data.readCode='./data_input/data_input_CUB-200-2011/';
stability=computeStability(netname,selectedPatternNum,patchNumPerPattern,partList,conf,model);
stability_raw=computeStability_raw(netname,selectedPatternNum,patchNumPerPattern,partList,conf,batch_f);
fprintf('location stability of our graph nodes is %f\n. location stability of raw CNN filters is %f\n',stability/len,stability_raw/len)
end
