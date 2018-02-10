function showPatch(netname)
layerList=[2,3];
layerScale=[60,80];
scale_show=150;
topPatchNum_forRank=50;
topPatchNum_show=5;
topPatternNum=20;


addpath(genpath('./tool'));
mkdir('./output/patch/');
load(['./mat/',netname,'/roughCNN.mat'],'conf');
load(['./mat/',netname,'/images.mat'],'images');
load(['./mat/',netname,'/model.mat'],'model');
lc=0;
for layerID=layerList
    lc=lc+1;
    scale=layerScale(lc);
    prob=model.layer(layerID).prob_record;
    tmp=sort(prob,2,'descend');
    tmp=sum(tmp(:,1:topPatchNum_forRank),2);
    [~,idx_m]=sort(tmp,'descend');
    for c=1:topPatternNum
        p=idx_m(c);
        [~,idx]=sort(prob(p,:),'descend');
        for i=1:topPatchNum_show
            imgID=idx(i);
            I_obj=images(:,:,:,imgID);
            pos=model.layer(layerID).pos_record(:,p,imgID);
            pHW=round(x2p_(pos,layerID,conf));
            delta=floor(min([min(pHW)-1,min(conf.convnet.imgSize(1:2)'-pHW),(scale-1)/2]));
            I_patch=I_obj(pHW(1)-delta:pHW(1)+delta,pHW(2)-delta:pHW(2)+delta,:);
            I_patch=imresize(I_patch,[scale_show,scale_show],'bilinear');
            imwrite(I_patch,sprintf('./output/patch/%s_layer%02d_pattern%02d_%02d.png',netname,layerID,c,i));
        end
    end
end
end

