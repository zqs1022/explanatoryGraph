function stability=computeStability_raw(Name_batch,selectedPatternNum,patchNumPerPattern,partList,theConf,batch_f)
imgNum=length(batch_f);
layerNum=length(theConf.convnet.targetLayers);
avgDistSqrtVar=zeros(layerNum,1);
for layerID=1:layerNum
    oriLayer=theConf.convnet.targetLayers(layerID);
    patNum=double(batch_f{1}(oriLayer).size(3));
    pos=zeros(2,patNum,imgNum);
    score=zeros(patNum,imgNum);
    for imgID=1:imgNum
        res=roughCNN_uncompress(batch_f{imgID},theConf);
        x=double(res(oriLayer).x);
        xh=size(x,1);
        [v,idx]=max(x,[],1);
        [v,tmp]=max(v,[],2);
        tmp=reshape(tmp,[1,patNum]);
        idx_h=idx(tmp+(0:patNum-1).*xh);
        idx_w=reshape(tmp,[1,patNum]);
        theScore=reshape(v,[patNum,1]);
        thePos=x2p_([idx_h;idx_w],layerID,theConf);
        pos(:,:,imgID)=thePos;
        score(:,imgID)=theScore;
    end
    tmp=sort(score,2,'descend');
    [~,patList]=sort(mean(tmp(:,1:patchNumPerPattern),2),'descend');
    patList=patList(1:min(selectedPatternNum,patNum));
    distSqrtVar=getDistSqrtVar(pos(:,patList,:),score(patList,:),patchNumPerPattern,partList,Name_batch,theConf);
    avgDistSqrtVar(layerID)=mean(distSqrtVar);
end
stability=mean(avgDistSqrtVar);
end
