function stability=computeStability(Name_batch,selectedPatternNum,patchNumPerPattern,partList,theConf,model)
imgSize=theConf.convnet.imgSize;
imgNum=size(model.layer(1).pos_record,3);
layerNum=length(model.layer);
avgDistSqrtVar=zeros(layerNum,1);
for layerID=1:layerNum
    for imgID=1:imgNum
        flip=model.layer(layerID).flip_record(imgID);
        [~,prob]=getAlterProbPos(model,flip,layerID,imgID,theConf,imgSize);
        if(imgID==1)
            patNum=numel(prob);
            theP=zeros(patNum,imgNum);
        end
        theP(:,imgID)=prob;
    end
    tmp=sort(theP,2,'descend');
    [~,patList]=sort(mean(tmp(:,1:patchNumPerPattern),2),'descend');
    patList=patList(1:selectedPatternNum);
    pos=zeros(2,patNum,imgNum);
    for imgID=1:imgNum
        flip=model.layer(layerID).flip_record(imgID);
        [tmp,~]=getAlterProbPos(model,flip,layerID,imgID,theConf,imgSize);
        pos(:,:,imgID)=tmp;
    end
    distSqrtVar=getDistSqrtVar(pos(:,patList,:),theP(patList,:),patchNumPerPattern,partList,Name_batch,theConf);
    avgDistSqrtVar(layerID)=mean(distSqrtVar);
end
stability=mean(avgDistSqrtVar);
end


function [pos,prob]=getAlterProbPos(model,flip,layerID,imgID,theConf,imgSize)
patNum=size(model.layer(layerID).prob_record,1);
IsFlip=repmat(flip,[patNum,1]);
pos=x2p_(model.layer(layerID).pos_record(:,:,imgID),layerID,theConf);
prob=model.layer(layerID).prob_record(:,imgID);
pos_alter=x2p_(model.layer(layerID).alter.pos_record(:,:,imgID),layerID,theConf);
prob_alter=model.layer(layerID).alter.prob_record(:,imgID);
list=find(prob-prob_alter<0);
pos(:,list)=pos_alter(:,list);
prob(list)=prob_alter(list);
IsFlip(list)=1-IsFlip(list);
list=find(IsFlip==true);
pos(2,list)=repmat(imgSize(2),[1,numel(list)])+1-pos(2,list);
end
