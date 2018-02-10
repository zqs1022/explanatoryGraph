function learn_explanatoryGraph(netname)
eta=1;
iterNum=20;

try
    parpool;
catch
    delete(gcp);
    parpool;
end
addpath(genpath('./tool'));
mex ./tool/main/getProb_local_mex.cpp
mex ./tool/main/getOtherProb.cpp
load(['./mat/',netname,'/roughCNN.mat'],'batch_f','batch_f_flip','conf','stat_all');
model=initializeModel(conf,batch_f{1});
imgNum=length(batch_f);
layerNum=length(conf.convnet.targetLayers);
imgFlip=ones(imgNum,1).*(-1);
totalPos_pre=[];
inferredProb=[];
for layerID=layerNum:-1:1
    tmp=roughCNN_uncompress(batch_f{1},conf);
    oriLayer=conf.convnet.targetLayers(layerID);
    [xh,~,channelNum]=size(tmp(oriLayer).x);
    [posCand,~,~]=getBasicSettings(tmp(oriLayer).x,conf,model.layer(layerID).deform,layerID);
    [posCandNum,tmp]=size(posCand);
    patternNum=tmp/channelNum;
    totalPos=zeros(2,patternNum*channelNum,imgNum);
    totalPos_alter=zeros(2,patternNum*channelNum,imgNum);
    for iter=1:iterNum
        coord=model.layer(layerID).coord;
        deform=model.layer(layerID).deform;
        valid=((layerID<layerNum)&&(iter>1));
        [totalIdx,curDelta,curMu,curLambda]=getCurDistribution(valid,model,layerID,totalPos_pre,totalPos,inferredProb,conf,patternNum,channelNum,xh);
        
        isLastIter=(iter==iterNum);
        parSet=repmat(struct('conf',conf,'layerID',layerID,'coord',coord,'deform',deform,'isLastIter',isLastIter,'num',[posCandNum,patternNum,channelNum]),[1,imgNum]);
        totalProb=zeros(patternNum*channelNum,imgNum);
        totalProb_alter=zeros(patternNum*channelNum,imgNum);
        totalNormProb=zeros(patternNum*channelNum,imgNum);
        totalLogDer=zeros(2,patternNum*channelNum,imgNum);
        totalPosVar=zeros(patternNum*channelNum,imgNum);
        inferredProb=zeros(patternNum*channelNum,imgNum);
        parfor imgID=1:imgNum %%%%%%%%%%%%%%%%%%%%%%%
            IsAlter=false;
            [prob,prior,logDer,pos,imgFlip(imgID)]=doFlipConfigureation(IsAlter,batch_f{imgID},batch_f_flip{imgID},parSet(imgID).conf,imgFlip(imgID),parSet(imgID).layerID,parSet(imgID).coord,parSet(imgID).deform,curMu(:,:,imgID),curDelta(:,imgID),curLambda(:,imgID));
            IsAlter=true;
            [prob_alter,prior_alter,~,pos_alter,~]=doFlipConfigureation(IsAlter,batch_f{imgID},batch_f_flip{imgID},parSet(imgID).conf,1-imgFlip(imgID),parSet(imgID).layerID,parSet(imgID).coord,parSet(imgID).deform,curMu(:,:,imgID),curDelta(:,imgID),curLambda(:,imgID));
            
            if(mean(isnan(prob(:)))>0)||(mean(isnan(prior(:)))>0)||(mean(isnan(logDer(:)))>0)||(mean(isnan(pos(:)))>0)
                disp([(mean(isnan(prob(:)))),(mean(isnan(prior(:)))),(mean(isnan(logDer(:)))),(mean(isnan(pos(:))))]);
                disp([imgID,layerID,iter]);
                error('Here');
            end
            
            parPosCandNum=parSet(imgID).num(1);
            parPatternNum=parSet(imgID).num(2);
            parChannelNum=parSet(imgID).num(3);
            sumProb=zeros(size(prob));
            for ch=1:parChannelNum
                list=(ch-1)*parPatternNum+(1:parPatternNum);
                otherProb=getOtherProb(prob(:,list),pos(:,:,list),parPosCandNum^2,parPatternNum,1);
                otherProb=reshape(otherProb,[parPatternNum,parPosCandNum^2,parPatternNum]);
                sumProb(:,list)=reshape(sum(otherProb,1),[parPosCandNum^2,parPatternNum]);
            end
            tau=parSet(imgID).conf.learn.validTau.*mean(sum(prob,1),2);
            sumProb=sumProb+tau;
            normProb=prob./max(sumProb,0.0000001);
            finalProb=normProb.*prior;
            totalNormProb(:,imgID)=sum(finalProb,1)';
            delta=sum(repmat(reshape(finalProb,[1,size(finalProb)]),[2,1,1]).*logDer,2);
            totalProb(:,imgID)=sum(prob.*prior,1)';
            totalProb_alter(:,imgID)=sum(prob_alter.*prior_alter,1)';
            totalLogDer(:,:,imgID)=reshape(delta,[2,parPatternNum*parChannelNum]);
            tmp=1:parPosCandNum;tmp=(tmp-mean(tmp)).^2;
            tmp=repmat(tmp,[parPosCandNum,1]);tmp=tmp+tmp';
            tmp=repmat(reshape(tmp,[parPosCandNum^2,1]),[1,parPatternNum*parChannelNum]);
            totalPosVar(:,imgID)=sum(prob.*prior.*tmp,1)';
            [~,idx]=max(prob.*prior,[],1);
            pos=reshape(pos,[2,(parPosCandNum^2)*parPatternNum*parChannelNum]);
            pos_alter=reshape(pos_alter,[2,(parPosCandNum^2)*parPatternNum*parChannelNum]);
            idx=idx+(0:parPatternNum*parChannelNum-1).*(parPosCandNum^2);
            totalPos(:,:,imgID)=pos(:,idx);
            totalPos_alter(:,:,imgID)=pos_alter(:,idx);
            
            inferredProb(:,imgID)=prob(idx).*prior(idx); %%%%%%%%%%%%%%%% adding prior or not?
            %inferredProb(:,imgID)=prob(idx); %%%%%%%%%%%%%%%% adding prior or not?
        end
        minDelta=min(curDelta(:))/(iter^0.25);
        model.layer(layerID)=updateModel(totalProb,totalNormProb,totalLogDer,totalPosVar,totalPos,totalIdx,coord,posCand,eta,xh,conf,layerID,minDelta,imgFlip,totalProb_alter,totalPos_alter);
        fprintf('layerID = %d   iter = %d\n',layerID,iter)
    end
    totalPos_pre=totalPos;
    save(['./mat/',netname,'/model.mat'],'model');
end
end


function layer=updateModel(totalProb,totalNormProb,totalLogDer,totalPosVar,totalPos,totalIdx,coord,posCand,eta,xh,conf,layerID,minDelta,imgFlip,totalProb_alter,totalPos_alter)
layer.coord=coord+reshape(sum(totalLogDer,3)./repmat(max(sum(totalNormProb,2),0.00000001)',[2,1]),size(coord)).*(eta*(minDelta^2));
unit=(posCand(2,:)-posCand(1,:))';
layer.deform=sqrt(sum(totalPosVar,2)./sum(totalProb,2)).*unit;
maxDeform=conf.learn.deform_.max_delta(layerID)*xh;
minDeform=conf.learn.deform_.min_delta(layerID)*xh;
layer.deform=min(max(layer.deform,minDeform),maxDeform);
norma=mean(mean(totalProb,2),1);
totalProb=totalProb./norma;
layer.prob_record=totalProb;
layer.pos_record=totalPos;
layer.idx_record=totalIdx;
layer.flip_record=imgFlip;
totalProb_alter=totalProb_alter./norma;
layer.alter.prob_record=totalProb_alter;
layer.alter.pos_record=totalPos_alter;
end
