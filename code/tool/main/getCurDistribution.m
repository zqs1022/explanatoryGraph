function [totalIdx,curDelta,curMu,curLambda]=getCurDistribution(valid,model,layerID,totalPos_pre,totalPos,inferredProb,theConf,patternNum,channelNum,xh)
minVarRate=0.5;
edgeCutRate=0.1;

imgNum=size(totalPos,3);
if(valid)
    tmp_pre=totalPos_pre-repmat(mean(totalPos_pre,3),[1,1,imgNum]);
    tmp=totalPos-repmat(mean(totalPos,3),[1,1,imgNum]);
    upper_delta=model.layer(layerID+1).deform;
    topM=theConf.learn.topM(layerID);
    topN=theConf.learn.topN(layerID);
    
    [~,tmp1]=sort(var(tmp_pre(1,:,:),[],3)+var(tmp_pre(2,:,:),[],3),'ascend');
    tmp1=tmp1(1:ceil(numel(tmp1)*minVarRate));
    tmp2=find(min(xh/2-abs(xh/2-mean(totalPos_pre,3)),[],1)<xh*edgeCutRate);
    [~,candIdx]=sort(sum(model.layer(layerID+1).prob_record,2),'descend');
    candIdx=setdiff(candIdx,tmp1,'stable');
    candIdx=setdiff(candIdx,tmp2,'stable');
    candIdx=candIdx(1:topM);
%     if(layerID==1)
%         layerID
%     end
    tmp_pre=tmp_pre(:,candIdx,:);
    upper_delta=upper_delta(candIdx);
    idx=ones(topN,patternNum,channelNum);
    hw_pre=x2p_(tmp_pre,layerID+1,theConf);
    hw=x2p_(reshape(tmp,[2,patternNum,channelNum,imgNum]),layerID,theConf);
    parSet=repmat(struct('hw_pre',hw_pre,'pNum',patternNum,'parN',topN),[1,channelNum]);
    inferP=reshape(inferredProb,[patternNum,channelNum,imgNum]);
    parfor ch=1:channelNum
        pNum=parSet(ch).pNum;
        the_hw=hw(:,:,ch,:);
        imgN=size(the_hw,4);
        the_hw=reshape(the_hw,[2,pNum,imgN]);
        parM=size(parSet(ch).hw_pre,2);
        parN=parSet(ch).parN;
        tmpIdx=zeros(parN,pNum);
        theProb=inferP(:,ch,:);
        theProb=reshape(theProb,[pNum,imgN]);
        for p=1:pNum
            H=the_hw(1,p,:);
            W=the_hw(2,p,:);
            tmpP=repmat(theProb(p,:),[parM,1]);
            % sqrdist=sum(((repmat(H(:)',[parM,1])-reshape(parSet(ch).hw_pre(1,:,:),[parM,imgN])).^2).*tmpP,2);
            % sqrdist=sqrdist+sum(((repmat(W(:)',[parM,1])-reshape(parSet(ch).hw_pre(2,:,:),[parM,imgN])).^2).*tmpP,2);
            % [~,tmp]=sort(sqrdist,'ascend');
            
            sqrd=(repmat(H(:)',[parM,1])-reshape(parSet(ch).hw_pre(1,:,:),[parM,imgN])).^2;
            sqrd=sqrd+(repmat(W(:)',[parM,1])-reshape(parSet(ch).hw_pre(2,:,:),[parM,imgN])).^2;
            probSum=-sum(exp(-sqrd./(2*mean(sqrd(:)))).*tmpP,2);
            [~,tmp]=sort(probSum,'ascend');
            
            tmpIdx(:,p)=tmp(1:parN);
        end
        idx(:,:,ch)=tmpIdx;
        parSet(ch).hw_pre=[];
    end
    idx=reshape(idx,[topN,patternNum*channelNum]);
    clear parSet hw
    infer_poster=(1/topN).*ones(topN,patternNum*channelNum,imgNum);
    curMu=zeros(2,patternNum*channelNum,imgNum);
    sumWeight=zeros(patternNum*channelNum,imgNum);
    for i=1:topN
        list=idx(i,:);
        w=reshape(infer_poster(i,:,:),[patternNum*channelNum,imgNum])./repmat((upper_delta(list).^2),[1,imgNum]);
        sumWeight=sumWeight+w;
        list=repmat(idx(i,:)',[1,imgNum])+repmat((0:imgNum-1).*topM,[patternNum*channelNum,1]);
        curMu(1,:,:)=curMu(1,:,:)+reshape(tmp_pre(list.*2-1).*w,[1,patternNum*channelNum,imgNum]);
        curMu(2,:,:)=curMu(2,:,:)+reshape(tmp_pre(list.*2).*w,[1,patternNum*channelNum,imgNum]);
    end
    totalIdx=candIdx(idx);
    curDelta=sqrt(1./sumWeight);
    curMu=curMu./repmat(reshape(sumWeight,[1,patternNum*channelNum,imgNum]),[2,1,1]);
    curLambda=ones(patternNum*channelNum,imgNum);
    for i=1:topN
        list=idx(i,:);
        updelta=repmat(upper_delta(list),[1,imgNum]);
        list=repmat(idx(i,:)',[1,imgNum])+repmat((0:imgNum-1).*topM,[patternNum*channelNum,1]);
        tmp=(curMu(1,:,:)-reshape(tmp_pre(list.*2-1),[1,patternNum*channelNum,imgNum])).^2;
        tmp=tmp+(curMu(2,:,:)-reshape(tmp_pre(list.*2),[1,patternNum*channelNum,imgNum])).^2;
        tmp=reshape(tmp,[patternNum*channelNum,imgNum]);
        curLambda=curLambda.*exp(-tmp./((2*topN).*(updelta.^2)))./updelta;
    end
    curLambda=curLambda.*curDelta;
else
    curMu=zeros(2,patternNum*channelNum,imgNum);
    tmp=xh*theConf.learn.deform_.init_delta;
    curDelta=ones(patternNum*channelNum,imgNum).*tmp;
    curLambda=ones(patternNum*channelNum,imgNum);
    totalIdx=[];
end
end
