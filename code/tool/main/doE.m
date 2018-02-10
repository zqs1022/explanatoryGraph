function [prob,prior,logDer,pos,posCand]=doE(x,theConf,coord,deform_delta,layerID,curMu,curDelta,curLambda)
x=double(x);
[xh,xw,channelNum]=size(x);
if(xh~=xw)
    error('Errors here.');
end

[posCand,map_delta,halfRange]=getBasicSettings(x,theConf,deform_delta,layerID);
posCandNum=size(posCand,1);
patternNum=size(coord,2);
curMu=reshape(curMu,[2,patternNum,channelNum]);
curDelta=reshape(curDelta,[patternNum,channelNum]);
coord=coord+curMu;
[prob,prior,logDer,pos]=getProb_local_mex(x,coord,posCand,halfRange,curDelta,map_delta,channelNum);
prob=reshape(prob,[posCandNum^2,patternNum*channelNum]).*repmat(curLambda',[posCandNum^2,1]);
prior=reshape(prior,[posCandNum^2,patternNum*channelNum]);
logDer=reshape(logDer,[2,posCandNum^2,patternNum*channelNum]);
pos=reshape(pos,[2,posCandNum^2,patternNum*channelNum]);

% for par=5 %1:channelNum
%     i=13;
%     tmp=reshape(prob(:,i+(par-1)*patternNum),[6,6]);tmp(tmp<0)=0;
%     subplot(1,2,1);hold on;imagesc(tmp);
%     tmp=x(:,:,par);tmp(tmp<0)=0;
%     xmin=coord(2,i,par)-halfRange;
%     xmax=coord(2,i,par)+halfRange;
%     ymin=coord(1,i,par)-halfRange;
%     ymax=coord(1,i,par)+halfRange;
%     %[coord(:,i,par)',x(round(coord(1,i,par)),round(coord(2,i,par)),par)]
%     subplot(1,2,2);hold on;imagesc(tmp);plot(coord(2,i,par),coord(1,i,par),'wo');plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],'w-');
% end

end
