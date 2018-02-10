function [prob,prior,logDer,pos,imgFlip]=doFlipConfigureation(IsAlter,batch_f,batch_f_flip,theConf,imgFlip,layerID,coord,deform,curMu,curDelta,curLambda)
oriLayer=theConf.convnet.targetLayers(layerID);
layerNum=length(theConf.convnet.targetLayers);
if((layerID==layerNum)&&(~IsAlter))
    tmp=roughCNN_uncompress(batch_f,theConf);
    x=tmp(oriLayer).x;
    tmp=roughCNN_uncompress(batch_f_flip,theConf);
    x_flip=tmp(oriLayer).x;
    [prob,prior,logDer,pos,~]=doE(x,theConf,coord,deform,layerID,curMu,curDelta,curLambda);
    [prob_flip,prior_flip,logDer_flip,pos_flip,~]=doE(x_flip,theConf,coord,deform,layerID,curMu,curDelta,curLambda);
    if(sum(log(prob(:)).*prior(:))<sum(log(prob_flip(:)).*prior_flip(:)))
        prob=prob_flip;
        prior=prior_flip;
        logDer=logDer_flip;
        pos=pos_flip;
        imgFlip=true;
    else
        imgFlip=false;
    end
else
    switch(imgFlip)
        case 1
            tmp=roughCNN_uncompress(batch_f_flip,theConf);
        case 0
            tmp=roughCNN_uncompress(batch_f,theConf);
        otherwise
            error('Errors here.');
    end
    x=tmp(oriLayer).x;
    [prob,prior,logDer,pos,~]=doE(x,theConf,coord,deform,layerID,curMu,curDelta,curLambda);
end
end
