function pHW=x2p_(xHW,layer,theConf)
if(isfield(theConf,'convnet'))
    Stride=theConf.convnet.targetStride(layer);
    centerStart=theConf.convnet.targetCenter(layer);
else
    Stride=theConf.targetStride(layer);
    centerStart=theConf.targetCenter(layer);
end
pHW=centerStart+(xHW-1).*Stride;
end
