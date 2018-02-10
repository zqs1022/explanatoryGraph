function x=f(res,layer,theConf)
x=gather(res(theConf.convnet.targetLayers(layer)).x);
end

