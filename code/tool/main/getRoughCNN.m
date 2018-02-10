function [batch_f,batch_f_flip,stat_all]=getRoughCNN(net,theConf,images,images_neg)
parSize=theConf.parallel.parpoolSize;
[stat_all.avg_res,stat_all.avgrelu_res,stat_all.sqrtvar_layer]=getAvgResponse(images_neg,theConf,net);
objNum=size(images,4);
stat_all.objNum=objNum;
batch_f=cell(objNum,1);
batch_f_flip=cell(objNum,1);
for id_start=1:parSize:objNum
    for par=1:min(objNum-id_start+1,parSize)
        objID=id_start+par-1;
        [res,~]=getObjFeature_img(images(:,:,:,objID),theConf,net);
        [res_flip,~]=getObjFeature_img(images(:,:,:,objID),theConf,net);
        for layerID=1:length(theConf.convnet.targetLayers)
            oriLayer=theConf.convnet.targetLayers(layerID);
            res(oriLayer).x=getX(res,layerID,theConf,stat_all);
            res_flip(oriLayer).x=getX(res_flip,layerID,theConf,stat_all);
        end
        batch_f{objID}=roughCNN_compress(res,theConf);
        batch_f_flip{objID}=roughCNN_compress(res_flip,theConf);
        clear res res_flip
    end
    disp(id_start/objNum);
end
end


function x=getX(res,layerID,theConf,stat_all)
x=gather(res(theConf.convnet.targetLayers(layerID)).x);
v.x=x./stat_all.sqrtvar_layer(layerID).x;
v.valid=(x>0);
v.x(v.valid==0)=-1;
x=v.x;
end


function [avg_res,avgrelu_res,sqrtvar_layer]=getAvgResponse(images,theConf,theNet)
objNum=size(images,4);
layerNum=length(theConf.convnet.targetLayers);
avg_res(layerNum).x=[];
avgrelu_res(layerNum).x=[];
sqrtvar_layer(layerNum).x=[];
for objID=1:objNum
    [res,~]=getObjFeature_img(images(:,:,:,objID),theConf,theNet);
    for layer=1:layerNum
        x=f(res,layer,theConf);
        if(objID==1)
            avg_res(layer).x=x;
            avgrelu_res(layer).x=max(x,0);
            sqrtvar_layer(layer).x=x.^2;
        else
            avg_res(layer).x=avg_res(layer).x+x;
            avgrelu_res(layer).x=avgrelu_res(layer).x+max(x,0);
            sqrtvar_layer(layer).x=sqrtvar_layer(layer).x+x.^2;
        end
    end
    fprintf('%d/%d\n',objID,objNum)
end
for layer=1:layerNum
    avg_res(layer).x=avg_res(layer).x./objNum;
    avgrelu_res(layer).x=avgrelu_res(layer).x./objNum;
    sqrtvar_layer(layer).x=sqrt(sqrtvar_layer(layer).x./objNum-avg_res(layer).x.^2);
end
end
