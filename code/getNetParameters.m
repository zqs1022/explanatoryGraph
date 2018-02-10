function conf=getNetParameters(type,net)
if(nargin<2)
    net=[];
end

try
    parpool;
catch
    delete(gcp);
    parpool;
end
p=gcp;
conf.parallel.parpoolSize=p.NumWorkers;
switch(type)
    case 'vgg16'
        conf=getConvNet_VGG16(conf,net); %VGG16
    case 'VAEGAN'
        conf=getConvNet_ResNet_VAEGAN(conf,[32,16,8],[8,12,18]); %VAEGAN
    case 'ResNet'
        conf=getConvNet_ResNet_VAEGAN(conf,[28,14,7],[15,21,30]); %ResNet
    otherwise
        % please design your own net parameters
end
end



function convnet=getConvNetPara(convnet,net)
len=length(convnet.convLayers);
convnet.lastLayer=length(net.layers)-1;
convnet.targetLayers=1+convnet.convLayers;
convnet.targetScale=zeros(1,len);
convnet.targetStride=zeros(1,len);
convnet.targetCenter=zeros(1,len);
for i=1:len
    tarLay=convnet.convLayers(i);
    layer=net.layers{tarLay};
    if(((~strcmp(layer.type,'conv'))&&(~strcmp(layer.type,'dagnn.Conv')))||(var(layer.pad)>0)||(layer.size(1)~=layer.size(2))||(layer.stride(1)~=layer.stride(2)))
        error('Errors in function getConvNetPara.');
    end
    pad=layer.pad(1);
    scale=layer.size(1);
    stride=layer.stride(1);
    if(i==1)
        convnet.targetStride(i)=stride;
        convnet.targetScale(i)=scale;
        convnet.targetCenter(i)=(1+scale-pad*2)/2;
    else
        IsPool=false;
        poolStride=0;
        poolSize=0;
        poolPad=0;
        for j=convnet.convLayers(i-1)+1:tarLay-1
            if(strcmp(net.layers{j}.type,'pool'))
                IsPool=true;
                poolSize=net.layers{j}.pool(1);
                poolStride=net.layers{j}.stride(1);
                poolPad=net.layers{j}.pad(1);
            end
        end
        convnet.targetStride(i)=(1+IsPool*(poolStride-1))*stride*convnet.targetStride(i-1);
        convnet.targetScale(i)=convnet.targetScale(i-1)+IsPool*(poolSize-1)*convnet.targetStride(i-1)+convnet.targetStride(i)*(scale-1);
        if(IsPool)
            convnet.targetCenter(i)=(scale-pad*2-1)*poolStride*convnet.targetStride(i-1)/2+(convnet.targetCenter(i-1)+convnet.targetStride(i-1)*(poolSize-2*poolPad-1)/2);
        else
            convnet.targetCenter(i)=(scale-pad*2-1)*convnet.targetStride(i-1)/2+convnet.targetCenter(i-1);
        end
    end
end
convnet.targetLayers=convnet.targetLayers(convnet.validLayers);
convnet.targetScale=convnet.targetScale(convnet.validLayers);
convnet.targetStride=convnet.targetStride(convnet.validLayers);
convnet.targetCenter=convnet.targetCenter(convnet.validLayers);
convnet=rmfield(convnet,{'convLayers','validLayers'});
end


function conf=getConvNet_VGG16(conf,net)
convnet.codedir='../matconvnet-1.0-beta24/matlab/';
convnet.convLayers=[1,3,6,8,11,13,15,18,20,22,25,27,29];
convnet.validLayers=[9,10,12,13];
convnet.imgSize=[224,224];
conf.convnet=getConvNetPara(convnet,net);
conf.learn.positionCandNum=6;
conf.learn.patternDensity=[0.05,0.05,0.1,0.1]./1.0;
conf.learn.search_.maxRange=[0.3,0.3,0.3,0.3];
conf.learn.search_.deform_ratio=3;
conf.learn.deform_.init_delta=0.15;
conf.learn.deform_.max_delta=1.0./sqrt(conf.learn.patternDensity.*([28,28,14,14].^2));
conf.learn.deform_.min_delta=conf.learn.deform_.max_delta./10;
conf.learn.map_delta=0.025;
conf.learn.topN=[15,15,15,0];
conf.learn.topM=[600,600,600,0];
conf.learn.validTau=0.1;
end



function conf=getConvNet_ResNet_VAEGAN(conf,mapSize,startCenter)
convnet.imgSize=[224,224];
layerNum=numel(mapSize);
convnet.lastLayer=layerNum;
convnet.targetLayers=1:layerNum;
convnet.targetScale=startCenter.*2-1;
stride=(convnet.imgSize(1)-convnet.targetScale)./(mapSize-1);
convnet.targetStride=stride;
convnet.targetCenter=startCenter;
conf.convnet=convnet;

if(layerNum~=3)
    error('Errors here');
else
    conf.learn.positionCandNum=6;
    conf.learn.patternDensity=[0.05,0.1,0.2]./1.0;
    conf.learn.search_.maxRange=[0.3,0.3,0.3];
    conf.learn.search_.deform_ratio=3;
    conf.learn.deform_.init_delta=0.15;
    conf.learn.deform_.max_delta=1.0./sqrt(conf.learn.patternDensity.*(mapSize.^2));
    conf.learn.deform_.min_delta=conf.learn.deform_.max_delta./10;
    conf.learn.map_delta=0.025;
    conf.learn.topN=[15,15,0];
    conf.learn.topM=[600,600,0];
    conf.learn.validTau=0.1;
end
end
