function res_c=roughCNN_compress(res,theConf)
IsUseUInt8=true;

num=length(res);
res_c=repmat(struct('size',[],'x',[],'rangeX',[],'minX',[]),[1,num]);
for i=theConf.convnet.targetLayers
    x=gather(res(i).x);
    theSize=size(x);
    res_c(i).size=uint16(theSize);
    minX=min(min(x,[],1),[],2);
    x=x-repmat(minX,[theSize(1),theSize(2),1]);
    rangeX=max(max(x,[],1),[],2);
    res_c(i).minX=minX;
    res_c(i).rangeX=rangeX;
    if(IsUseUInt8)
        res_c(i).x=uint8(x.*repmat(255.0./rangeX,[theSize(1),theSize(2),1]));
    else
        res_c(i).x=uint16(x.*repmat(65535.0./rangeX,[theSize(1),theSize(2),1]));
    end
end
end
