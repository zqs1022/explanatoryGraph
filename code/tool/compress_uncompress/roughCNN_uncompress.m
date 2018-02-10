function res=roughCNN_uncompress(res_c,theConf)
IsUseUInt8=true;

num=length(res_c);
res=repmat(struct('x',[],'dzdx',[],'dzdw',[],'aux',[],'stats',[],'time',0,'backwardTime',0),[1,num]);
for i=theConf.convnet.targetLayers
    try
        theSize=res_c(i).size;
        RC=res_c(i);
    catch
        theSize=res_c{i}.size;
        RC=res_c{i};
    end
    if(IsUseUInt8)
        x=(single(RC.x).*repmat(single(RC.rangeX),[theSize(1),theSize(2),1]))./255.0;
    else
        x=(single(RC.x).*repmat(single(RC.rangeX),[theSize(1),theSize(2),1]))./65535.0;
    end
    x=x+repmat(single(RC.minX),[theSize(1),theSize(2),1]);
    res(i).x=reshape(x,theSize);
end
end
