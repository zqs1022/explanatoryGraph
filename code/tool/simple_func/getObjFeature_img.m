function [res,I_patch]=getObjFeature_img(I_patch,theConf,theNet)
I_patch=single(I_patch);
if(size(theNet.meta.normalization.averageImage,1)==1)
    im_=I_patch-repmat(theNet.meta.normalization.averageImage,[theNet.meta.normalization.imageSize(1:2),1]);
else
    im_=I_patch-theNet.meta.normalization.averageImage;
end
if(isa(theNet.layers{1}.weights{1},'gpuArray'))
    res=vl_simplenn(theNet,gpuArray(im_));
    for i=1:length(res)
        res(i).x=gather(res(i).x);
    end
else
    res=vl_simplenn(theNet,im_);
end
I_patch=uint8(I_patch);
end
