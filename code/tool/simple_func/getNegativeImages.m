function images_neg=getNegativeImages(tarSize)
imgNum=1000;
root='../data/neg/';
images_neg=zeros(tarSize(1),tarSize(2),3,imgNum,'uint8');
for imgID=1:imgNum
    I=imread(sprintf('%s%05d.JPEG',root,imgID));
    if(size(I,3)==1)
        I=repmat(I,[1,1,3]);
    end
    I=imresize(I,tarSize,'bilinear');
    images_neg(:,:,:,imgID)=I;
end
end
