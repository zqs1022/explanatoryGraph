function IsValid=IsAreaValid(bndbox,MinArea)
xmin=str2double(bndbox.xmin);
xmax=str2double(bndbox.xmax);
ymin=str2double(bndbox.ymin);
ymax=str2double(bndbox.ymax);
area=(xmax-xmin+1)*(ymax-ymin+1);
IsValid=(area>=MinArea);
end

