for k = 1:289
    a = imread(sprintf('image (%d).png',k));
    b = imresize(a,[299 299]);
    imwrite(b,sprintf('image_224 (%d).png',k))
end