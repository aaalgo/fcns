#!/usr/bin/env python
import numpy as np

class Stitcher:
    def __init__ (self, image, patch):
        N, H, W, C = image.shape
        assert N == 1
        bh = (H + patch - 1) // patch
        bw = (W + patch - 1) // patch
        grid = []
        for h in range(bh):
            for w in range(bw):
                grid.append((h * patch, min(H, (h+1)*patch), w * patch, min(W, (w+1)*patch)))
        self.image = image
        self.patch = patch
        self.grid = grid
        pass

    def split (self):
        _, _, _, C = self.image.shape
        patches = np.zeros((len(self.grid), self.patch, self.patch, C))
        for i, (h1, h2, w1, w2) in enumerate(self.grid):
            patches[i,:(h2-h1),:(w2-w1),:] = self.image[0,h1:h2,w1:w2,:]
        return patches

    def stitch (self, patches):
        _, H, W, _ = self.image.shape
        pad = False
        if len(patches.shape) == 3:
            pad = True
            patches = patches.reshape(patches.shape + (1,))
            pass
        _, _, _, C = patches.shape
        image = np.zeros((1, H, W, C), dtype=patches.dtype)
        for i, (h1, h2, w1, w2) in enumerate(self.grid):
            image[0,h1:h2,w1:w2,:] = patches[i,:(h2-h1),:(w2-w1),:]
        if pad:
            image = image.reshape(image.shape[:3])
        return image

if __name__ == '__main__':
    import sys
    import cv2

    image = cv2.imread('xxx.png', -1)
    if len(image.shape) == 2:
        image = image.reshape((1,) + image.shape + (1,))
    else:
        image = image.reshape((1,) + image.shape)

    for ps in [10, 15, 17, 19, 30, 31]:
        stitcher = Stitcher(image, ps)
        assert (image == stitcher.stitch(stitcher.split())).all()

