"""Produce reference output with the OpenCV 4.x Caffe importer, saved as npz for comparison on the TF side."""
import numpy as np, cv2, sys
rng = np.random.default_rng(0)
out = {}

# SR
H, W = 96, 128
img = (rng.random((H, W)) * 255).astype(np.uint8)
net = cv2.dnn.readNetFromCaffe('models/sr.prototxt', 'models/sr.caffemodel')
net.setInput(cv2.dnn.blobFromImage(img, 1.0/255, (W, H), (0.0,), False, False))
out['sr_in'] = img
out['sr_out'] = net.forward()

# Detect, fixed at 384x384 as declared in the prototxt
H2 = W2 = 384
img2 = (rng.random((H2, W2)) * 255).astype(np.uint8)
det = cv2.dnn.readNetFromCaffe('models/detect.prototxt', 'models/detect.caffemodel')
det.setInput(cv2.dnn.blobFromImage(img2, 1.0/255, (W2, H2), (0.0,0.0,0.0), False, False))
names = ['mbox_loc', 'mbox_conf_flatten', 'mbox_priorbox', 'detection_output']
res = det.forward(names)
out['det_in'] = img2
for n, r in zip(names, res):
    out['det_' + n] = r
    print(n, r.shape)

# A non-square size as well, to exercise dynamic shapes
H3, W3 = 224, 320
img3 = (rng.random((H3, W3)) * 255).astype(np.uint8)
det2 = cv2.dnn.readNetFromCaffe('models/detect.prototxt', 'models/detect.caffemodel')
det2.setInput(cv2.dnn.blobFromImage(img3, 1.0/255, (W3, H3), (0.0,0.0,0.0), False, False))
res2 = det2.forward(names)
out['det2_in'] = img3
for n, r in zip(names, res2):
    out['det2_' + n] = r
    print('dyn', n, r.shape)

np.savez('ref_outputs.npz', **out)
print('saved')
