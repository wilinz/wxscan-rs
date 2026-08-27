import numpy as np, tensorflow as tf, caffe2tf
ref = np.load('ref_outputs.npz')

def cmp(name, a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    n = min(a.size, b.size)
    d = np.abs(a[:n] - b[:n])
    print(f"{name:24s} shapes {a.size} vs {b.size}  max={d.max():.3e} mean={d.mean():.3e}")

# ── SR ──
c = caffe2tf.Converter('models/sr.caffemodel')
m = c.build((None, None), ['fc'])
img = ref['sr_in']
x = (img.astype(np.float32)/255.0)[None, :, :, None]
out = m(x).numpy()
print('sr tf out', out.shape, 'caffe', ref['sr_out'].shape)
cmp('sr', ref['sr_out'], out[0, :, :, 0])

# ── Detect ──
c2 = caffe2tf.Converter('models/detect.caffemodel')
md = c2.build((None, None), ['mbox_loc', 'mbox_conf_flatten'])
for tag in ['det', 'det2']:
    img = ref[tag + '_in']
    x = (img.astype(np.float32)/255.0)[None, :, :, None]
    loc, conf = [t.numpy() for t in md(x)]
    cmp(tag + ' mbox_loc', ref[tag + '_mbox_loc'], loc)
    cmp(tag + ' mbox_conf', ref[tag + '_mbox_conf_flatten'], conf)
