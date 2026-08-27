"""Export detect.onnx and sr.onnx and compare them against the Caffe reference output."""
import os, sys
import numpy as np, onnx, onnxruntime as ort
import caffe2onnx

OUT = sys.argv[1] if len(sys.argv) > 1 else 'onnx_out'
os.makedirs(OUT, exist_ok=True)
ref = np.load('ref_outputs.npz')


def export(model, path):
    onnx.save(model, path)
    print(f"{path}: {os.path.getsize(path)/1024:.1f} KB")
    return path


def run(path, x, names):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return sess.run(names, {'data': x})


# ── SR ──
sr = caffe2onnx.Converter('models/sr.caffemodel').build({'fc': [1, 1, None, None]}, 'sr')
sr_path = export(sr, f'{OUT}/sr.onnx')
x = (ref['sr_in'].astype(np.float32) / 255.0)[None, None]
(out,) = run(sr_path, x, ['fc'])
d = np.abs(ref['sr_out'].ravel() - out.ravel())
print(f"  sr onnx vs caffe: max={d.max():.3e} mean={d.mean():.3e}")

# ── Detect ──
out_shapes = {'mbox_loc': [1, None], 'mbox_conf_flatten': [1, None]}
names = list(out_shapes)
det = caffe2onnx.Converter('models/detect.caffemodel').build(out_shapes, 'detect')
det_path = export(det, f'{OUT}/detect.onnx')
for tag in ['det', 'det2']:
    x = (ref[tag + '_in'].astype(np.float32) / 255.0)[None, None]
    loc, conf = run(det_path, x, names)
    dl = np.abs(ref[tag + '_mbox_loc'].ravel() - loc.ravel())
    dc = np.abs(ref[tag + '_mbox_conf_flatten'].ravel() - conf.ravel())
    print(f"  {tag} onnx: loc max={dl.max():.3e}  conf max={dc.max():.3e}")
