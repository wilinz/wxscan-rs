"""Export detect.tflite and sr.tflite and compare them against the Caffe reference output."""
import numpy as np, tensorflow as tf, caffe2tf, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'tflite_out'
os.makedirs(OUT, exist_ok=True)
ref = np.load('ref_outputs.npz')


def export(model, path, sample_shapes):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = []
    # Keep float32 and allow dynamic H/W
    conv._experimental_lower_tensor_list_ops = False
    tfl = conv.convert()
    open(path, 'wb').write(tfl)
    print(f"{path}: {len(tfl)/1024:.1f} KB")
    return tfl


def run_tflite(tfl_bytes, x, n_out):
    it = tf.lite.Interpreter(model_content=tfl_bytes)
    inp = it.get_input_details()[0]
    it.resize_tensor_input(inp['index'], list(x.shape), strict=False)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]['index'], x)
    it.invoke()
    return [it.get_tensor(o['index']) for o in it.get_output_details()[:n_out]], it


# ── SR ──
c = caffe2tf.Converter('models/sr.caffemodel')
sr = c.build((None, None), ['fc'])
sr_bytes = export(sr, f'{OUT}/sr.tflite', None)
x = (ref['sr_in'].astype(np.float32)/255.0)[None, :, :, None]
(out,), _ = run_tflite(sr_bytes, x, 1)
d = np.abs(ref['sr_out'].ravel() - out.ravel())
print(f"  sr tflite vs caffe: max={d.max():.3e} mean={d.mean():.3e}")

# ── Detect ──
c2 = caffe2tf.Converter('models/detect.caffemodel')
det = c2.build((None, None), ['mbox_loc', 'mbox_conf_flatten'])
det_bytes = export(det, f'{OUT}/detect.tflite', None)
names = [o.name for o in det.outputs]
print('  detect outputs order:', names)
for tag in ['det', 'det2']:
    x = (ref[tag + '_in'].astype(np.float32)/255.0)[None, :, :, None]
    outs, it = run_tflite(det_bytes, x, 2)
    od = it.get_output_details()
    got = {}
    for o, arr in zip(od, outs):
        got[o['name']] = arr
    # Distinguish loc (4*N) from conf (2*N) by element count
    arrs = sorted(outs, key=lambda a: -a.size)
    loc, conf = arrs[0], arrs[1]
    dl = np.abs(ref[tag+'_mbox_loc'].ravel() - loc.ravel())
    dc = np.abs(ref[tag+'_mbox_conf_flatten'].ravel() - conf.ravel())
    print(f"  {tag} tflite: loc max={dl.max():.3e}  conf max={dc.max():.3e}")
