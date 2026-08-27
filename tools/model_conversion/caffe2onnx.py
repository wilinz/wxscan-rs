"""Rebuild the two wechat_qrcode Caffe models layer by layer as ONNX graphs.

The companion of `caffe2tf.py`, for the pure Rust (tract) backend. ONNX is
NCHW like Caffe, so unlike the TFLite path there is no layout conversion and
most layers map one to one:
  * Conv/ConvTranspose take Caffe's weight layout, groups, dilation and
    symmetric padding directly
  * MaxPool with `ceil_mode=1` is Caffe's CEIL windowing
  * BatchNorm (plus Scale) is folded into a per-channel Mul/Add pair
  * Permute is a real Transpose here: the SSD head flattens in NHWC order, and
    in an NCHW graph that ordering has to be produced explicitly

Height and width stay symbolic, since super resolution runs on crops of
whatever size the pipeline hands it.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import caffe_pb2


def blob_array(blob):
    if len(blob.shape.dim):
        shape = list(blob.shape.dim)
    else:
        shape = [blob.num, blob.channels, blob.height, blob.width]
    return np.array(blob.data, dtype=np.float32).reshape(shape)


def first(vals, default):
    return vals[0] if len(vals) else default


class Converter:
    def __init__(self, caffemodel_path):
        net = caffe_pb2.NetParameter()
        net.ParseFromString(open(caffemodel_path, 'rb').read())
        self.net = net
        self.blobs = {}      # Caffe blob name -> current ONNX tensor name
        self.nodes = []
        self.inits = []
        self.counter = {}

    # Naming: Caffe blobs are reused in place, ONNX tensors are single
    # assignment, so every produced tensor gets a fresh name.
    def _name(self, base):
        n = self.counter.get(base, 0)
        self.counter[base] = n + 1
        return base if n == 0 else f'{base}_{n}'

    def _const(self, base, array):
        name = self._name(base)
        self.inits.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def _node(self, op, inputs, out_base, **attrs):
        out = self._name(out_base)
        self.nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    def build(self, outputs, model_name):
        """outputs: {Caffe blob name: shape}, where `None` is a dimension that
        follows the input size. H and W stay symbolic, since super resolution
        runs on crops of whatever size the pipeline hands it."""
        self.blobs['data'] = 'data'
        inp = helper.make_tensor_value_info(
            'data', TensorProto.FLOAT, [1, 1, 'H', 'W'])

        for layer in self.net.layer:
            if layer.type == 'Input':
                continue
            handler = getattr(self, f'_op_{layer.type.lower()}', None)
            if handler is None:
                raise NotImplementedError(f'layer type {layer.type} ({layer.name})')
            handler(layer)

        outs = []
        for o, shape in outputs.items():
            # Tensors are named after the layer that writes them, so an output
            # blob keeps its Caffe name unless some later layer wrote to it in
            # place. Neither model does that to an output.
            assert self.blobs[o] == o, f'{o} was rewritten in place'
            outs.append(helper.make_tensor_value_info(
                o, TensorProto.FLOAT, shape))

        graph = helper.make_graph(self.nodes, model_name, [inp], outs, self.inits)
        model = helper.make_model(
            graph, producer_name='wxscan caffe2onnx',
            opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        # Deliberately no `shape_inference.infer_shapes`: it would record a
        # shape for every intermediate tensor in terms of the symbols H and W,
        # and a runtime that pins those to a concrete size then has to reconcile
        # the two. Only the input and output shapes are declared, and they leave
        # the varying dimensions unknown.
        return model

    # Layers
    def _op_split(self, layer):
        src = self.blobs[layer.bottom[0]]
        for t in layer.top:
            self.blobs[t] = src

    def _op_batchnorm(self, layer):
        x = self.blobs[layer.bottom[0]]
        name = layer.name.replace('/', '_')
        mean = blob_array(layer.blobs[0]).ravel()
        var = blob_array(layer.blobs[1]).ravel()
        scale_factor = blob_array(layer.blobs[2]).ravel()[0]
        if scale_factor != 0:
            mean = mean / scale_factor
            var = var / scale_factor
        eps = layer.batch_norm_param.eps or 1e-5
        inv = 1.0 / np.sqrt(var + eps)
        self.blobs[layer.top[0]] = self._affine(x, name, inv, -mean * inv)

    def _op_scale(self, layer):
        x = self.blobs[layer.bottom[0]]
        name = layer.name.replace('/', '_')
        gamma = blob_array(layer.blobs[0]).ravel()
        beta = (blob_array(layer.blobs[1]).ravel() if len(layer.blobs) > 1
                else np.zeros_like(gamma))
        self.blobs[layer.top[0]] = self._affine(x, name, gamma, beta)

    def _affine(self, x, name, scale, bias):
        """Per-channel `x * scale + bias`, broadcast over H and W."""
        s = self._const(name + '_scale', scale.astype(np.float32).reshape(-1, 1, 1))
        b = self._const(name + '_bias', bias.astype(np.float32).reshape(-1, 1, 1))
        y = self._node('Mul', [x, s], name + '_mul')
        return self._node('Add', [y, b], name)

    def _op_relu(self, layer):
        x = self.blobs[layer.bottom[0]]
        name = layer.name.replace('/', '_')
        slope = layer.relu_param.negative_slope
        if slope:
            self.blobs[layer.top[0]] = self._node(
                'LeakyRelu', [x], name, alpha=float(slope))
        else:
            self.blobs[layer.top[0]] = self._node('Relu', [x], name)

    def _op_convolution(self, layer):
        x = self.blobs[layer.bottom[0]]
        p = layer.convolution_param
        name = layer.name.replace('/', '_')
        w = blob_array(layer.blobs[0])           # (out, in/g, kh, kw), as ONNX wants
        ins = [x, self._const(name + '_w', w)]
        if p.bias_term:
            ins.append(self._const(name + '_b', blob_array(layer.blobs[1]).ravel()))
        stride, dil, pad = first(p.stride, 1), first(p.dilation, 1), first(p.pad, 0)
        self.blobs[layer.top[0]] = self._node(
            'Conv', ins, name,
            kernel_shape=[w.shape[2], w.shape[3]],
            strides=[stride, stride], dilations=[dil, dil],
            pads=[pad, pad, pad, pad], group=p.group or 1)

    def _op_deconvolution(self, layer):
        x = self.blobs[layer.bottom[0]]
        p = layer.convolution_param
        name = layer.name.replace('/', '_')
        w = blob_array(layer.blobs[0])           # (in, out/g, kh, kw), as ONNX wants
        ins = [x, self._const(name + '_w', w)]
        if p.bias_term:
            ins.append(self._const(name + '_b', blob_array(layer.blobs[1]).ravel()))
        stride, dil, pad = first(p.stride, 1), first(p.dilation, 1), first(p.pad, 0)
        # Caffe: out = stride*(H-1) + k - 2*pad, which is what ONNX computes
        # from symmetric pads, so no output_padding is needed.
        self.blobs[layer.top[0]] = self._node(
            'ConvTranspose', ins, name,
            kernel_shape=[w.shape[2], w.shape[3]],
            strides=[stride, stride], dilations=[dil, dil],
            pads=[pad, pad, pad, pad], group=p.group or 1)

    def _op_pooling(self, layer):
        p = layer.pooling_param
        assert p.pool == 0, 'only MAX pooling'
        assert p.pad == 0
        k, s = p.kernel_size, p.stride
        self.blobs[layer.top[0]] = self._node(
            'MaxPool', [self.blobs[layer.bottom[0]]], layer.name.replace('/', '_'),
            kernel_shape=[k, k], strides=[s, s], pads=[0, 0, 0, 0], ceil_mode=1)

    def _op_eltwise(self, layer):
        assert layer.eltwise_param.operation == 1, 'only SUM'
        xs = [self.blobs[b] for b in layer.bottom]
        y = xs[0]
        name = layer.name.replace('/', '_')
        for x in xs[1:]:
            y = self._node('Add', [y, x], name)
        self.blobs[layer.top[0]] = y

    def _op_concat(self, layer):
        if any(b not in self.blobs for b in layer.bottom):
            return          # mbox_priorbox: prior boxes are generated on the Rust side
        self.blobs[layer.top[0]] = self._node(
            'Concat', [self.blobs[b] for b in layer.bottom],
            layer.name.replace('/', '_'), axis=layer.concat_param.axis)

    def _op_permute(self, layer):
        # PermuteParameter from the SSD branch is absent from the BVLC
        # caffe.proto, so its parameters cannot be read. Every Permute in this
        # prototxt has order (0,2,3,1); in NCHW that is a real transpose, and it
        # is what puts the SSD head's flattened output in upstream's order.
        self.blobs[layer.top[0]] = self._node(
            'Transpose', [self.blobs[layer.bottom[0]]],
            layer.name.replace('/', '_'), perm=[0, 2, 3, 1])

    def _op_flatten(self, layer):
        assert layer.flatten_param.axis == 1
        self.blobs[layer.top[0]] = self._node(
            'Flatten', [self.blobs[layer.bottom[0]]],
            layer.name.replace('/', '_'), axis=1)

    def _op_reshape(self, layer):
        dims = list(layer.reshape_param.shape.dim)
        assert dims == [0, -1, 2], f'unexpected reshape {dims}'
        name = layer.name.replace('/', '_')
        # A 0 keeps the input dimension, in Caffe and in ONNX alike
        shape = self._const(name + '_shape', np.array(dims, dtype=np.int64))
        self.blobs[layer.top[0]] = self._node(
            'Reshape', [self.blobs[layer.bottom[0]], shape], name)

    def _op_softmax(self, layer):
        self.blobs[layer.top[0]] = self._node(
            'Softmax', [self.blobs[layer.bottom[0]]],
            layer.name.replace('/', '_'), axis=layer.softmax_param.axis)

    def _op_crop(self, layer):
        x = self.blobs[layer.bottom[0]]
        ref = self.blobs[layer.bottom[1]]
        assert layer.crop_param.axis == 2 and not list(layer.crop_param.offset)
        name = layer.name.replace('/', '_')
        # Crop x to the reference's H and W, which are only known at run time
        ref_shape = self._node('Shape', [ref], name + '_refshape')
        ends = self._node(
            'Slice',
            [ref_shape,
             self._const(name + '_hw_start', np.array([2], dtype=np.int64)),
             self._const(name + '_hw_end', np.array([4], dtype=np.int64))],
            name + '_hw')
        self.blobs[layer.top[0]] = self._node(
            'Slice',
            [x,
             self._const(name + '_start', np.array([0, 0], dtype=np.int64)),
             ends,
             self._const(name + '_axes', np.array([2, 3], dtype=np.int64))],
            name)

    def _op_priorbox(self, layer):
        pass          # prior boxes are generated on the Rust side

    def _op_detectionoutput(self, layer):
        pass          # decoding and NMS are done on the Rust side
