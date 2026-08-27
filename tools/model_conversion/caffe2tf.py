"""Rebuild the two wechat_qrcode Caffe models layer by layer as TF graphs and export them to TFLite.

Only the layers these two models use are covered. The main correspondences:
  * Caffe NCHW -> TF NHWC; conv weights (out,in/g,kh,kw) -> (kh,kw,in/g,out)
  * depthwise (group == num_output) → DepthwiseConv2D
  * dilation is expanded into a sparse larger kernel, since TF does not support
    stride > 1 together with dilation > 1
  * Caffe pads symmetrically on all sides, which is not equivalent to TF
    padding='same' when stride > 1 (TF puts the extra padding on the right and
    bottom), so ZeroPadding2D followed by 'valid' is used throughout
  * MaxPool k3 s2 p0 uses CEIL mode, equivalent to padding one -inf column and
    row on the right and bottom and then pooling with valid padding
  * BatchNorm (plus Scale) is folded into a per-channel affine transform
"""
import numpy as np
import tensorflow as tf
import caffe_pb2

L = tf.keras.layers


def blob_array(blob):
    if len(blob.shape.dim):
        shape = list(blob.shape.dim)
    else:
        shape = [blob.num, blob.channels, blob.height, blob.width]
    return np.array(blob.data, dtype=np.float32).reshape(shape)


def expand_dilation(w, dil):
    """Sparsify (kh,kw,...) by folding dilation into the kernel, returning an equivalent larger kernel."""
    if dil == 1:
        return w
    kh, kw = w.shape[0], w.shape[1]
    nh, nw = (kh - 1) * dil + 1, (kw - 1) * dil + 1
    out = np.zeros((nh, nw) + w.shape[2:], dtype=w.dtype)
    out[::dil, ::dil] = w
    return out


def first(vals, default):
    return vals[0] if len(vals) else default


class Converter:
    def __init__(self, caffemodel_path):
        net = caffe_pb2.NetParameter()
        net.ParseFromString(open(caffemodel_path, 'rb').read())
        self.net = net
        self.blobs = {}

    def build(self, input_shape, outputs):
        """input_shape: (H, W), where None means dynamic. outputs: list of Caffe blob names."""
        h, w = input_shape
        inp = L.Input(shape=(h, w, 1), batch_size=1, name='data')
        self.blobs['data'] = inp

        for layer in self.net.layer:
            t = layer.type
            if t == 'Input':
                continue
            handler = getattr(self, f'_op_{t.lower()}', None)
            if handler is None:
                raise NotImplementedError(f'layer type {t} ({layer.name})')
            handler(layer)

        return tf.keras.Model(inp, [self.blobs[o] for o in outputs])

    # Layers
    def _op_split(self, layer):
        src = self.blobs[layer.bottom[0]]
        for t in layer.top:
            self.blobs[t] = src

    skip_bn = False

    def _op_batchnorm(self, layer):
        if self.skip_bn:
            self.blobs[layer.top[0]] = self.blobs[layer.bottom[0]]
            return
        x = self.blobs[layer.bottom[0]]
        mean = blob_array(layer.blobs[0]).ravel()
        var = blob_array(layer.blobs[1]).ravel()
        scale_factor = blob_array(layer.blobs[2]).ravel()[0]
        if scale_factor != 0:
            mean = mean / scale_factor
            var = var / scale_factor
        eps = layer.batch_norm_param.eps or 1e-5
        inv = 1.0 / np.sqrt(var + eps)
        self.blobs[layer.top[0]] = L.Lambda(
            lambda z, m=mean, i=inv: (z - m) * i, name=layer.name.replace('/', '_')
        )(x)

    def _op_scale(self, layer):
        if self.skip_bn:
            self.blobs[layer.top[0]] = self.blobs[layer.bottom[0]]
            return
        x = self.blobs[layer.bottom[0]]
        gamma = blob_array(layer.blobs[0]).ravel()
        beta = blob_array(layer.blobs[1]).ravel() if len(layer.blobs) > 1 else np.zeros_like(gamma)
        self.blobs[layer.top[0]] = L.Lambda(
            lambda z, g=gamma, b=beta: z * g + b, name=layer.name.replace('/', '_')
        )(x)

    def _op_relu(self, layer):
        x = self.blobs[layer.bottom[0]]
        slope = layer.relu_param.negative_slope
        name = layer.name.replace('/', '_')
        if slope:
            self.blobs[layer.top[0]] = L.LeakyReLU(negative_slope=float(slope), name=name)(x)
        else:
            self.blobs[layer.top[0]] = L.ReLU(name=name)(x)

    def _op_convolution(self, layer):
        x = self.blobs[layer.bottom[0]]
        p = layer.convolution_param
        w = blob_array(layer.blobs[0])           # (out, in/g, kh, kw)
        bias = blob_array(layer.blobs[1]).ravel() if p.bias_term else None
        stride = first(p.stride, 1)
        dil = first(p.dilation, 1)
        pad = first(p.pad, 0)
        num_out = p.num_output
        group = p.group or 1
        name = layer.name.replace('/', '_')

        k = w.shape[2]
        # Caffe pads symmetrically and then convolves with valid padding; TF 'same'
        # pads differently when stride > 1, so the padding is applied explicitly
        if pad:
            x = L.ZeroPadding2D(padding=pad, name=name + '_pad')(x)
        padding = 'valid'

        if group == num_out and group == w.shape[0] and w.shape[1] == 1 and group > 1:
            # depthwise: (out,1,kh,kw) → (kh,kw,out,1)
            kernel = expand_dilation(np.transpose(w, (2, 3, 0, 1)), dil)
            lyr = L.DepthwiseConv2D(
                kernel_size=kernel.shape[:2], strides=stride, padding=padding,
                use_bias=bias is not None, name=name)
            y = lyr(x)
            lyr.set_weights([kernel] + ([bias] if bias is not None else []))
        else:
            assert group == 1, f'{layer.name}: group={group} is not supported'
            kernel = expand_dilation(np.transpose(w, (2, 3, 1, 0)), dil)
            lyr = L.Conv2D(
                filters=num_out, kernel_size=kernel.shape[:2], strides=stride,
                padding=padding, use_bias=bias is not None, name=name)
            y = lyr(x)
            lyr.set_weights([kernel] + ([bias] if bias is not None else []))
        self.blobs[layer.top[0]] = y

    def _op_deconvolution(self, layer):
        x = self.blobs[layer.bottom[0]]
        p = layer.convolution_param
        w = blob_array(layer.blobs[0])           # (in, out/g, kh, kw)
        bias = blob_array(layer.blobs[1]).ravel() if p.bias_term else None
        stride = first(p.stride, 1)
        pad = first(p.pad, 0)
        num_out = p.num_output
        group = p.group or 1
        name = layer.name.replace('/', '_')
        k = w.shape[2]
        padding = 'same' if pad else 'valid'

        if group > 1:
            # Keras Conv2DTranspose has no group support, so an equivalent dense
            # block-diagonal kernel is used instead
            # Caffe deconv weight layout is (in, out/g, kh, kw); when grouped,
            # in == out == group
            dense = np.zeros((k, k, num_out, num_out), dtype=np.float32)
            for c in range(num_out):
                dense[:, :, c, c] = np.transpose(w[c, 0], (0, 1))
            kernel = dense
        else:
            # (in,out,kh,kw) -> TF Conv2DTranspose expects (kh,kw,out,in)
            kernel = np.transpose(w, (2, 3, 1, 0))

        # caffe: out = stride*(H-1) + k - 2*pad
        # For a TF 'valid' transposed convolution out = stride*(H-1) + k, so
        # cropping pad on all sides gives the equivalent result
        lyr = L.Conv2DTranspose(
            filters=num_out, kernel_size=(k, k), strides=stride, padding='valid',
            use_bias=bias is not None, name=name)
        y = lyr(x)
        lyr.set_weights([kernel] + ([bias] if bias is not None else []))
        if pad:
            y = L.Lambda(lambda t, p=pad: t[:, p:-p, p:-p, :], name=name + '_crop')(y)
        self.blobs[layer.top[0]] = y

    def _op_pooling(self, layer):
        x = self.blobs[layer.bottom[0]]
        p = layer.pooling_param
        assert p.pool == 0, 'only MAX pooling'
        assert p.pad == 0
        name = layer.name.replace('/', '_')
        # CEIL mode: pad one -inf column and row on the right and bottom, then pool
        # with valid padding, which matches how Caffe clips the window
        pad_val = -3.4e38
        x = L.Lambda(
            lambda t, v=pad_val: tf.pad(t, [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=v),
            name=name + '_ceilpad')(x)
        self.blobs[layer.top[0]] = L.MaxPool2D(
            pool_size=p.kernel_size, strides=p.stride, padding='valid', name=name)(x)

    def _op_eltwise(self, layer):
        assert layer.eltwise_param.operation == 1, 'only SUM'
        xs = [self.blobs[b] for b in layer.bottom]
        self.blobs[layer.top[0]] = L.Add(name=layer.name.replace('/', '_'))(xs)

    def _op_concat(self, layer):
        if any(b not in self.blobs for b in layer.bottom):
            return          # mbox_priorbox: prior boxes are generated on the Rust side,
                            # so there is no corresponding tensor here
        xs = [self.blobs[b] for b in layer.bottom]
        axis = layer.concat_param.axis
        name = layer.name.replace('/', '_')
        if axis == 1 and len(xs[0].shape) == 4:
            # Caffe channel axis -> last axis in NHWC
            self.blobs[layer.top[0]] = L.Concatenate(axis=-1, name=name)(xs)
        else:
            self.blobs[layer.top[0]] = L.Concatenate(axis=axis, name=name)(xs)

    def _op_permute(self, layer):
        # PermuteParameter from the SSD branch is absent from the BVLC caffe.proto,
        # so its parameters cannot be read. Every Permute in this prototxt has order
        # (0,2,3,1), i.e. NCHW -> NHWC, and the tensors here are already NHWC, so
        # this is the identity.
        self.blobs[layer.top[0]] = self.blobs[layer.bottom[0]]

    def _op_flatten(self, layer):
        x = self.blobs[layer.bottom[0]]
        assert layer.flatten_param.axis == 1
        self.blobs[layer.top[0]] = L.Flatten(name=layer.name.replace('/', '_'))(x)

    def _op_reshape(self, layer):
        x = self.blobs[layer.bottom[0]]
        dims = list(layer.reshape_param.shape.dim)
        assert dims == [0, -1, 2], f'unexpected reshape {dims}'
        self.blobs[layer.top[0]] = L.Reshape((-1, 2), name=layer.name.replace('/', '_'))(x)

    def _op_softmax(self, layer):
        x = self.blobs[layer.bottom[0]]
        self.blobs[layer.top[0]] = L.Softmax(
            axis=layer.softmax_param.axis, name=layer.name.replace('/', '_'))(x)

    def _op_crop(self, layer):
        x = self.blobs[layer.bottom[0]]
        ref = self.blobs[layer.bottom[1]]
        assert layer.crop_param.axis == 2 and not list(layer.crop_param.offset)
        name = layer.name.replace('/', '_')
        self.blobs[layer.top[0]] = L.Lambda(
            lambda t: t[0][:, : tf.shape(t[1])[1], : tf.shape(t[1])[2], :], name=name
        )([x, ref])

    def _op_priorbox(self, layer):
        pass          # prior boxes are generated on the Rust side

    def _op_detectionoutput(self, layer):
        pass          # decoding and NMS are done on the Rust side
