import cv2, glob, json, os, sys
# wechat_qrcode in OpenCV 5 accepts only ONNX; run the Caffe models under 4.10
d = cv2.wechat_qrcode_WeChatQRCode('models/detect.prototxt','models/detect.caffemodel','models/sr.prototxt','models/sr.caffemodel')
out={}
for f in sorted(glob.glob(sys.argv[1]+'/*.png')):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    res, pts = d.detectAndDecode(img)
    out[os.path.basename(f)] = {"texts": list(res), "points": [p.reshape(-1,2).tolist() for p in pts]}
json.dump(out, open(sys.argv[2],'w'), ensure_ascii=False)
print("cpp+nn done", len(out), "hits", sum(1 for v in out.values() if v["texts"]))
