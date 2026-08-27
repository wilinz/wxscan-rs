import cv2, glob, json, os, sys
d = cv2.wechat_qrcode_WeChatQRCode()
out={}
for f in sorted(glob.glob(sys.argv[1]+'/*.png')):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    res, pts = d.detectAndDecode(img)
    out[os.path.basename(f)] = {"texts": list(res), "points": [p.reshape(-1,2).tolist() for p in pts]}
json.dump(out, open(sys.argv[2],'w'), ensure_ascii=False)
print("cpp done", len(out))
