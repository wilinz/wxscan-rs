import segno, numpy as np, random, os, string, json
from PIL import Image, ImageFilter
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'corpus')
os.makedirs(out, exist_ok=True)
rng = random.Random(7); nprng = np.random.default_rng(7)
def find_coeffs(pa, pb):
    m=[]
    for p1,p2 in zip(pa,pb):
        m.append([p1[0],p1[1],1,0,0,0,-p2[0]*p1[0],-p2[0]*p1[1]])
        m.append([0,0,0,p1[0],p1[1],1,-p2[1]*p1[0],-p2[1]*p1[1]])
    A=np.matrix(m,dtype=float); B=np.array(pb).reshape(8)
    return np.array(np.dot(np.linalg.inv(A.T*A)*A.T, B)).reshape(8)
payloads=[]
for i in range(40):
    kind=rng.choice(['num','alnum','byte','url','utf8'])
    if kind=='num': p=''.join(rng.choice(string.digits) for _ in range(rng.randint(5,80)))
    elif kind=='alnum': p=''.join(rng.choice(string.ascii_uppercase+string.digits+' $%*+-./:') for _ in range(rng.randint(5,80)))
    elif kind=='byte': p=''.join(rng.choice(string.printable[:94]) for _ in range(rng.randint(5,120)))
    elif kind=='url': p='https://example.com/'+''.join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3,40)))
    else: p='测试'+''.join(rng.choice('中文二维码扫描器识别汉字编码') for _ in range(rng.randint(2,20)))
    payloads.append(p)
n=0; manifest={}
for idx,p in enumerate(payloads):
    for ec in ['l','m','q','h']:
        try: q=segno.make(p, error=ec, micro=False)
        except Exception: continue
        scale=rng.choice([3,4,6,8,10])
        q.save(f"{out}/tmp.png", scale=scale, border=4)
        im=Image.open(f"{out}/tmp.png").convert('L')
        variant=rng.choice(['plain','rot','scale','blur','noise','persp','inv'])
        if variant=='rot': im=im.rotate(rng.uniform(-45,45), expand=True, fillcolor=255, resample=Image.BICUBIC)
        elif variant=='scale':
            f=rng.uniform(0.4,0.9); im=im.resize((max(30,int(im.width*f)),max(30,int(im.height*f))), Image.LANCZOS)
        elif variant=='blur': im=im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5,2.5)))
        elif variant=='noise':
            a=np.array(im,dtype=np.float32)+nprng.normal(0,rng.uniform(8,35),(im.height,im.width))
            im=Image.fromarray(np.clip(a,0,255).astype(np.uint8))
        elif variant=='persp':
            w,h=im.size; k=rng.uniform(0.05,0.25)
            pa=[(0,0),(w,0),(int(w*(1-k)),h),(int(w*k),h)]
            im=im.transform((w,h), Image.PERSPECTIVE, find_coeffs(pa,[(0,0),(w,0),(w,h),(0,h)]), Image.BICUBIC, fillcolor=255)
        elif variant=='inv': im=Image.fromarray(255-np.array(im))
        name=f"{idx:03d}_{ec}_{variant}.png"
        im.save(f"{out}/{name}"); manifest[name]=p; n+=1
os.remove(f"{out}/tmp.png")
json.dump(manifest, open(f"{out}/manifest.json","w"), ensure_ascii=False)
print("generated", n)
