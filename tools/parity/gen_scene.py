"""Generate small QR codes within large scenes, the case the CNN detector targets."""
import segno, numpy as np, os, random
from PIL import Image, ImageFilter
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'scenes')
os.makedirs(out, exist_ok=True)
rng=random.Random(11); nprng=np.random.default_rng(11)
manifest={}
for i in range(24):
    payload=f"https://wxscan.test/{i:03d}/"+''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(rng.randint(4,20)))
    q=segno.make(payload, error='m', micro=False)
    q.save(f"{out}/tmp.png", scale=rng.choice([2,3,4]), border=3)
    code=Image.open(f"{out}/tmp.png").convert('L')
    W,H=rng.choice([(1280,960),(960,1280),(1920,1080)])
    # Background: low-frequency noise approximating a real scene
    bg=nprng.integers(60,200,(H//16+1,W//16+1)).astype(np.uint8)
    bg=Image.fromarray(bg).resize((W,H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))
    scene=bg.copy()
    ang=rng.uniform(-20,20)
    c=code.rotate(ang, expand=True, fillcolor=255, resample=Image.BICUBIC)
    x=rng.randint(0, max(1,W-c.width-1)); y=rng.randint(0, max(1,H-c.height-1))
    scene.paste(c,(x,y))
    a=np.array(scene,dtype=np.float32)+nprng.normal(0,4,(H,W))
    Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(f"{out}/scene_{i:02d}.png")
    manifest[f"scene_{i:02d}.png"]=payload
os.remove(f"{out}/tmp.png")
import json; json.dump(manifest, open(f"{out}/manifest.json","w"))
print("scenes:", len(manifest))
