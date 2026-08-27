import json, sys, numpy as np
cpp=json.load(open(sys.argv[1])); rust=json.load(open(sys.argv[2]))
manifest=json.load(open(sys.argv[3])) if len(sys.argv)>3 else {}
n=len(cpp); same=0; diff=[]; maxd=0.0
cpp_hit=0; rust_hit=0; cpp_correct=0; rust_correct=0
for k in sorted(cpp):
    c=cpp[k]; r=rust.get(k,{"texts":[],"points":[]})
    cpp_hit += 1 if c["texts"] else 0
    rust_hit += 1 if r["texts"] else 0
    exp = manifest.get(k)
    if exp is not None:
        cpp_correct += 1 if (c["texts"] and c["texts"][0]==exp) else 0
        rust_correct += 1 if (r["texts"] and r["texts"][0]==exp) else 0
    if c["texts"]==r["texts"]:
        same+=1
        for cp,rp in zip(c["points"], r["points"]):
            a=np.array(cp,dtype=float).reshape(-1,2); b=np.array(rp,dtype=float).reshape(-1,2)
            if a.shape==b.shape: maxd=max(maxd, float(np.abs(a-b).max()))
    else:
        diff.append((k, c["texts"], r["texts"]))
print(f"images           : {n}")
print(f"text identical   : {same}/{n}")
print(f"max point diff   : {maxd:.4f} px")
print(f"cpp decoded      : {cpp_hit}/{n}   correct: {cpp_correct}/{n}")
print(f"rust decoded     : {rust_hit}/{n}   correct: {rust_correct}/{n}")
for k,c,r in diff[:15]:
    print(" DIFF", k, "\n   cpp :", [x[:40] for x in c], "\n   rust:", [x[:40] for x in r])
