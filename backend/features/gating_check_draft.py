"""开发阶段端到端门控检验。

用 dev 阶段（训练 r5–r6、早停 r7、健康分标定 r5）的 ConcatMLP-mask，在 r1–r3 上比较
"四条等权"与"按健康分加权"的融合定位误差。
运行（需先 link_health d1 --calib r5，并训练 dev 阶段 ConcatMLP-mask 三个种子）：
    python -X utf8 -m backend.features.gating_check_draft 42 1 2
"[诊断]固定去某条链路"不是方法，只用来看"有没有可改进空间"，不进论文主线。
尚未整理成正式脚本；正式版要把健康分方案在 D+3 之前定死后再写。
"""
import sys, numpy as np, torch
from backend import config
from backend.data import sessions
from backend.training import train_d1
grid=sessions.session_grid("d1")
h=np.load(config.DATASET_ARTIFACTS_DIR/"d1"/"link_health.npz")
assert [str(c) for c in h["calib_rounds"]]==["r5"]
data,_=train_d1.load("dev","ampphase")
X,yxy,lab,bat=data["test"]
cal=np.isin(h["batch"],["r5"])
fp_ratio=h["raw_fp_dist"]/np.median(h["raw_fp_dist"][cal],axis=0,keepdims=True)
def health(name,r):
    m=h["batch"]==r
    assert np.array_equal(h["label"][m],lab[bat==r])
    if name=="q_mean": return h["q_mean"][m]
    if name=="q_max": return h["q_max"][m]
    if name=="fp": return 1-h["ind_fp"][m]
    if name=="fp_ratio": return 1/fp_ratio[m]
def run(seed):
    model,_=train_d1.make_model("ConcatMLP-mask",seed,grid,"cls",2)
    model.load(train_d1.out_dir("dev",f"cls_ampphase_seed{seed}")/"ConcatMLP-mask"); net=model.net.eval()
    res={}
    for r in ["r1","r2","r3"]:
        m=bat==r; Xt=torch.tensor(X[m]); yt=yxy[m]
        def err(w):
            with torch.no_grad(): lg=net(Xt,link_weights=torch.tensor(w,dtype=torch.float32)).numpy()
            l=lg.argmax(1); p=np.stack([l%5,l//5],1)
            e=np.linalg.norm(p-yt,axis=1)*grid.spacing_m; return e.mean()
        n=m.sum(); out={"等权": err(np.ones((n,4)))}
        for hn in ["q_mean","q_max","fp","fp_ratio"]:
            q=health(hn,r); w=q/np.maximum(q.max(1,keepdims=True),1e-6)
            out[f"软:{hn}"]=err(w)
            hard=np.ones((n,4)); hard[np.arange(n),q.argmin(1)]=0
            out[f"硬去最差:{hn}"]=err(hard)
        for i,l in enumerate(config.all_link_ids()):           # 诊断：固定去掉某一条（不是方法，只看"有没有可改进空间"）
            w=np.ones((n,4)); w[:,i]=0; out[f"[诊断]固定去{l}"]=err(w)
        res[r]=out
    return res
seeds=[int(s) for s in sys.argv[1:]]
allr=[run(s) for s in seeds]
keys=list(allr[0]["r1"].keys())
print(f"{'权重方式':24s}"+"".join(f"{r:>8s}" for r in ["r1","r2","r3"])+f"{'平均':>8s}")
for k in keys:
    v=[np.mean([a[r][k] for a in allr]) for r in ["r1","r2","r3"]]
    print(f"{k:24s}"+"".join(f"{x:>8.3f}" for x in v)+f"{np.mean(v):>8.3f}")
