import json,numpy as np,sys
S=sys.argv[1]
sync=json.load(open(S+'/sync.json'))
fine={'WhatsApp Video 2026-10-22 at 14.06.22':0.725,'WhatsApp Video 2026-10-22 at 22.47.08':8.763,'WhatsApp Video 2026-10-22 at 12.40.57':3.288,'WhatsApp Video 2026-10-22 at 12.38.08':7.583,'WhatsApp Video 2026-10-22 at 17.36.26':3.928,'WhatsApp Video 2026-10-22 at 16.20.55':21.858,'WhatsApp Video 2026-10-22 at 12.58.15 (1)':17.685}
def sig(name,off=0):
    r=np.array(json.load(open(f'{S}/met/{name}.json'))['rows'])
    t=r[:,0]+off; e=r[:,3]
    grid=np.arange(-30,220,0.1)
    v=np.interp(grid,t,e,left=np.nan,right=np.nan)
    # high-pass: remove slow trend
    m=~np.isnan(v); x=v.copy(); 
    k=np.convolve(np.nan_to_num(v),np.ones(30)/30,'same'); x=v-k
    idx=np.where(m)[0]; x[:idx[0]+40]=np.nan; x[idx[-1]-40:]=np.nan
    return grid,x
for d in ['DJI_20261022115523_0002_D','DJI_20261022115924_0003_D']:
    _,dx=sig(d,0)  # drone time on grid (grid = drone time)
    for g,o in fine.items():
        _,gx=sig(g,o)  # song time
        best=[]
        for lag in np.arange(-60,60,0.1):
            # song_time = drone_time + lag
            sh=int(round(lag/0.1))
            a=dx; b=np.roll(gx,-sh)
            m=~np.isnan(a)&~np.isnan(b)
            if m.sum()<600: best.append(-1);continue
            best.append(np.corrcoef(a[m],b[m])[0,1])
        best=np.array(best); i=int(np.nanargmax(best)); lags=np.arange(-60,60,0.1)
        srt=np.sort(best)[::-1]
        print(d[-6:-2],g[-8:],f"lag={lags[i]:6.1f} r={best[i]:.3f} median={np.nanmedian(best):.3f}")
print('--- combined')
res={}
for d in ['DJI_20261022115239_0001_D','DJI_20261022115523_0002_D','DJI_20261022115924_0003_D']:
    _,dx=sig(d,0); lags=np.arange(-60,60,0.033); tot=np.zeros(len(lags))
    for g,o in fine.items():
        _,gx=sig(g,o)
        for j,lag in enumerate(lags):
            sh=int(round(lag/0.1)); b=np.roll(gx,-sh); m=~np.isnan(dx)&~np.isnan(b)
            if m.sum()>60: tot[j]+=np.corrcoef(dx[m],b[m])[0,1]
    i=int(np.argmax(tot)); far=tot.copy(); far[max(0,i-60):i+60]=-9
    print(d, f"lag={lags[i]:.2f} sum={tot[i]:.2f} 2nd={far.max():.2f}")
    res[d]=round(float(lags[i]),2)
json.dump(res,open(S+'/dronesync.json','w'))
