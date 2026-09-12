"""Offline geometry audit of Q4 two-ring coverage (no source truth involved)."""
import itertools
import json
import math
from pathlib import Path
import numpy as np


def stations(inner=950, outer=1900, ni=8, no=16):
    return [(0.,0.)]+[(r*math.cos(2*math.pi*k/n),r*math.sin(2*math.pi*k/n))
                       for r,n in ((inner,ni),(outer,no)) for k in range(n)]


def audit(points, step=30):
    # Each square cell lies in a disc of this radius about its centre.
    half = step / math.sqrt(2)
    xy = np.arange(-1800,1800+step/2,step)
    x,y = np.meshgrid(xy,xy)
    cells = np.stack((x.ravel(),y.ravel()),axis=-1)
    cells = cells[np.linalg.norm(cells,axis=1) <= 1800+half]
    v = np.array(points)[None,:,:] - cells[:,None,:]
    d = np.linalg.norm(v,axis=-1)
    # Whole-cell range guarantee and angular uncertainty from moving inside it.
    valid = (d+half <= 1000) & (d > half)
    a = np.arctan2(v[:,:,1],v[:,:,0]) % (2*np.pi)
    a[~valid] = np.inf
    order = np.argsort(a,axis=1)
    a = np.take_along_axis(a,order,axis=1)
    slack = np.arcsin(np.minimum(1., half / np.maximum(d,half)))
    slack = np.take_along_axis(slack,order,axis=1)
    count = valid.sum(axis=1)
    worst = np.zeros(len(cells))
    for k in range(len(points)-1):
        gap = a[:,k+1] - a[:,k] + slack[:,k+1] + slack[:,k]
        worst = np.maximum(worst,np.where(k+1<count,gap,0))
    idx = np.maximum(0,count-1)
    end = a[np.arange(len(a)),idx]
    wrap = a[:,0] + 2*np.pi-end + slack[:,0]+slack[np.arange(len(a)),idx]
    worst = np.maximum(worst,wrap)
    passed = (count>=3)&(worst <= np.pi)
    return dict(cells=len(cells),uncovered_cells=int((~passed).sum()),
                worst_gap_deg=float(np.nanmax(np.where(count>=3,worst,2*np.pi))*180/np.pi))


if __name__ == '__main__':
    rows=[]
    for ni,no,inner,outer in itertools.product((6,8,10),(12,16,20),(850,1000,1150),(1850,1950,2050)):
        pts=stations(inner,outer,ni,no)
        result=audit(pts,step=50)
        row=dict(ni=ni,no=no,inner=inner,outer=outer,**result)
        rows.append(row)
    rows.sort(key=lambda r:(r['uncovered_cells'],r['ni']+r['no'],r['inner']+r['outer']))
    print(json.dumps(rows[:20],indent=2))
    Path('results/q4_450_station_design.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
