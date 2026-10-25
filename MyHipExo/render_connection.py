from pathlib import Path
import struct,json,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
p=Path(__file__).parent
root=ET.parse(p/'preview.xml');paths={e.get('name'):e.get('file') for e in root.findall('./asset/mesh')}
triangles=[]
for geom in root.find("./worldbody/body[@name='waist']").findall('geom'):
    name=geom.get('name');data=(p/paths[geom.get('mesh')]).read_bytes()
    # Right side orthographic view of the actual exported STL meshes.
    for i in range(struct.unpack_from('<I',data,80)[0]):
        rec=struct.unpack_from('<12fH',data,84+50*i);tri=np.array(rec[3:12]).reshape(3,3)
        n=np.array(rec[:3]);light=np.array([-.3,.6,.7]);light/=np.linalg.norm(light)
        factor=.6+.65*abs(np.dot(n,light))
        base=np.array([.22,.25,.28]) if 'belt' not in name else np.array([.08,.095,.11])
        if 'bolt' in name:base=np.array([.48,.5,.52])
        triangles.append((np.mean(tri[:,2]),tri[:,:2],np.clip(base*factor,0,1)))
triangles.sort(key=lambda t:t[0])
fig,ax=plt.subplots(figsize=(10,7));fig.patch.set_facecolor('#edf2f6');ax.set_facecolor('#edf2f6')
ax.add_collection(PolyCollection([t[1] for t in triangles],facecolors=[t[2] for t in triangles],edgecolors='none'))
points=np.concatenate([t[1] for t in triangles]);lo=points.min(axis=0);hi=points.max(axis=0)
ax.set(xlim=(lo[0]-.02,hi[0]+.02),ylim=(lo[1]-.045,hi[1]+.04),aspect='equal');ax.axis('off')
params=json.loads((p/'parameters.json').read_text());hx=params['housing_center_x'];hy=params['housing_top_y']
ax.annotate('Upper housing connection',xy=(hx+.035,hy-.020),xytext=(lo[0],hi[1]+.025),arrowprops=dict(arrowstyle='->',color='#aa5528'),color='#29343e')
ax.annotate('Contoured housing',xy=(hx+.025,hy-.100),xytext=(lo[0],lo[1]-.032),arrowprops=dict(arrowstyle='->',color='#aa5528'),color='#29343e')
fig.tight_layout();fig.savefig(p/'connection_side.png',dpi=150)
