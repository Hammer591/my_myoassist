from pathlib import Path
import json,struct,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
p=Path(__file__).parent;q=json.loads((p/'parameters.json').read_text())
data=(p/'mesh/waist_rail_r.stl').read_bytes()
tri=np.array([struct.unpack_from('<12fH',data,84+50*i)[3:12] for i in range(struct.unpack_from('<I',data,80)[0])]).reshape(-1,3,3)
# STL triangle order retains sweep rings; collect the cross-section points.
points=tri.reshape(-1,3);out=np.array([np.cos(np.deg2rad(30)),-np.sin(np.deg2rad(30)),0])
def project(v):return np.stack([v[...,2],-v@out],axis=-1)*1000
# Lines are recovered from the first, middle, and last straight segments.
def ring(k):
    return np.mean(tri[k*16:k*16+16].reshape(-1,3),axis=0)
# Direct control corners derived from the rounded mesh straight tangents.
centers=[]
for k in range(67):
    faces=tri[k*16:k*16+16]
    # First vertex of each side's first triangle is the current ring vertex.
    centers.append(faces[::2,0,:].mean(axis=0))
centers.append(tri[66*16:67*16:2,2,:].mean(axis=0))
c=project(np.array(centers))
def intersection(a,u,b,v):
    t=np.linalg.solve(np.column_stack([u,-v]),b-a)[0];return a+t*u
b1=intersection(c[0],c[1]-c[0],c[33],c[34]-c[33])
b2=intersection(c[33],c[34]-c[33],c[-2],c[-1]-c[-2])
fig,ax=plt.subplots(figsize=(7,7));ax.plot(c[:,0],c[:,1],lw=14,color='#94a7b3',solid_capstyle='round')
ax.plot(c[:,0],c[:,1],lw=1.2,color='#243d4c')
for b,a1,a2 in [(b1,180,315),(b2,135,270)]:
    ax.add_patch(Arc(b,36,36,theta1=a1,theta2=a2,color='#ba5a27',lw=2))
    angle=np.deg2rad((a1+a2)/2);label=b+27*np.array([np.cos(angle),np.sin(angle)])
    ax.text(*label,'135°',ha='center',va='center',fontsize=14,color='#974718')
ax.set_aspect('equal');ax.axis('off');ax.set_title('Connector bend plane · inner angles',fontsize=15)
ax.margins(.24);fig.tight_layout();fig.savefig(p/'bend_angles.png',dpi=160)
