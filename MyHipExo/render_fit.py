from pathlib import Path
import json,struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p=Path(__file__).parent
q=json.loads((p/'parameters.json').read_text())
axis=np.array([q['link_root_forward_offset'],0,q['hinge_lateral_offset']-q['link_contact_radius']-q['link_thickness']/2])
def mesh(name):
    data=(p/'mesh'/(name+'.stl')).read_bytes()
    return np.array([struct.unpack_from('<12fH',data,84+50*i)[3:12] for i in range(struct.unpack_from('<I',data,80)[0])]).reshape(-1,3,3)
fig=plt.figure(figsize=(10,7),facecolor='#eef2f5')
ax=fig.add_subplot(121,projection='3d')
top=-q['motor_radius']-q.get('link_root_drop',0)
phi,y=np.meshgrid(np.linspace(0,2*np.pi,81),np.linspace(top,top-q['thigh_length']-.08,2))
x=axis[0]+.075*np.sin(phi);z=axis[2]+.075*np.cos(phi)
ax.plot_surface(x,z,y,color='#6fa8c2',alpha=.15,linewidth=0,shade=False)
names=['curved_link_r','distal_plate_r','distal_cover_r']+[f.stem for f in (p/'mesh').glob('strap_slot_r*.stl')]
for name in names:
    tri=mesh(name)[:,:,[0,2,1]]
    ax.add_collection3d(Poly3DCollection(tri,facecolors='#263d4b',edgecolors='none'))
ax.set(xlim=(axis[0]-.09,axis[0]+.10),ylim=(axis[2]-.09,axis[2]+.10),zlim=(top-q['thigh_length']-.09,top+.015),title='Link and cuff on the same cylinder')
ax.set_box_aspect((.19,.19,.415));ax.view_init(18,35);ax.set_axis_off()
ax=fig.add_subplot(122)
ax.add_patch(plt.Circle((axis[0]*1000,axis[2]*1000),75,facecolor='#d4e4eb',edgecolor='#6fa8c2'))
tri=mesh('curved_link_r')
ax.scatter(tri[:,:,0].ravel()*1000,tri[:,:,2].ravel()*1000,s=.05,color='#263d4b')
ax.set(aspect='equal',title='Top projection: quarter-cylinder wrap',xlabel='Forward (mm)',ylabel='Lateral (mm)')
ax.annotate('R = 75 mm',xy=(axis[0]*1000,axis[2]*1000),xytext=(-28,-95))
fig.suptitle('V7 | Cylindrical contact surface, not a locally twisted flat strip')
fig.tight_layout();fig.savefig(p/'cylinder_fit.png',dpi=160)
