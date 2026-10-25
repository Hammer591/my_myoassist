from pathlib import Path
import os,mujoco,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p=Path(__file__).parent
os.chdir(p)
m=mujoco.MjModel.from_xml_path('preview.xml');d=mujoco.MjData(m)
fig=plt.figure(figsize=(12,7),facecolor='#edf2f6')
for j,angle in enumerate([0,20]):
    d.qpos[:]=0;d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'passive_abduction_r')]]=np.deg2rad(angle)
    mujoco.mj_forward(m,d)
    ax=fig.add_subplot(1,2,j+1,projection='3d')
    for gid in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gid)
        if not ((name.startswith(('hinge_','motor_')) and ('_r' in name)) or name=='curved_link_r'):continue
        mid=m.geom_dataid[gid];va=m.mesh_vertadr[mid];fa=m.mesh_faceadr[mid]
        verts=m.mesh_vert[va:va+m.mesh_vertnum[mid]]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        faces=m.mesh_face[fa:fa+m.mesh_facenum[mid]]
        tri=verts[faces];tri=tri[tri[:,:,1].max(axis=1)>-.115]
        color='#9babad' if 'pin' in name else ('#3c7385' if 'knuckle' in name or 'tongue' in name else '#36444d')
        ax.add_collection3d(Poly3DCollection(tri[:,:,[2,0,1]],facecolors=color,edgecolors='none'))
    ax.set(xlim=(.12,.24),ylim=(-.045,.05),zlim=(-.125,.045),title=f'Passive lateral hinge: {angle} degrees')
    ax.set_box_aspect((.12,.095,.17));ax.view_init(12,48);ax.set_axis_off()
fig.tight_layout();fig.savefig(p/'hinge_detail.png',dpi=160)
