"""Uniform-density closed-triangle solid integration; SI units throughout."""
import numpy as np

def solid_properties(vertices, faces):
    ref=np.mean(vertices,axis=0)
    tri=vertices[faces]-ref
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    dv=np.einsum('ij,ij->i',a,np.cross(b,c))/6
    total=dv.sum()
    if abs(total)<1e-15:raise ValueError('Zero solid volume')
    dv*=np.sign(total);volume=abs(total)
    s=a+b+c
    centroid=np.einsum('i,ij->j',dv,s)/(4*volume)
    second=np.einsum('i,ij,ik->jk',dv,s,s)
    for vertex in [a,b,c]:second+=np.einsum('i,ij,ik->jk',dv,vertex,vertex)
    second=second/20-volume*np.outer(centroid,centroid)
    inertia=np.trace(second)*np.eye(3)-second
    inertia=(inertia+inertia.T)/2
    if np.linalg.eigvalsh(inertia).min()<=0:raise ValueError('Invalid inertia')
    return volume,centroid+ref,inertia

def aggregate(items):
    mass=sum(x[0] for x in items)
    center=sum(m*c for m,c,i in items)/mass
    inertia=sum(i+m*((c-center)@(c-center)*np.eye(3)-np.outer(c-center,c-center)) for m,c,i in items)
    return mass,center,inertia

def calculate(parts,groups,p,hinge_pos):
    categories={'link_and_cuff_l':[], 'link_and_cuff_r':[], 'remaining_hardware':[]}
    raw={name:solid_properties(v,f) for name,(v,f) in parts.items()}
    for group,names in groups.items():
        for name in names:
            measured=group.startswith('link_') and not name.startswith('hinge_')
            categories['link_and_cuff_'+group[-1] if measured else 'remaining_hardware'].append(name)
    added_mass=p.get('housing_added_mass',0.)
    if added_mass < 0:raise ValueError('housing_added_mass must be nonnegative')
    targets={'link_and_cuff_l':p['link_cuff_mass'],'link_and_cuff_r':p['link_cuff_mass'],
             'remaining_hardware':(p['total_mass']-added_mass)-2*p['link_cuff_mass']}
    category_report={};per_part={};numeric={}
    for cat,names in categories.items():
        volume=sum(raw[n][0] for n in names);density=targets[cat]/volume
        category_report[cat]={'mass_kg':targets[cat],'modeled_volume_m3':volume,'effective_density_kg_m3':density,'parts':sorted(names)}
        for n in names:
            v,c,iv=raw[n];mass=v*density;inertia=iv*density
            numeric[n]=(mass,c,inertia)
            per_part[n]={'category':cat,'mass_kg':mass,'volume_m3':v,'center_mesh_frame_m':c.tolist(),'inertia_at_center_kg_m2':inertia.tolist()}
    if added_mass:
        # Distribute the added payload over the main housing envelope only.
        # Preserve every original component's mass and inertia contribution.
        v,c,iv=raw['battery'];extra_inertia=iv*(added_mass/v)
        mass,_,inertia=numeric['battery']
        numeric['battery']=(mass+added_mass,c,inertia+extra_inertia)
        per_part['battery']['mass_kg']=mass+added_mass
        per_part['battery']['added_housing_mass_kg']=added_mass
        per_part['battery']['inertia_at_center_kg_m2']=(inertia+extra_inertia).tolist()
        category_report['added_housing_payload']={'mass_kg':added_mass,'parts':['battery'],
            'distribution':'Uniform over main housing envelope; original hardware mass distribution preserved.'}
    bodies={};assembled=[]
    for group,names in groups.items():
        mass,c,inertia=aggregate([numeric[n] for n in names])
        side=1 if group.endswith('r') else -1
        local=c-hinge_pos(side) if group.startswith('link_') else c
        bodies[group]={'mass_kg':mass,'center_body_frame_m':local.tolist(),'inertia_at_center_body_axes_kg_m2':inertia.tolist()}
        offset=np.zeros(3) if group=='waist' else np.array([0,0,side*p['hip_half_width']])
        assembled.append((mass,c+offset,inertia))
    mass,c,inertia=aggregate(assembled)
    assert abs(mass-p['total_mass'])<1e-10
    for item in bodies.values():
        eig=np.linalg.eigvalsh(item['inertia_at_center_body_axes_kg_m2'])
        assert eig[0]>0 and eig[-1]<=eig[0]+eig[1]+1e-12
    report={'method':'Signed tetrahedral integration of closed component meshes, followed by parallel-axis aggregation. Overlapping component volumes are summed; modeled shells and motors are equivalent solid envelopes.',
            'units':{'mass':'kg','length':'m','inertia':'kg*m^2'},
            'mass_constraints':category_report,'bodies':bodies,'parts':per_part,
            'assembled_neutral_pose':{'mass_kg':mass,'center_world_frame_m':c.tolist(),'inertia_at_center_world_axes_kg_m2':inertia.tolist()},
            'assumptions':['Uniform density within each measured mass category, not one density shared across the entire machine.','No measured shell thickness, internal motor or battery construction, or individual component masses.','Body inertia is about its own center of mass in body coordinate axes; link body origin is the passive hinge.','Measured 150 g category excludes hinge knuckles, tongues, and their fasteners.','Collision contacts remain disabled; actuator/friction/damping data remain provisional.']}
    return bodies,report
