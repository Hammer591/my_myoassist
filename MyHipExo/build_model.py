"""Photo-based approximate exoskeleton. Units m; x anterior, y up, z right."""
from pathlib import Path
import json, struct, math, xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).parent
MESH=ROOT/'mesh'; MESH.mkdir(exist_ok=True)
P=dict(hip_half_width=.165, motor_radius=.040, motor_thickness=.050,
       thigh_length=.290, battery_width=.205, battery_height=.110,
       battery_depth=.060, housing_top_y=.120, housing_center_x=-.173205,
       total_mass=3.15, link_cuff_mass=.150, housing_added_mass=.600,
       link_twist_deg=90, link_thickness=.006, link_width_root=.028,
       link_width_tip=.047, side_beam_height=.028, side_beam_depth=.022,
       first_bend_spacing=.280, second_segment_length=.065,
       bend_to_motor_length=.140, beam_slope_deg=30,
       housing_max_plan_depth=.070, housing_upper_offset=.010,
       housing_center_plan_depth=.055, housing_bottom_chamfer=.010,
       surface_link_length=.300, cuff_height=.080, cuff_radius=.075,
       cuff_radius_estimate_min=.070, cuff_radius_estimate_max=.080,
       link_contact_radius=.075, link_root_drop=.036,
       passive_hinge_y=-.054, passive_limit_deg=90, passive_damping=.05,
       hinge_lateral_offset=-.012, link_root_forward_offset=0.0,
       corner_inner_angle_deg=135)
# Equal 135-degree inner angles bisect the 90-degree change from the
# lateral inlet to the downward-pitched sagittal outlet. Motor spacing is
# derived from these angles and measured lengths, not fixed at old 330 mm.
pitch=math.radians(P['beam_slope_deg'])
outlet=np.array([math.cos(pitch),-math.sin(pitch),0.])
inlet=np.array([0.,0.,1.])
middle_direction=(inlet+outlet)/math.sqrt(2)
def routed_path(unknowns):
    first_z,middle,last=unknowns
    P['hip_half_width']=float(first_z+middle*middle_direction[2])
    b2=np.array([0,0,P['hip_half_width']])-last*outlet
    b1=b2-middle*middle_direction
    port=b1.copy();port[2]=P['battery_width']/2-.003
    controls=np.array([port,b1,b2,[0,0,P['hip_half_width']]])
    path=[port];marks=[]
    for j in [1,2]:
        a=controls[j]+.006*(controls[j-1]-controls[j])/np.linalg.norm(controls[j-1]-controls[j])
        c=controls[j]+.006*(controls[j+1]-controls[j])/np.linalg.norm(controls[j+1]-controls[j])
        path.append(a)
        for i,t in enumerate(np.linspace(0,1,33)[1:],1):
            path.append((1-t)**2*a+2*t*(1-t)*controls[j]+t*t*c)
            if i==16:marks.append(len(path)-1)
    path.append(controls[-1])
    return np.array(path),marks,b1,b2
def route_errors(unknowns):
    path,(a,b),_,_=routed_path(unknowns)
    return np.array([2*path[a,2]-P['first_bend_spacing'],np.linalg.norm(np.diff(path[a:b+1],axis=0),axis=1).sum()-P['second_segment_length'],np.linalg.norm(np.diff(path[b:],axis=0),axis=1).sum()-P['bend_to_motor_length']])
unknowns=np.array([P['first_bend_spacing']/2,P['second_segment_length'],P['bend_to_motor_length']])
for _ in range(10):
    residual=route_errors(unknowns)
    if np.max(abs(residual))<1e-12:break
    jac=np.column_stack([(route_errors(unknowns+np.eye(3)[i]*1e-6)-residual)/1e-6 for i in range(3)])
    unknowns-=np.linalg.solve(jac,residual)
ROUTE,BEND_MARKS,B1,B2=routed_path(unknowns)
P['housing_center_x']=float(B1[0]-.035)
P['housing_top_y']=float(B1[1]+.020)
parts={}
def surface(name,rings):
    r=np.array(rings); n,k,_=r.shape; v=r.reshape(-1,3); f=[]
    for i in range(n-1):
        for j in range(k):
            a=i*k+j;b=i*k+(j+1)%k;c=b+k;d=a+k
            f.extend([[a,b,c],[a,c,d]])
    for j in range(1,k-1): f.extend([[0,j+1,j],[(n-1)*k,(n-1)*k+j,(n-1)*k+j+1]])
    parts[name]=(v,np.array(f))
def box(name,c,s):
    c=np.array(c);s=np.array(s)/2
    surface(name,[[c+[-s[0],y,-s[2]],c+[s[0],y,-s[2]],c+[s[0],y,s[2]],c+[-s[0],y,s[2]]] for y in [-s[1],s[1]]])
def cylinder(name,c,r,h):
    surface(name,[[np.array(c)+[r*math.cos(t),r*math.sin(t),z] for t in np.linspace(0,2*math.pi,65)[:-1]] for z in [-h/2,h/2]])

def cylinder_x(name,c,r,h):
    cylinder(name,[0,0,0],r,h)
    v,f=parts[name];parts[name]=(v[:,[2,1,0]]+np.array(c),f[:,::-1])

def hinge_pos(side):
    return np.array([P['link_root_forward_offset'],P['passive_hinge_y'],side*P['hinge_lateral_offset']])
def bar(name,points,w,t):
    surface(name,[[np.array(p)+[dx,0,dz] for dx,dz in [(-w/2,-t/2),(w/2,-t/2),(w/2,t/2),(-w/2,t/2)]] for p in points])
def beam(name,points,height,depth):
    pts=np.array(points);rings=[];h=height/2;d=depth/2;b=.003
    section=[(-h+b,-d),(h-b,-d),(h,-d+b),(h,d-b),(h-b,d),(-h+b,d),(-h,d-b),(-h,-d+b)]
    for i,p in enumerate(pts):
        tangent=pts[min(i+1,len(pts)-1)]-pts[max(i-1,0)]
        tangent/=np.linalg.norm(tangent)
        up=np.array([0.,1.,0.]);up-=tangent*np.dot(up,tangent);up/=np.linalg.norm(up)
        lateral=np.cross(tangent,up)
        rings.append([p+up*a+lateral*b for a,b in section])
    surface(name,rings)
def plate(name,polygon,z,depth):
    surface(name,[[[x,y,zz] for x,y in polygon] for zz in [z-depth/2,z+depth/2]])
def rounded_outline(w,h,r):
    out=[]
    for x,y,a in [(w/2-r,h/2-r,0),(-w/2+r,h/2-r,90),(-w/2+r,-h/2+r,180),(w/2-r,-h/2+r,270)]:
        for angle in np.linspace(a,a+90,9):
            q=math.radians(angle);out.append([x+r*math.cos(q),y+r*math.sin(q)])
    return np.array(out)
def housing():
    # Side profile follows the supplied drawing. Top view has a concave inner face.
    height=P['battery_height'];offset=P['housing_upper_offset'];bottom=P['housing_top_y']-height
    slope_top=.080  # unmeasured shoulder height, follows drawing proportions
    stations=sorted(set([0,.010,.060,slope_top,height]))
    def rear(y):return np.interp(y,[0,.010,.060,height],[.009,0,0,offset])
    def front(y):return np.interp(y,[0,.010,slope_top,height],[.050,.060,.060+offset,.060+offset])
    sag=P['housing_max_plan_depth']-P['housing_center_plan_depth']
    zz=np.linspace(-P['battery_width']/2,P['battery_width']/2,49);rings=[]
    for y in stations:
        back=[[P['housing_center_x']+rear(y),bottom+y,z] for z in zz]
        face=[[P['housing_center_x']+front(y)-sag*(1-(z/(P['battery_width']/2))**2),bottom+y,z] for z in zz[::-1]]
        rings.append(back+face)
    surface('battery',rings)
    # Strip caps preserve the concave outline without convex fan fill-in.
    v,f=parts['battery'];k=len(rings[0]);n=len(zz);f=f[:-2*(k-2)].tolist()
    for level,reverse in [(0,True),(len(rings)-1,False)]:
        o=level*k
        for j in range(n-1):
            a=o+j;b=a+1;c=o+k-2-j;d=o+k-1-j
            tris=[[a,b,c],[a,c,d]]
            f.extend([t[::-1] if reverse else t for t in tris])
    parts['battery']=(v,np.array(f))
    for name,cy,h,w in [('battery_cover',.085,.035,.185),('housing_lower_panel',.033,.039,.180)]:
        outline=rounded_outline(h,w,.006)
        surface(name,[[[P['housing_center_x']+rear(cy+y)+d,bottom+cy+y,z] for y,z in outline] for d in [0,.001]])
def frame(name,c,w,h,t):
    # Rounded rectangular strap slot frame, through hole preserved visually.
    pts=[]
    for cx,cy,a in [(w/2-.012,h/2-.012,0),(-w/2+.012,h/2-.012,90),(-w/2+.012,-h/2+.012,180),(w/2-.012,-h/2+.012,270)]:
        for ang in np.linspace(a,a+90,9):
            q=math.radians(ang);pts.append(np.array(c)+[cx+.012*math.cos(q),cy+.012*math.sin(q),0])
    # Closed sweep with cross-sections normal to the perimeter.
    dense=[]
    for i,p in enumerate(pts):
        nxt=pts[(i+1)%len(pts)]
        for u in np.linspace(0,1,max(2,int(np.linalg.norm(nxt-p)/.006)+1),endpoint=False):dense.append(p+(nxt-p)*u)
    pts=np.array(dense);rings=[]
    for i,p in enumerate(pts):
        tangent=pts[(i+1)%len(pts)]-pts[(i-1)%len(pts)]
        tangent/=np.linalg.norm(tangent)
        outward=np.array([tangent[1],-tangent[0],0])
        rings.append([p+outward*dx+np.array([0,0,dz]) for dx,dz in [(-.004,-t/2),(.004,-t/2),(.004,t/2),(-.004,t/2)]])
    surface(name,rings+[rings[0]])
    v,f=parts[name];parts[name]=(v,f[:-4])  # periodic tube has no end caps

def cuff_panel(name,c,dimensions):
    c=np.array(c);w,h,t=dimensions
    surface(name,[[c+[x,y,z] for y,z in [(-h/2,-t/2),(h/2,-t/2),(h/2,t/2),(-h/2,t/2)]] for x in np.linspace(-w/2,w/2,17)])

def smooth(u):
    u=np.clip(u,0,1);return u*u*(3-2*u)
def link_center(u,side):
    return link_point(u,side,0,0)

def link_phase(u):
    q=np.clip((u-.04)/.78,0,1)
    angle=math.radians(P['link_twist_deg'])
    return angle*q*q*(3-2*q),angle*6*q*(1-q)/.78

def link_axis(side):
    return np.array([P['link_root_forward_offset'],0,side*(P['hinge_lateral_offset']-P['link_contact_radius']-P['link_thickness']/2)])

def link_point(u,side,across,radial):
    # Cut a curved ribbon in the unrolled cylinder, then wrap it onto the
    # cylinder. Across-width vertices also follow circular arcs, not chords.
    r=P['link_contact_radius']+P['link_thickness']/2
    phi,dp=link_phase(u);h=P['thigh_length'];speed=math.hypot(h,r*dp)
    theta=phi+across*h/speed/r
    y=-P['motor_radius']-P['link_root_drop']-h*u+across*r*dp/speed
    return link_axis(side)+np.array([(r+radial)*math.sin(theta),y,side*(r+radial)*math.cos(theta)])
def link_frame(u,side):
    tangent=link_center(u+.0001,side)-link_center(u-.0001,side)
    tangent/=np.linalg.norm(tangent)
    width=link_point(u,side,.00001,0)-link_point(u,side,-.00001,0)
    width/=np.linalg.norm(width)
    normal=np.cross(tangent,width)
    return width,tangent,normal

def surface_trace(length):
    P['thigh_length']=length
    return np.array([link_point(u,1,0,P['link_thickness']/2) for u in np.linspace(0,1,1601)])
lo,hi=.20,.35
for _ in range(35):
    mid=(lo+hi)/2;trace=surface_trace(mid)
    if np.linalg.norm(np.diff(trace,axis=0),axis=1).sum()>P['surface_link_length']:hi=mid
    else:lo=mid
trace=surface_trace((lo+hi)/2)
(ROOT/'parameters.json').write_text(json.dumps(P,indent=2))
groups={'waist':[], 'thigh_r':[], 'thigh_l':[], 'link_r':[], 'link_l':[]}
colors={}
def assign(g,before,color):
    for n in set(parts)-before:groups[g].append(n);colors[n]=color
dark='.055 .065 .075 1'; trim='.12 .14 .16 1';cloth='.025 .029 .032 1';metal='.5 .52 .54 1'
b=set(parts)
housing()
# Wide lateral beams and separate mounting blocks reconstructed from side photo.
for side in [-1,1]:
    s='r' if side==1 else 'l';z=side*P['hip_half_width']
    path=ROUTE*np.array([1,1,side]);port=path[0]
    beam('waist_rail_'+s,path,P['side_beam_height'],P['side_beam_depth'])
    # Rear clevis and visible pivot boss at the main housing.
    px,py=port[:2];pz=side*(P['battery_width']/2)
    plate('housing_ear_'+s,[(px-.014,py-.015),(px+.012,py-.015),(px+.012,py+.022),(px,py+.029),(px-.014,py+.022)],pz,.010)
    box('housing_clamp_'+s,[px,py,pz],[.028,.025,.018])
    cylinder('housing_pivot_'+s,[px,py+.017,pz+side*.008],.008,.008)
    cylinder('housing_pivot_bolt_'+s,[px,py+.017,pz+side*.014],.004,.004)
    # Motor-side sleeve bridges the wide beam into the round motor housing.
    motor_before=set(parts)
    plate('motor_socket_'+s,[[-.080,-.029],[-.020,-.040],[-.014,.004],[-.070,.024],[-.080,.016]],z,.043)
    plate('socket_inset_'+s,[[-.073,-.021],[-.043,-.026],[-.028,-.004],[-.065,.014],[-.073,.008]],z+side*.023,.003)
    # Upper inclined gusset follows the triangular support visible below the belt.
    plate('upper_gusset_'+s,[[-.100,.012],[-.014,-.004],[-.019,.022],[-.075,.068],[-.088,.063]],z-side*.018,.012)
    plate('gusset_inset_'+s,[[-.085,.024],[-.037,.015],[-.076,.054]],z-side*.010,.003)
    for j,(xx,yy) in enumerate([(-.079,.054),(-.027,.017),(-.066,-.013)]):
        cylinder('mount_bolt_'+s+str(j),[xx,yy,z+side*(-.008 if j<2 else .025)],.0035,.004)
    # Scale previous support shapes about the old motor axis; keep axial depth.
    for name in set(parts)-motor_before:
        v,f=parts[name];v=v.copy();v[:,0]*=P['motor_radius']/.061;v[:,1]=(v[:,1]+.045)*P['motor_radius']/.061;parts[name]=(v,f)
    cylinder('motor_'+s,[0,0,z-side*.0025],P['motor_radius'],P['motor_thickness']-.005)
    cylinder('motor_cap_'+s,[0,0,z+side*(P['motor_thickness']/2-.0025)],P['motor_radius']*.86,.005)
    for j in range(10):
        a=j*2*math.pi/10
        cylinder('motor_bolt_'+s+str(j),[P['motor_radius']*.72*math.cos(a),P['motor_radius']*.72*math.sin(a),z+side*(P['motor_thickness']/2-.001)],.0022,.002)
assign('waist',b,dark)
# Hardware only: no waist textile belt, buckle or lumbar pad.
for side in [-1,1]:
    s='r' if side==1 else 'l';h=hinge_pos(side);b=set(parts)
    # Motor output clevis: two fore/aft ears straddle the moving knuckle.
    box('hinge_bridge_'+s,[h[0],-.032,side*P['hinge_lateral_offset']/2],[.038,.016,.040])
    for j,dx in enumerate([-.015,.015]):
        cylinder_x('hinge_ear_'+s+str(j),h+[dx,0,0],.012,.006)
        box('hinge_cheek_'+s+str(j),h+[dx,.013,0],[.006,.026,.024])
    cylinder_x('hinge_pin_'+s,h,.0045,.043)
    for j,dx in enumerate([-.022,.022]):
        cylinder_x('hinge_pin_head_'+s+str(j),h+[dx,0,0],.006,.002)
    assign('thigh_'+s,b,dark)
    for n in groups['thigh_'+s]:
        if 'pin' in n:colors[n]=metal
    g='link_'+s;b=set(parts)
    cylinder_x('hinge_knuckle_'+s,h,.009,.022)
    box('hinge_tongue_'+s,[h[0],-.073,h[2]],[.022,.026,.008])
    for j,dx in enumerate([-.007,.007]):
        cylinder('hinge_tongue_bolt_'+s+str(j),[h[0]+dx,-.080,h[2]+side*.005],.002,.002)
    # A 90-degree circumferential wrap, with a cylindrical contact surface.
    rings=[];ribs=[[],[]]
    for u in np.r_[np.linspace(0,1,121),1.02,1.04]:
        w=P['link_width_root']+(P['link_width_tip']-P['link_width_root'])*smooth(u)
        t=P['link_thickness'];across=np.linspace(-w/2,w/2,11)
        section=[(a,-t/2) for a in across]+[(a,t/2) for a in across[::-1]]
        rings.append([link_point(u,side,a,b) for a,b in section])
        if .12<=u<=.82:
            for k,offset in enumerate([-.006,.006]):
                ribs[k].append([link_point(u,side,offset+a,t/2+.001+b) for a,b in [(-.001,-.001),(.001,-.001),(.001,.001),(-.001,.001)]])
    surface('curved_link_'+s,rings)
    # Cap the annular strip without fan triangles crossing the cylinder.
    v,f=parts['curved_link_'+s];f=f[:-40].tolist()
    for o,rev in [(0,True),((len(rings)-1)*22,False)]:
        for j in range(10):
            for tri in [[o+j,o+j+1,o+20-j],[o+j,o+20-j,o+21-j]]:
                f.append(tri[::-1] if rev else tri)
    parts['curved_link_'+s]=(v,np.array(f))
    for k,r in enumerate(ribs):surface('link_rib_'+s+str(k),r)
    cuff_before=set(parts)
    cuff_panel('distal_plate_'+s,[0,0,0],[.049,P['cuff_height'],.012])
    cuff_panel('distal_cover_'+s,[0,0,side*.008],[.040,.064,.005])
    for x in [-.043,.043]:frame('strap_slot_'+s+str(x),[x,0,0],.045,P['cuff_height']-.008,.008)
    for yy in [-.024,.024]:
        for xx in [-.016,.016]:cylinder('cuff_bolt_'+s+str(xx)+str(yy),[xx,yy,side*.013],.003,.003)
    # Cuff long axis is vertical in the neutral hip pose. Preserve the
    # terminal twist heading, but do not inherit the curved rod's lean.
    width,_,_=link_frame(1,side)
    width[1]=0; width/=np.linalg.norm(width)
    up=np.array([0.,1.,0.]);normal=np.cross(width,up)
    rotation=np.column_stack([width,up,normal])
    origin=link_center(1,side)-np.array([0,P['cuff_height']/2,0])+side*normal*(.006-P['link_thickness']/2)
    for name in set(parts)-cuff_before:
        v,f=parts[name];v=v.copy()
        # Cylindrical saddle wraps around the thigh, concave toward the wearer.
        # R is the inner contact radius; the plate is 12 mm thick.
        radius=P['cuff_radius']+.006;theta=v[:,0]/radius;rho=radius+side*v[:,2]
        v[:,0]=rho*np.sin(theta);v[:,2]=side*(rho*np.cos(theta)-radius)
        parts[name]=(v@rotation.T+origin,f)
    assign(g,b,dark)
# Verify the generated vertices, not only the parameter dictionary.
shell=parts['battery'][0];a,b=BEND_MARKS
beam_centers=parts['waist_rail_r'][0].reshape(-1,8,3).mean(axis=1)
straight_segments=[beam_centers[1]-beam_centers[0],beam_centers[34]-beam_centers[33],beam_centers[-1]-beam_centers[-2]]
straight_segments=[v/np.linalg.norm(v) for v in straight_segments]
corner_angles=[float(math.degrees(math.acos(np.clip(np.dot(-u,v),-1,1)))) for u,v in zip(straight_segments[:-1],straight_segments[1:])]
assert all(abs(a-P['corner_inner_angle_deg'])<1e-7 for a in corner_angles)
right_rings=parts['curved_link_r'][0].reshape(-1,22,3)
measured_trace=right_rings[:121,16,:]
cuffs=np.concatenate([parts[n][0] for n in groups['link_r'] if n.startswith(('distal_','strap_slot_','cuff_bolt_'))])
motor=np.concatenate([v for n,(v,f) in parts.items() if n.startswith(('motor_r','motor_cap_r','motor_bolt_r'))])
mid_shell=shell[np.isclose(shell[:,2],0)]
cw=link_frame(1,1)[0];cw[1]=0;cw/=np.linalg.norm(cw)
cuff_axis=link_axis(1)
cuff_radial=parts['distal_plate_r'][0]-cuff_axis
cuff_inner_radius=float(np.linalg.norm(cuff_radial[:,[0,2]],axis=1).min())
assert abs(cuff_inner_radius-P['cuff_radius'])<1e-8
checks={
 'first_corner_inner_angle_deg':corner_angles[0],
 'second_corner_inner_angle_deg':corner_angles[1],
 'motor_axis_spacing_mm':P['hip_half_width']*2000,
 'root_extension_to_motor_axis_distance_mm':float(abs(right_rings[0,[5,16],0].mean())*1000),
 'hinge_lateral_offset_mm':P['hinge_lateral_offset']*1000,
 'bilateral_hinge_axis_spacing_mm':(P['hip_half_width']+P['hinge_lateral_offset'])*2000,
 'link_contact_radius_max_error_mm':float(np.abs(np.linalg.norm((right_rings[:,:11,:]-link_axis(1))[:,:,[0,2]],axis=2)-P['link_contact_radius']).max()*1000),
 'cuff_inner_contact_radius_mm':cuff_inner_radius*1000,
 'housing_width_mm':float(np.ptp(shell[:,2])*1000),
 'housing_height_mm':float(np.ptp(shell[:,1])*1000),
 'housing_max_plan_depth_mm':float(np.ptp(shell[:,0])*1000),
 'housing_center_plan_depth_mm':float(np.ptp(mid_shell[:,0])*1000),
 'first_bend_spacing_mm':float(beam_centers[a,2]*2000),
 'second_segment_centerline_mm':float(np.linalg.norm(np.diff(beam_centers[a:b+1],axis=0),axis=1).sum()*1000),
 'second_bend_to_axis_centerline_mm':float(np.linalg.norm(np.diff(beam_centers[b:],axis=0),axis=1).sum()*1000),
 'motor_diameter_mm':float(np.ptp(parts['motor_r'][0][:,0])*1000),
 'motor_total_thickness_mm':float(np.ptp(motor[:,2])*1000),
 'cuff_total_height_mm':float(np.ptp(cuffs[:,1])*1000),
 'link_marked_surface_path_mm':float(np.linalg.norm(np.diff(measured_trace,axis=0),axis=1).sum()*1000),
 'last_segment_side_pitch_deg':float(math.degrees(math.atan2(beam_centers[-2,1],-beam_centers[-2,0]))),
 'entry_centerline_vertical':bool(np.allclose(link_frame(1,1)[1],[0,-1,0],atol=1e-8)),
 'entry_width_axis_horizontal':bool(abs(link_frame(1,1)[0][1])<1e-8)
}
targets={'housing_width_mm':205,'housing_height_mm':110,'housing_max_plan_depth_mm':70,'housing_center_plan_depth_mm':55,'first_bend_spacing_mm':280,'second_segment_centerline_mm':65,'second_bend_to_axis_centerline_mm':140,'motor_diameter_mm':80,'motor_total_thickness_mm':50,'cuff_total_height_mm':80,'link_marked_surface_path_mm':300,'last_segment_side_pitch_deg':30}
for key,target in targets.items():assert abs(checks[key]-target)<.05,(key,checks[key],target)
assert checks['entry_centerline_vertical'] and checks['entry_width_axis_horizontal']
assert checks['link_contact_radius_max_error_mm']<.00001
assert checks['root_extension_to_motor_axis_distance_mm']<.00001
(ROOT/'dimension_check.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))

def stl(name,v,f):
    with (MESH/(name+'.stl')).open('wb') as out:
        out.write(b'PhotoHipExo estimated geometry'.ljust(80,b' '));out.write(struct.pack('<I',len(f)))
        for tri in v[f]:
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=max(np.linalg.norm(normal),1e-20)
            out.write(struct.pack('<12fH',*normal,*tri.flatten(),0))
for n,(v,f) in parts.items():stl(n,v,f)
from mass_properties import calculate
body_physics,physics_report=calculate(parts,groups,P,hinge_pos)
(ROOT/'physical_properties.json').write_text(json.dumps(physics_report,indent=2),encoding='utf-8')
fmt=lambda values:' '.join(f'{float(v):.12g}' for v in values)
def make(preview=False):
    root=ET.Element('mujoco',model='PhotoHipExo_L1');ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',gravity='0 -9.81 0',timestep='.002')
    asset=ET.SubElement(root,'asset')
    for n in parts:ET.SubElement(asset,'mesh',name=n,file='mesh/'+n+'.stl')
    world=ET.SubElement(root,'worldbody')
    bodies={}
    for g,names in groups.items():
        parent=world
        pos='0 0 0'
        side=1 if g.endswith('r') else -1
        islink=g.startswith('link_')
        if islink:
            parent=bodies['thigh_'+g[-1]];pos=' '.join(map(str,hinge_pos(side)))
        elif preview and g!='waist':pos=f"0 0 {P['hip_half_width']*side}"
        body=ET.SubElement(parent,'body',name=g,pos=pos)
        bodies[g]=body
        physical=body_physics[g];tensor=np.array(physical['inertia_at_center_body_axes_kg_m2'])
        ET.SubElement(body,'inertial',pos=fmt(physical['center_body_frame_m']),mass=f"{physical['mass_kg']:.12g}",fullinertia=fmt([tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]]))
        if islink:
            limit=math.radians(P['passive_limit_deg'])
            ET.SubElement(body,'joint',name='passive_abduction_'+g[-1],type='hinge',axis=f'{-side} 0 0',limited='true',range=f'{-limit} {limit}',damping=str(P['passive_damping']))
        elif preview and g!='waist':ET.SubElement(body,'joint',name='hip_'+g[-1],axis='0 0 1',range='-0.5 1.6',damping='1')
        for n in sorted(names):
            geo=ET.SubElement(body,'geom',name=n,type='mesh',mesh=n,rgba=colors[n],contype='0',conaffinity='0',group='1')
            if islink:geo.set('pos',' '.join(map(str,-hinge_pos(side))))
        ET.SubElement(body,'site',name=g+'_mount',size='.003',rgba='1 .5 0 1')
    if preview:
        act=ET.SubElement(root,'actuator')
        for s in ['r','l']:ET.SubElement(act,'motor',name='demo_motor_'+s,joint='hip_'+s,gear='1',ctrlrange='-10 10')
    ET.indent(root);ET.ElementTree(root).write(ROOT/('preview.xml' if preview else 'L1model.xml'),encoding='utf-8',xml_declaration=True)
make();make(True)
# Keep the calibrated attachment config. Regenerate only the rigid training variant.
bound = ET.parse(ROOT/'L1model.xml')
for body in bound.iter('body'):
    for joint in list(body.findall('joint')):
        body.remove(joint)
ET.indent(bound.getroot())
bound.write(ROOT/'L1bound.xml', encoding='utf-8', xml_declaration=True)
# Three views, using the exact generated mesh geometry.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig=plt.figure(figsize=(15,7),facecolor='#e7ecf0')
for i,(el,az,title) in enumerate([(18,38,'Perspective'),(0,0,'Side'),(0,90,'Rear')]):
    ax=fig.add_subplot(1,3,i+1,projection='3d');ax.set_facecolor('#e7ecf0')
    for g,names in groups.items():
        offset=np.array([0,0,0]) if g=='waist' else np.array([0,0,P['hip_half_width']*(1 if g.endswith('r') else -1)])
        for n in names:
            v,f=parts[n];q=(v+offset)[:,[2,0,1]]
            ax.add_collection3d(Poly3DCollection(q[f],facecolors='#343e48',edgecolors='none',alpha=1))
    ax.set(xlim=(-.24,.24),ylim=(-.24,.16),zlim=(-.44,.16),title=title);ax.set_box_aspect((.48,.4,.60));ax.view_init(el,az);ax.set_axis_off()
fig.suptitle(f"PhotoHipExo V13 | Total mass {P['total_mass']:.2f} kg | Geometry-derived inertias",fontsize=15)
plt.tight_layout();plt.savefig(ROOT/'preview.png',dpi=160)
print(f'Generated {len(parts)} STL meshes and MJCF/YAML files in {ROOT}')
