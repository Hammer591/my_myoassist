from pathlib import Path
import struct,json,xml.etree.ElementTree as ET
p=Path(__file__).parent
tree=ET.parse(p/'preview.xml')
meshes={m.get('name'):m.get('file') for m in tree.findall('./asset/mesh')}
payload=[]
parents={child:parent for parent in tree.iter() for child in parent}
for b in tree.findall('./worldbody//body'):
    group=dict(name=b.get('name'),parent=parents[b].get('name') if parents[b].tag=='body' else None,pos=list(map(float,b.get('pos','0 0 0').split())),parts=[])
    for geom in b.findall('geom'):
        data=(p/meshes[geom.get('mesh')]).read_bytes();n=struct.unpack_from('<I',data,80)[0]
        points=[];faces=[];lookup={}
        for i in range(n):
            triangle=struct.unpack_from('<12fH',data,84+50*i)[3:12]
            for j in range(0,9,3):
                offset=list(map(float,geom.get('pos','0 0 0').split()))
                q=tuple(round(x+offset[k],6) for k,x in enumerate(triangle[j:j+3]))
                if q not in lookup:lookup[q]=len(points)//3;points.extend(q)
                faces.append(lookup[q])
        group['parts'].append(dict(name=geom.get('name'),vertices=points,faces=faces,color=list(map(float,geom.get('rgba').split()))[:3]))
    payload.append(group)
fragment=(p/'viewer-template.html').read_text(encoding='utf-8').replace('>+    <label','>\n    <label').replace('__MODEL_DATA__',json.dumps(payload,separators=(',',':')))
viz=Path('C:/Users/Hammer/.codex/visualizations/2026/10/24/01a16c9f-92a5-7533-90b2-77b95ce0c2e5/hip-exo-viewer.html')
viz.write_text(fragment,encoding='utf-8')
wrapper='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>髋关节外骨骼 · 三维查看</title><style>
:root{--foreground:#253243;--background:#edf2f6;--muted-foreground:#556575;color-scheme:light}*{box-sizing:border-box}body{margin:0;padding:24px;background:var(--background);color:var(--foreground);font:14px/1.5 system-ui,sans-serif}#hip-exo-viewer{max-width:1100px;margin:auto}.viz-row,.viz-controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.text-small{font-size:12px}.text-muted{color:var(--muted-foreground)}.btn,.form-select{font:inherit;border:1px solid #c5ced7;border-radius:8px;padding:8px 12px;background:white;color:inherit;cursor:pointer}.form-label{display:flex;align-items:center;gap:10px}.form-check{display:flex;align-items:center;gap:6px}.form-range{width:180px;accent-color:#425f79}#hip-exo-viewer #exo-stage{height:min(65vh,650px);min-height:330px}strong{font-size:18px;font-weight:500}@media(max-width:480px){body{padding:12px}.form-label{flex-wrap:wrap}}
</style><body>'''
(p/'模型查看器.html').write_text(wrapper+fragment+'</body></html>',encoding='utf-8')
print('Generated viewer:',viz,'bytes:',viz.stat().st_size)
