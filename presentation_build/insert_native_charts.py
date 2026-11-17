"""Insert editable native charts with embedded workbook data without starting Excel."""
import io
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
C = 'http://schemas.openxmlformats.org/drawingml/2006/chart'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
for prefix, uri in [('p', P), ('a', A), ('c', C), ('r', R)]:
    ET.register_namespace(prefix, uri)

def node(parent, uri, tag, attrs=None, text=None):
    item = ET.SubElement(parent, f'{{{uri}}}{tag}', attrs or {})
    if text is not None:
        item.text = str(text)
    return item

def rawxml(root):
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)

def workbook(data):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    sheet = ET.Element(f'{{{ns}}}worksheet')
    sd = node(sheet, ns, 'sheetData')
    rows = [['Model'] + [s['name'] for s in data['series']]]
    rows += [[cat] + [s['values'][i] for s in data['series']] for i, cat in enumerate(data['categories'])]
    for i, vals in enumerate(rows, 1):
        row = node(sd, ns, 'row', {'r': str(i)})
        for j, val in enumerate(vals):
            attrs = {'r': f'{chr(65+j)}{i}'}
            if isinstance(val, (float, int)):
                cell = node(row, ns, 'c', attrs)
                node(cell, ns, 'v', text=val)
            else:
                attrs['t'] = 'inlineStr'
                cell = node(row, ns, 'c', attrs)
                node(node(cell, ns, 'is'), ns, 't', text=val)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', f'''<Types xmlns="{CT}"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>''')
        z.writestr('_rels/.rels', f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Type="{R}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{R}"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Type="{R}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', rawxml(sheet))
    return buf.getvalue()

def cached(parent, ref_type, formula, values, numeric=False):
    ref = node(parent, C, ref_type)
    node(ref, C, 'f', text=formula)
    cache = node(ref, C, 'numCache' if numeric else 'strCache')
    if numeric:
        node(cache, C, 'formatCode', text='0.000')
    node(cache, C, 'ptCount', {'val': str(len(values))})
    for i, val in enumerate(values):
        node(node(cache, C, 'pt', {'idx': str(i)}), C, 'v', text=val)

def make_chart(data):
    root = ET.Element(f'{{{C}}}chartSpace')
    node(root, C, 'lang', {'val': 'en-US'})
    chart = node(root, C, 'chart')
    node(chart, C, 'autoTitleDeleted', {'val': '1'})
    plot = node(chart, C, 'plotArea')
    node(plot, C, 'layout')
    bar = node(plot, C, 'barChart')
    node(bar, C, 'barDir', {'val': 'col'})
    node(bar, C, 'grouping', {'val': 'clustered'})
    node(bar, C, 'varyColors', {'val': '0'})
    colors = ['193449', '199C9C', 'DA9C44', 'A1A8AE']
    for i, s in enumerate(data['series']):
        ser = node(bar, C, 'ser')
        node(ser, C, 'idx', {'val': str(i)})
        node(ser, C, 'order', {'val': str(i)})
        letter = chr(66+i)
        cached(node(ser, C, 'tx'), 'strRef', f'Sheet1!${letter}$1', [s['name']])
        sp = node(ser, C, 'spPr')
        node(node(sp, A, 'solidFill'), A, 'srgbClr', {'val': colors[i]})
        node(node(sp, A, 'ln'), A, 'noFill')
        cached(node(ser, C, 'cat'), 'strRef', f'Sheet1!$A$2:$A${len(data["categories"])+1}', data['categories'])
        cached(node(ser, C, 'val'), 'numRef', f'Sheet1!${letter}$2:${letter}${len(s["values"])+1}', s['values'], True)
    if len(data['series']) <= 2:
        dl = node(bar, C, 'dLbls')
        node(dl, C, 'numFmt', {'formatCode': '0.000', 'sourceLinked': '0'})
        node(dl, C, 'dLblPos', {'val': 'outEnd'})
        node(dl, C, 'showVal', {'val': '1'})
        for tag in ['showLegendKey', 'showCatName', 'showSerName', 'showPercent', 'showBubbleSize']:
            node(dl, C, tag, {'val': '0'})
    node(bar, C, 'gapWidth', {'val': '90'})
    node(bar, C, 'overlap', {'val': '0'})
    for aid in ['100', '200']:
        node(bar, C, 'axId', {'val': aid})
    for tag, aid, pos, cross in [('catAx', '100', 'b', '200'), ('valAx', '200', 'l', '100')]:
        axis = node(plot, C, tag)
        node(axis, C, 'axId', {'val': aid})
        sc = node(axis, C, 'scaling')
        node(sc, C, 'orientation', {'val': 'minMax'})
        if tag == 'valAx':
            node(sc, C, 'max', {'val': str(data['maximum'])})
            node(sc, C, 'min', {'val': '0'})
        node(axis, C, 'delete', {'val': '0'})
        node(axis, C, 'axPos', {'val': pos})
        if tag == 'valAx':
            gridline = node(axis, C, 'majorGridlines')
            line = node(node(gridline, C, 'spPr'), A, 'ln')
            node(node(line, A, 'solidFill'), A, 'srgbClr', {'val': 'E4E8EB'})
            node(axis, C, 'numFmt', {'formatCode': '0.00', 'sourceLinked': '0'})
        node(axis, C, 'majorTickMark', {'val': 'none'})
        node(axis, C, 'minorTickMark', {'val': 'none'})
        node(axis, C, 'tickLblPos', {'val': 'nextTo'})
        node(axis, C, 'crossAx', {'val': cross})
        node(axis, C, 'crosses', {'val': 'autoZero'})
        if tag == 'catAx':
            node(axis, C, 'auto', {'val': '1'})
            node(axis, C, 'lblAlgn', {'val': 'ctr'})
            node(axis, C, 'lblOffset', {'val': '100'})
        else:
            node(axis, C, 'crossBetween', {'val': 'between'})
    if len(data['series']) > 1:
        legend = node(chart, C, 'legend')
        node(legend, C, 'legendPos', {'val': 'b'})
        node(legend, C, 'layout')
        node(legend, C, 'overlay', {'val': '0'})
    node(chart, C, 'plotVisOnly', {'val': '1'})
    node(chart, C, 'dispBlanksAs', {'val': 'gap'})
    tx = node(root, C, 'txPr')
    node(tx, A, 'bodyPr')
    node(tx, A, 'lstStyle')
    para = node(tx, A, 'p')
    pr = node(node(para, A, 'pPr'), A, 'defRPr', {'sz': '1400'})
    node(pr, A, 'latin', {'typeface': 'Malgun Gothic'})
    node(pr, A, 'ea', {'typeface': 'Malgun Gothic'})
    node(para, A, 'endParaRPr', {'lang': 'en-US'})
    node(node(root, C, 'externalData', {f'{{{R}}}id': 'rId1'}), C, 'autoUpdate', {'val': '0'})
    return rawxml(root)

path = OUT / 'GMSC_7min.pptx'
with zipfile.ZipFile(path) as z:
    files = {n: z.read(n) for n in z.namelist()}
types = ET.fromstring(files['[Content_Types].xml'])
node(types, CT, 'Default', {'Extension': 'xlsx', 'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'})
spec = json.loads((OUT / 'deck_spec.json').read_text(encoding='utf-8'))
chart_number = 0
for slide_number, slide in enumerate(spec, 1):
    if not slide['chart']:
        continue
    chart_number += 1
    cid = chart_number
    part = f'ppt/charts/chart{cid}.xml'
    files[part] = make_chart(slide['chart'])
    files[f'ppt/embeddings/workbook{cid}.xlsx'] = workbook(slide['chart'])
    files[f'ppt/charts/_rels/chart{cid}.xml.rels'] = f'<Relationships xmlns="{REL}"><Relationship Id="rId1" Type="{R}/package" Target="../embeddings/workbook{cid}.xlsx"/></Relationships>'.encode()
    node(types, CT, 'Override', {'PartName': '/'+part, 'ContentType': 'application/vnd.openxmlformats-officedocument.drawingml.chart+xml'})
    slidepath = f'ppt/slides/slide{slide_number}.xml'
    root = ET.fromstring(files[slidepath])
    tree = root.find(f'{{{P}}}cSld/{{{P}}}spTree')
    frame = node(tree, P, 'graphicFrame')
    nv = node(frame, P, 'nvGraphicFramePr')
    node(nv, P, 'cNvPr', {'id': str(900+cid), 'name': 'Editable evidence chart'})
    node(nv, P, 'cNvGraphicFramePr')
    node(nv, P, 'nvPr')
    xf = node(frame, P, 'xfrm')
    node(xf, A, 'off', {'x': str(35*12700), 'y': str(135*12700)})
    node(xf, A, 'ext', {'cx': str(605*12700), 'cy': str(340*12700)})
    gd = node(node(frame, A, 'graphic'), A, 'graphicData', {'uri': C})
    rid = f'rIdEvidenceChart{cid}'
    node(gd, C, 'chart', {f'{{{R}}}id': rid})
    files[slidepath] = rawxml(root)
    relpath = f'ppt/slides/_rels/slide{slide_number}.xml.rels'
    rels = ET.fromstring(files[relpath])
    node(rels, REL, 'Relationship', {'Id': rid, 'Type': R+'/chart', 'Target': f'../charts/chart{cid}.xml'})
    files[relpath] = rawxml(rels)
ET.register_namespace('', CT)
files['[Content_Types].xml'] = rawxml(types)
with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, content in files.items():
        z.writestr(name, content)
print(f'Inserted {chart_number} native editable charts with embedded workbooks.')
