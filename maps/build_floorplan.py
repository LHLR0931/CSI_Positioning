"""Render the confirmed 2026-10-10 floorplan as vector PDF/SVG and PNG.

This is a measured-coordinate engineering drawing, not an edit of the source
hand sketch. Dimensions are net-space abstractions; heights remain unknown.
Run from the repository root with Python and PyMuPDF installed (or tmp/pdf_deps).
"""
from pathlib import Path
import json
import hashlib
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/pdf_deps'))
import pymupdf as fitz

PDF = ROOT / 'output/pdf/CSI_Apartment_Floorplan_v1_2026-10-10.pdf'
PNG = ROOT / 'docs/apartment_floorplan_v1_2026-10-10.png'
SVG = PNG.with_suffix('.svg')
QA = ROOT / 'tmp/pdfs/floorplan-v1'
DATA_PATH = Path(__file__).with_name('apartment_v1_2026-10-10.json')
DATA = json.loads(DATA_PATH.read_text(encoding='utf-8'))
REGIONS = {item['id']: item for item in DATA['regions']}
POINTS = {item['label']: item for item in DATA['grid']['points']}
assert DATA['metadata']['map_id'] == 'apartment_v1_2026-10-10'
# Geometry is data-driven; this sheet's dimension/label layout is specific to V1.
# Fail visibly if a later geometry version needs re-layout instead of silently
# retaining obsolete dimension captions.
assert DATA['coordinate_system']['derived_reference_lines_m'] == {
    'balcony_left_x': -2.742, 'balcony_divider_x': -.742, 'office_right_x': 3.2,
    'living_right_and_kitchen_left_x': 4, 'far_right_x': 5.758, 'bottom_y': -.4,
    'corridor_bottom_y': 1.6, 'office_and_kitchen_front_y': 2.8, 'top_y': 4.6,
}, 'Review the V1 dimension annotation layout for this geometry change.'
for folder in (PDF.parent, PNG.parent, QA):
    folder.mkdir(parents=True, exist_ok=True)

font_dir = Path(os.environ.get('WINDIR', '')) / 'Fonts'
FONTS = {'r': str(font_dir / 'msyh.ttc'), 'b': str(font_dir / 'msyhbd.ttc')}
FM = {k: fitz.Font(fontfile=v) for k, v in FONTS.items()}
INK = '#203745'
MUTED = '#64757F'
TEAL = '#087F74'
CYAN = '#167DA0'
AMBER = '#B97828'
GRAY = '#D5DEE2'
S = 82.0
OX, OY = 335.0, 592.2
W, H = 1191.0, 842.0
doc = fitz.open()
text_boxes = []


def rgb(value):
    return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))


def xy(x, y):
    return OX + S * x, OY - S * y


def text(p, value, x, y, size=12, font='r', color=INK, align='left'):
    width = FM[font].text_length(value, fontsize=size)
    if align == 'center':
        x -= width / 2
    elif align == 'right':
        x -= width
    baseline = y + FM[font].ascender * size
    p.insert_text((x, baseline), value, fontname=font, fontsize=size, color=rgb(color))
    text_boxes.append((p.number + 1, x, y, x + width, y + size * 1.4, value))


def multi(p, lines, x, y, size=12, color=INK, font='r', leading=None, align='left'):
    for i, line in enumerate(lines.split('\n')):
        text(p, line, x, y + i * (leading or size * 1.65), size, font, color, align)


def line(p, a, b, color=INK, width=1, dash=None):
    p.draw_line(a, b, color=rgb(color), width=width, dashes=dash)


def rect(p, box, fill, stroke=None, width=1):
    p.draw_rect(fitz.Rect(*box), fill=rgb(fill), color=rgb(stroke) if stroke else None, width=width)


def world_rect(p, x0, y0, x1, y1, fill):
    a, b = xy(x0, y1), xy(x1, y0)
    rect(p, (*a, *b), fill)


def world_line(p, a, b, color=INK, width=2.5, dash=None):
    line(p, xy(*a), xy(*b), color, width, dash)


def arrow(p, a, b, color=MUTED, width=.8, head=4):
    line(p, a, b, color, width)
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    for sign in (-1, 1):
        end = (b[0] - head * ux + sign * head * .45 * uy,
               b[1] - head * uy - sign * head * .45 * ux)
        line(p, b, end, color, width)


def hdim(p, x0, x1, y, ref_y, label, color=MUTED, size=11):
    a, b = xy(x0, 0)[0], xy(x1, 0)[0]
    for x in (a, b):
        line(p, (x, ref_y), (x, y + (5 if y > ref_y else -5)), GRAY, .7)
    arrow(p, (a, y), (b, y), color)
    arrow(p, (b, y), (a, y), color)
    text(p, label, (a + b) / 2, y - 20, size, 'r', color, 'center')


def vdim(p, y0, y1, x, ref_x, label, side='right', color=MUTED, size=11):
    a, b = xy(0, y0)[1], xy(0, y1)[1]
    for y in (a, b):
        line(p, (ref_x, y), (x + (5 if x > ref_x else -5), y), GRAY, .7)
    arrow(p, (x, a), (x, b), color)
    arrow(p, (x, b), (x, a), color)
    text(p, label, x + (8 if side == 'right' else -8), (a + b) / 2 - 8,
         size, 'r', color, 'left' if side == 'right' else 'right')


def page(title, subtitle, number):
    p = doc.new_page(width=W, height=H)
    for name, path in FONTS.items():
        p.insert_font(fontname=name, fontfile=path)
    text(p, title, 40, 24, 28, 'b')
    text(p, subtitle, 40, 70, 12, 'r', MUTED)
    line(p, (40, 102), (1151, 102), GRAY, .8)
    line(p, (40, 790), (1151, 790), GRAY, .8)
    text(p, 'CSI CAPSTONE  /  APARTMENT BASE MAP  /  V1.0  /  2026-10-10', 40, 805, 10, 'r', MUTED)
    text(p, f'{number} / 2', 1151, 805, 10, 'r', MUTED, 'right')
    return p


def table(p, x, y, widths, headers, rows, row_h=35):
    full = sum(widths)
    rect(p, (x, y, x + full, y + 34), INK)
    xx = x
    for cell, width in zip(headers, widths):
        text(p, cell, xx + 10, y + 7, 12, 'b', '#FFFFFF')
        xx += width
    yy = y + 34
    for index, row in enumerate(rows):
        rect(p, (x, yy, x + full, yy + row_h), '#F1F6F6' if index % 2 == 0 else '#FFFFFF', GRAY, .5)
        xx = x
        for cell, width in zip(row, widths):
            text(p, str(cell), xx + 10, yy + 8, 11.3)
            xx += width
        yy += row_h
    return yy


# PAGE 1: equal-scale, dimensioned plan. Unknown void is not floor.
p = page('公寓建模基础平面图', '依据三轮共同确认的手绘尺寸重绘 · 单位：米 · 按比例绘制 · 墙厚忽略，保留内净边界', 1)
region_colors = {'balcony': '#FFF5D9', 'living_room': '#EEF8F4',
                 'corridor_extension': '#EEF8F4', 'office': '#FFF0E9',
                 'kitchen': '#EDF0FA', 'unknown': '#F0F1F2'}
for region in DATA['regions']:
    world_rect(p, *region['min_xy_m'], *region['max_xy_m'], region_colors[region['id']])

# Hatch only the unknown footprint, clipping each diagonal analytically.
xa, ya = xy(3.2, 4.6)
xb, yb = xy(4, 2.8)
for offset in range(-150, 90, 12):
    start = max(0, -offset)
    end = min(yb - ya, xb - xa - offset)
    if end > start:
        line(p, (xa + offset + start, ya + start), (xa + offset + end, ya + end), '#D9DDDF', .55)

# Boundary types are deliberately different from physical height/material.
styles = {'wall': (INK, 2.5, None),
          'furniture_simplified_barrier': (AMBER, 4, '[8 3]'),
          'region_boundary_unspecified': (MUTED, 1, '[2 3]')}
for boundary in DATA['boundaries']:
    if boundary['kind'] == 'open_connection':
        continue
    color, width, dash = styles[boundary['kind']]
    world_line(p, boundary['start_xy_m'], boundary['end_xy_m'], color, width, dash)
for opening in DATA['openings']:
    a, b = opening['start_xy_m'], opening['end_xy_m']
    world_line(p, a, b, CYAN, 1, '[3 3]')
    for point in (a, b):
        p.draw_circle(xy(*point), 2.4, fill=rgb(CYAN), color=rgb(CYAN))

text(p, '阳台', *xy(-1.742, 3.85), 21, 'b', '#876126', 'center')
text(p, 'BALCONY', *xy(-1.742, 3.43), 10, 'r', '#876126', 'center')
text(p, '内部不细建', *xy(-1.742, 3.03), 11, 'r', '#876126', 'center')
text(p, '办公区', *xy(1.229, 3.96), 20, 'b', '#A45C39', 'center')
text(p, 'OFFICE', *xy(1.229, 3.58), 10, 'r', '#A45C39', 'center')
text(p, '入口两侧：桌柜简化边界', *xy(1.229, 3.26), 11, 'r', '#A45C39', 'center')
text(p, '厨房', *xy(4.879, 3.95), 20, 'b', '#536299', 'center')
text(p, 'KITCHEN', *xy(4.879, 3.53), 10, 'r', '#536299', 'center')
text(p, '内部不细建', *xy(4.879, 3.18), 10, 'r', '#536299', 'center')
multi(p, '未知\n区域\n不建模', *xy(3.6, 3.97), 11, MUTED, leading=21, align='center')
text(p, '客厅  /  CSI点阵', *xy(1.28, 2.42), 17, 'b', TEAL, 'center')
text(p, '15点 · 相邻间距0.800 m · 四邻接连接', *xy(1.28, 2.08), 10, 'r', TEAL, 'center')
text(p, '走廊', *xy(4.879, 2.39), 17, 'b', INK, 'center')
text(p, 'CORRIDOR', *xy(4.879, 2.04), 9, 'r', MUTED, 'center')

# Four-neighbour graph and 15 unchanged grid points.
for edge in DATA['grid']['edges']:
    a, b = POINTS[edge['from']], POINTS[edge['to']]
    world_line(p, (a['x_m'], a['y_m']), (b['x_m'], b['y_m']), '#AECBC2', 1.1)
for point in POINTS.values():
    cx, cy = xy(point['x_m'], point['y_m'])
    p.draw_circle((cx, cy), 12, color=rgb(TEAL), fill=rgb('#FFFFFF'), width=1.8)
    text(p, str(point['label']), cx, cy - 10, 13, 'b', TEAL, 'center')
    text(p, f"({point['i']},{point['j']})", cx, cy + 15, 8.4, 'r', MUTED, 'center')

# Opening labels remain outside walking lanes and do not invent a target point.
multi(p, 'D01 阳台门\n开口 0.800 m', 146, 467, 11, CYAN, leading=18)
line(p, (231, 495), (265, 495), CYAN, .8)
text(p, 'D02 · 0.800', *xy(2, 2.72), 10, 'b', CYAN, 'center')
text(p, 'D03 · 0.800', *xy(4.4, 2.72), 9.3, 'b', CYAN, 'center')

# Exterior dimensions: axis-projected distances, never image pixel estimates.
top = xy(0, 4.6)[1]
bottom = xy(0, -.4)[1]
hdim(p, -.742, 5.758, 135, top, '6.500 m（不含阳台）', INK, 13)
hdim(p, -2.742, -.742, 179, top, '2.000 m')
hdim(p, -.742, 3.2, 179, top, '3.942 m')
hdim(p, 4, 5.758, 179, top, '1.758 m*')
vdim(p, -.4, 4.6, 73, xy(-2.742, 0)[0], '5.000', 'left', INK, 11)
vdim(p, 2.8, 4.6, 842, xy(5.758, 0)[0], '1.800', 'right')
vdim(p, 1.6, 2.8, 842, xy(5.758, 0)[0], '1.200', 'right')
hdim(p, -.742, 0, 663, bottom, '0.742')
hdim(p, 0, 3.2, 663, bottom, '3.200 m')
hdim(p, 3.2, 4, 663, bottom, '0.800')
hdim(p, 4, 5.758, 503, xy(0, 1.6)[1], '1.758 m*')
# Point 0 to bottom wall, without shifting the displayed coordinate origin.
vdim(p, -.4, 0, 306, OX, '', 'left')
text(p, '0.400', 265, 602, 9.5, 'r', MUTED, 'right')
line(p, (268, 610), (302, 610), MUTED, .7)

# Coordinate inset in the exterior blank area.
arrow(p, (742, 601), (792, 601), TEAL, 1.1, 5)
arrow(p, (742, 601), (742, 551), TEAL, 1.1, 5)
text(p, '+x', 797, 590, 11, 'b', TEAL)
text(p, '+y', 742, 531, 11, 'b', TEAL, 'center')
text(p, 'O = 0号点中心', 692, 624, 10, 'r', MUTED)

# Scale bar is valid independent of print scaling.
text(p, '图示比例尺', 110, 694, 10, 'r', MUTED)
line(p, (110, 729), (274, 729), INK, 2)
for xx, label in [(110, '0'), (192, '1'), (274, '2 m')]:
    line(p, (xx, 724), (xx, 734), INK, 1.5)
    text(p, label, xx, 740, 10, 'r', MUTED, 'center')
text(p, '* 1.758 = 2.558 - 0.800，为推算尺寸。', 328, 704, 11, 'r', MUTED)
text(p, '点下方(i,j)为格点索引；米制坐标为(0.8i, 0.8j)。', 328, 729, 11, 'r', MUTED)
text(p, '四邻连线表示规划通路，不表示已验证的CSI覆盖边界。', 328, 754, 11, 'r', MUTED)

# Right-side notes: leave actual drawing entirely unobstructed.
rect(p, (900, 122, 1151, 768), '#F6F8F9', GRAY, .7)
text(p, '建模与读图说明', 916, 137, 18, 'b')
text(p, '01  边界图例', 916, 177, 13, 'b', TEAL)
legend = [(INK, None, 2.5, '实墙 / 内净边界'), (AMBER, '[8 3]', 4, '桌柜简化阻挡边界'),
          (CYAN, '[3 3]', 1, '门洞或无门通行开口'), ('#AECBC2', None, 1.1, '四邻接格点通路'),
          (MUTED, '[2 3]', 1, '空间分界 / 材质未测')]
for index, (color, dash, width, label) in enumerate(legend):
    yy = 207 + 25 * index
    line(p, (918, yy + 7), (950, yy + 7), color, width, dash)
    text(p, label, 960, yy - 2, 11)
line(p, (916, 337), (1135, 337), GRAY, .7)
text(p, '02  采用的简化', 916, 354, 13, 'b', TEAL)
multi(p, '墙厚暂不计，保持通行净空间。\n办公入口两侧桌柜视为阻挡。\n仅保留此图中的空间；\n未知区域不生成可走地面。\n区域内部不做家具细节建模。', 916, 384, 11, leading=22)
line(p, (916, 508), (1135, 508), GRAY, .7)
text(p, '03  尚未冻结的参数', 916, 525, 13, 'b', TEAL)
multi(p, 'T01：客厅侧阳台门前站立点。\n阳台出口格点编号尚未绑定。\n门高、门槛、开门后实际净宽。\n桌柜高度与AR参考图片位姿。', 916, 555, 11, leading=23)
line(p, (916, 660), (1135, 660), GRAY, .7)
text(p, '04  数据来源', 916, 677, 13, 'b', TEAL)
multi(p, '2026-10-10 用户手绘图\n及三轮文字确认。\n数字小数位不等于测量精度。', 916, 707, 10.5, MUTED, leading=18)

# PAGE 2: compact reference for future modelling, no invented heights or targets.
p = page('建模坐标与入口索引', '与第1页共用同一坐标系 · O = 0号点 · +x沿0→4，+y沿0→10 · 以下均为地面米制坐标', 2)
text(p, '入口端点与中点', 40, 122, 18, 'b', TEAL)
table(p, 40, 158, [170, 226, 226, 226, 263], ['对象', '端点A (x, y)', '端点B (x, y)', '几何中点 (x, y)', '说明'], [
    ['D01 阳台门', '(-0.742, 1.600)', '(-0.742, 0.800)', '(-0.742, 1.200)', 'A上端 / B下端'],
    ['D02 办公区入口', '(1.600, 2.800)', '(2.400, 2.800)', '(2.000, 2.800)', 'A左端 / B右端；无门扇'],
    ['D03 厨房门', '(4.000, 2.800)', '(4.800, 2.800)', '(4.400, 2.800)', 'A左端 / B右端'],
], 37)
text(p, '三处开口跨度均为0.800 m。几何中点用于模型定位，不自动作为到达判定点。', 40, 314, 12, 'b', CYAN)

text(p, '空间轮廓 / 坐标范围', 40, 353, 18, 'b', TEAL)
table(p, 40, 388, [145, 225, 225, 190], ['区域', 'x范围 / m', 'y范围 / m', '用途'], [
    ['阳台', '-2.742 至 -0.742', '-0.400 至 4.600', '背景轮廓'],
    ['客厅主体', '-0.742 至 4.000', '-0.400 至 2.800', '点阵所在空间'],
    ['走廊凸出段', '4.000 至 5.758', '1.600 至 2.800', '与客厅连通'],
    ['办公区', '-0.742 至 3.200', '2.800 至 4.600', '仅细化入口'],
    ['厨房', '4.000 至 5.758', '2.800 至 4.600', '仅细化入口'],
    ['未知区域', '3.200 至 4.000', '2.800 至 4.600', '不建模 / 不可通行'],
], 36)
text(p, '客厅与走廊交界 x=4.000、y=1.600至2.800 为开放连接，不额外建墙。', 40, 650, 11.5, 'b')

rect(p, (855, 353, 1151, 671), '#F6F8F9', GRAY, .7)
text(p, '15点编号与换算', 874, 371, 16, 'b', TEAL)
multi(p, 'i = 0..4，j = 0..2\nlabel = 5j + i\n(x, y) = (0.8i, 0.8j) m', 874, 408, 12, leading=27)
for j in range(3):
    for i in range(5):
        text(p, str(5 * j + i), 892 + i * 51, 510 + (2 - j) * 29, 13, 'b', TEAL, 'center')
text(p, '15个节点 / 22条无向四邻边', 874, 615, 11, 'b')
text(p, '不增加格点，不允许对角捷径。', 874, 641, 10.5, 'r', MUTED)

line(p, (40, 686), (1151, 686), GRAY, .8)
text(p, '建模边界', 40, 705, 13, 'b', TEAL)
text(p, '按净空间放置边界；添加显示厚度时不能挤占净宽。桌柜简化边界不等于真实满高墙。', 144, 705, 11.5)
text(p, '导航待定', 40, 735, 13, 'b', TEAL)
text(p, 'T01站立坐标与出口label留空；门洞中点、出口节点、实际到达位置分别记录。', 144, 735, 11.5)
text(p, '配套数据', 40, 763, 10, 'r', MUTED)
text(p, 'maps/apartment_v1_2026-10-10.json  +  maps/grid_points_v1_2026-10-10.csv', 144, 763, 10, 'r', MUTED)

doc.set_metadata({'title': 'CSI Apartment Base Map V1.0', 'author': 'CSI Capstone Project',
                  'subject': 'Confirmed net-space plan, 15 grid points, three openings; 2026-10-10'})
doc.set_toc([[1, '平面底图', 1], [1, '建模坐标与入口索引', 2]])
doc.subset_fonts()
doc.save(PDF, garbage=4, deflate=True)
doc.close()
check = fitz.open(PDF)
check[0].get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).save(PNG)
SVG.write_text(check[0].get_svg_image(text_as_path=True), encoding='utf-8')
for index, pg in enumerate(check):
    pg.get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(QA / f'page-{index + 1}.png')
assert len(check) == 2
for pn, x0, y0, x1, y1, value in text_boxes:
    assert x0 >= 25 and x1 <= 1161 and y0 >= 15 and y1 <= 831, (pn, value, (x0, y0, x1, y1))
all_text = '\n'.join(pg.get_text() for pg in check)
for required in ['3.942', '0.742', '1.758', 'D01', 'D02', 'D03', 'T01', '5.758', '未知']:
    assert required in all_text, required
summary = {'pdf': str(PDF), 'png': str(PNG), 'svg': str(SVG), 'pages': 2,
           'geometry_source': str(DATA_PATH), 'geometry_sha256': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
           'dimension_chains': 'pass', 'text_page_bounds': 'pass', 'visual_review': 'pending'}
(QA / 'qa.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
