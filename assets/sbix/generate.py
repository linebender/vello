#!/usr/bin/env python3
# Copyright 2026 the Vello Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fonttools==4.65.0",
#     "Pillow==12.3.0",
# ]
# ///

"""Generate the glyf and CFF sbix test fonts."""

import io
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables.sbixGlyph import Glyph
from fontTools.ttLib.tables.sbixStrike import Strike
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
FONT_PATHS = {"glyf": ROOT / "sbix.ttf", "cff": ROOT / "sbix.otf"}

UPEM = 1024
BASE = 64

PATTERNS = {
    "A": "01110 10001 10001 11111 10001 10001 10001",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01111 10000 10000 10000 10000 10000 01111",
    "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01111 10000 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "11111 00100 00100 00100 00100 00100 11111",
    "J": "00111 00010 00010 00010 10010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110",
    "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 11011 10001",
    "X": "10001 10001 01010 00100 01010 10001 10001",
    "!": "00100 00100 00100 00100 00100 00000 00100",
    "?": "01110 10001 00001 00010 00100 00000 00100",
}


CASES = {
    "A": {},
    "B": {"inner": [32, 0]},
    "C": {"inner": [-32, 0]},
    "D": {"inner": [0, 32]},
    "E": {"inner": [0, -32]},
    "F": {"outer": [512, 0]},
    "G": {"outer": [-512, 0]},
    "H": {"outer": [0, 512]},
    "I": {"outer": [0, -512]},
    "J": {"inner": [24, -32]},
    "K": {"inner": [-32, 24]},
    "L": {"outer": [384, -512]},
    "M": {"outer": [-512, 384]},
    "N": {"outer": [256, 256], "inner": [-24, -32]},
    "O": {"outer": [-256, -256], "inner": [24, 32]},
    "P": {"outer": [512, 512], "inner": [-32, -32]},
    "Q": {"lsb": 512, "no_contour": True},
    "R": {"lsb": -512, "no_contour": True},
    "S": {"outer": [384, 0], "lsb": 384},
    "T": {"outer": [-256, 0], "lsb": -256},
    "U": {"color": [24, 91, 177, 128]},
    "V": {"extra_padding": [113, 16]},
    "W": {"strikes": [32, 128]},
    "X": {"kind": "outline", "color": [0, 0, 0, 255]},
    "Y": {"kind": "dupe", "target": "A"},
    "Z": {"kind": "dupe", "target": "Y"},
    "!": {"extra_padding": [16, 91]},
    "?": {"lsb": 256},
}


def rectangles(char):
    return [
        [48 + x * 4, -96 + y * 4, 4, 4]
        for y, row in enumerate(PATTERNS[char].split())
        for x, v in enumerate(row)
        if v == "1"
    ]


def build_font(outline_format):
    is_glyf = outline_format == "glyf"
    font_path = FONT_PATHS[outline_format]
    family = "Sbix Test Glyf" if is_glyf else "Sbix Test CFF"
    ps_name = "SbixTestGlyf-Regular" if is_glyf else "SbixTestCFF-Regular"
    names = [".notdef", "space"] + [f"uni{ord(c):04X}" for c in CASES]
    cmap = {32: "space", **{ord(c): n for c, n in zip(CASES, names[2:])}}

    cmap.update({ord(c.lower()): cmap[ord(c)] for c in CASES if c.isalpha()})

    fb = FontBuilder(UPEM, isTTF=is_glyf)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap(cmap)

    char_strings = {}
    metrics = {}
    entries = {}

    for gid, (name, char) in enumerate(zip(names, [" ", " ", *CASES])):
        cfg = CASES.get(char, {})
        if char == "?" and not is_glyf:
            cfg = {**cfg, "no_contour": True}
        kind = cfg.get("kind", "png") if gid >= 2 else "empty"

        inner = cfg.get("inner", [0, 0])
        outer = cfg.get("outer")
        lsb = cfg.get("lsb", outer[0] if outer else 0)

        artwork = "A" if kind == "dupe" else char
        rects = rectangles(artwork) if gid >= 2 else []
        color = cfg.get("color", [24, 91, 177, 255])

        advance = 2304 if gid >= 2 else 1024
        pen = TTGlyphPen(None) if is_glyf else T2CharStringPen(advance, None)

        if kind == "outline":
            for x, y, w, h in rects:
                points = [(x, -y - h), (x + w, -y - h), (x + w, -y), (x, -y)]
                pen.moveTo(tuple(v * 16 for v in points[0]))
                for p in points[1:]:
                    pen.lineTo(tuple(v * 16 for v in p))
                pen.closePath()
            lsb = min(r[0] for r in rects) * 16

        elif gid >= 2 and not cfg.get("no_contour", False):
            x, y = outer or [0, 0]
            art_right = max(r[0] + r[2] for r in rects) * UPEM // BASE
            art_top = max(-r[1] for r in rects) * UPEM // BASE
            x_max = max(x + UPEM, x + art_right - lsb)
            y_max = max(y + UPEM, art_top)
            pen.moveTo((x, y))
            pen.lineTo((x_max, y_max))
            if is_glyf:
                pen.endPath()
            else:
                pen.closePath()

        char_strings[name] = pen.glyph() if is_glyf else pen.getCharString()
        metrics[name] = (advance, lsb)

        has_contour = gid >= 2 and not cfg.get("no_contour", False)
        origin = (
            [lsb, outer[1] if outer is not None else 0]
            if is_glyf and has_contour
            else [0, 0]
        )

        entries[name] = dict(
            char=char,
            kind=kind,
            offset=inner,
            outline=has_contour,
            placement_origin=origin,
            lsb=lsb,
            rectangles=rects,
            color=color,
            target=cfg.get("target"),
        )

    if is_glyf:
        fb.setupGlyf(char_strings)
    else:
        fb.setupCFF(
            ps_name,
            {"FullName": family, "FamilyName": family, "Weight": "Regular"},
            char_strings,
            {"defaultWidthX": 0, "nominalWidthX": 0},
        )
        for name, entry in entries.items():
            if entry["outline"]:
                assert char_strings[name].calcBounds(None)[0] == entry["lsb"]
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=2048, descent=-768)

    fb.setupNameTable(
        dict(
            copyright="Copyright 2026 the Vello Authors",
            licenseDescription=(
                "Licensed under Apache-2.0 OR MIT, at your option. "
                "This font may be freely used, modified, and redistributed "
                "under the terms of either license."
            ),
            licenseInfoURL="https://github.com/linebender/vello#license",
            familyName=family,
            styleName="Regular",
            uniqueFontIdentifier=f"{ps_name}-1",
            fullName=family,
            version="Version 1.000",
            psName=ps_name,
        )
    )

    fb.setupOS2(
        fsType=0,
        fsSelection=0x40,
        achVendID="    ",
        sTypoAscender=2048,
        sTypoDescender=-768,
        sTypoLineGap=0,
        usWinAscent=2304,
        usWinDescent=1024,
        sxHeight=1536,
        sCapHeight=1536,
    )

    fb.setupPost()

    fb.font["head"].flags &= ~2
    fb.font["head"].created = fb.font["head"].modified = 3762115200
    fb.font.recalcTimestamp = False

    table = newTable("sbix")
    table.version = 1
    table.flags = 1
    table.strikes = {}

    for ppem in [32, 64, 128]:
        scale = ppem / BASE
        strike = Strike(ppem=ppem, resolution=72)

        for name, e in entries.items():
            cfg = CASES.get(e["char"], {})
            if e["kind"] in ("empty", "outline") or ppem not in cfg.get(
                "strikes", [32, 64, 128]
            ):
                strike.glyphs[name] = Glyph(glyphName=name)
                continue

            if e["kind"] == "dupe":
                target = cmap[ord(e["target"])]
                strike.glyphs[name] = Glyph(
                    glyphName=name, graphicType="dupe", referenceGlyphName=target
                )
                continue

            dx, dy = [round(v * scale) for v in e["offset"]]
            ox, oy = [v * ppem / UPEM for v in e["placement_origin"]]

            left = min(r[0] * scale - ox - dx for r in e["rectangles"])
            right = max((r[0] + r[2]) * scale - ox - dx for r in e["rectangles"])
            top = min(r[1] * scale + oy + dy for r in e["rectangles"])
            bottom = max((r[1] + r[3]) * scale + oy + dy for r in e["rectangles"])

            assert left >= 0 and bottom <= 0, (name, ppem, "uncompensatable offset")

            padx, pady = cfg.get("extra_padding", [16, 16])
            w = int(right) + max(1, round(padx * scale))
            h = int(-top) + max(1, round(pady * scale))

            im = Image.new("RGBA", (w, h))
            d = ImageDraw.Draw(im)

            for x, y, rw, rh in e["rectangles"]:
                px = x * scale - ox - dx
                py = y * scale + oy + dy + h

                assert px == int(px) and py == int(py)
                assert 0 <= px < px + rw * scale <= w and 0 <= py < py + rh * scale <= h

                d.rectangle(
                    (px, py, px + rw * scale - 1, py + rh * scale - 1),
                    fill=tuple(e["color"]),
                )

            buf = io.BytesIO()
            im.save(buf, format="PNG")
            strike.glyphs[name] = Glyph(
                glyphName=name,
                originOffsetX=dx,
                originOffsetY=dy,
                graphicType="png ",
                imageData=buf.getvalue(),
            )

            canvas = Image.new("RGBA", (512, 512))
            canvas.alpha_composite(im, (int(96 + ox + dx), int(320 - oy - dy - h)))

            expected = Image.new("RGBA", canvas.size)
            ed = ImageDraw.Draw(expected)

            for x, y, rw, rh in e["rectangles"]:
                ed.rectangle(
                    (
                        96 + x * scale,
                        320 + y * scale,
                        96 + (x + rw) * scale - 1,
                        320 + (y + rh) * scale - 1,
                    ),
                    fill=tuple(e["color"]),
                )

            assert canvas.tobytes() == expected.tobytes(), (name, ppem)

        table.strikes[ppem] = strike

    fb.font["sbix"] = table
    fb.save(font_path)

    return font_path


if __name__ == "__main__":
    for outline_format in FONT_PATHS:
        print(f"Generated {build_font(outline_format).name}")
