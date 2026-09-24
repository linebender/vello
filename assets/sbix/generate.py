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

"""Generate an sbix test font with 27 glyph cases and three strikes."""

import hashlib
import io
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables.sbixGlyph import Glyph
from fontTools.ttLib.tables.sbixStrike import Strike
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
FONT_PATH = ROOT / "sbix.ttf"

UPEM = 1024
BASE = 64

SAMPLE = "ABCDEF\nGHIJKL\nMNOPQR\nSTUVWX\nYZ!"

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
}


CASES = {
    "A": dict(case="Control: all offsets zero"),
    "B": dict(case="Inner X +32 px", inner=[32, 0]),
    "C": dict(case="Inner X −32 px", inner=[-32, 0]),
    "D": dict(case="Inner Y +32 px", inner=[0, 32]),
    "E": dict(case="Inner Y −32 px", inner=[0, -32]),
    "F": dict(case="Outer X +512 units", outer=[512, 0]),
    "G": dict(case="Outer X −512 units", outer=[-512, 0]),
    "H": dict(case="Outer Y +512 units", outer=[0, 512]),
    "I": dict(case="Outer Y −512 units", outer=[0, -512]),
    "J": dict(case="Inner X/Y +24/−32", inner=[24, -32]),
    "K": dict(case="Inner X/Y −32/+24", inner=[-32, 24]),
    "L": dict(case="Outer X/Y +384/−512", outer=[384, -512]),
    "M": dict(case="Outer X/Y −512/+384", outer=[-512, 384]),
    "N": dict(
        case="Mixed outer +256/+256, inner −24/−32",
        outer=[256, 256],
        inner=[-24, -32],
    ),
    "O": dict(
        case="Mixed outer −256/−256, inner +24/+32",
        outer=[-256, -256],
        inner=[24, 32],
    ),
    "P": dict(
        case="Outer/inner cancel in both axes",
        outer=[512, 512],
        inner=[-32, -32],
    ),
    "Q": dict(case="LSB +512, no contours: ignored", lsb=512, no_contour=True),
    "R": dict(case="LSB −512, no contours: ignored", lsb=-512, no_contour=True),
    "S": dict(case="xMin −256, LSB +384", outer=[-256, 0], lsb=384),
    "T": dict(case="xMin +384, LSB −256", outer=[384, 0], lsb=-256),
    "U": dict(case="Half-transparent bitmap", color=[24, 91, 177, 128]),
    "V": dict(
        case="Extra horizontal padding: 113 px on the right",
        extra_padding=[113, 16],
    ),
    "W": dict(case="Missing 64-ppem bitmap: use another strike", strikes=[32, 128]),
    "X": dict(
        case="Outline fallback: no bitmap in any strike",
        kind="outline",
        color=[0, 0, 0, 255],
    ),
    "Y": dict(case="dupe → A (should draw A)", kind="dupe", target="A"),
    "Z": dict(case="dupe → Y → A (should draw A)", kind="dupe", target="Y"),
    "!": dict(
        case="Extra vertical padding: 91 px above the artwork",
        extra_padding=[16, 91],
    ),
}


def rectangles(char):
    return [
        [48 + x * 4, -96 + y * 4, 4, 4]
        for y, row in enumerate(PATTERNS[char].split())
        for x, v in enumerate(row)
        if v == "1"
    ]


def build():
    names = [".notdef", "space"] + [f"uni{ord(c):04X}" for c in CASES]
    cmap = {32: "space", **{ord(c): n for c, n in zip(CASES, names[2:])}}

    cmap.update({ord(c.lower()): cmap[ord(c)] for c in CASES if c.isalpha()})

    fb = FontBuilder(UPEM, isTTF=True)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap(cmap)

    glyphs = {}
    metrics = {}
    entries = {}

    for gid, (name, char) in enumerate(zip(names, [" ", " ", *CASES])):
        cfg = CASES.get(char, {"case": "Empty glyph / space"})
        kind = cfg.get("kind", "png") if gid >= 2 else "empty"

        inner = cfg.get("inner", [0, 0])
        outer = cfg.get("outer")
        lsb = cfg.get("lsb", outer[0] if outer else 0)

        artwork = "A" if kind == "dupe" else char
        rects = rectangles(artwork) if gid >= 2 else []
        color = cfg.get("color", [24, 91, 177, 255])

        pen = TTGlyphPen(None)

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
            pen.endPath()

        glyphs[name] = pen.glyph()
        advance = 2304 if gid >= 2 else 1024
        metrics[name] = (advance, lsb)

        has_contour = gid >= 2 and not cfg.get("no_contour", False)
        origin = [lsb, outer[1] if outer is not None else 0] if has_contour else [0, 0]

        entries[name] = dict(
            char=char,
            gid=gid,
            case=cfg["case"],
            kind=kind,
            offset=inner,
            outline=has_contour,
            bounds_origin=outer or [0, 0],
            placement_origin=origin,
            lsb=lsb,
            advance=advance,
            rectangles=rects,
            color=color,
            target=cfg.get("target"),
            strikes={},
        )

    fb.setupGlyf(glyphs)
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
            familyName="Sbix Test",
            styleName="Regular",
            uniqueFontIdentifier="SbixTest-Regular-1",
            fullName="Sbix Test",
            psName="SbixTest-Regular",
        )
    )

    fb.setupOS2(
        sTypoAscender=2048,
        sTypoDescender=-768,
        sTypoLineGap=0,
        usWinAscent=2304,
        usWinDescent=1024,
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
                e["strikes"][str(ppem)] = {"type": "dupe", "target": e["target"]}
                continue

            dx, dy = [round(v * scale) for v in e["offset"]]
            ox, oy = [v * ppem / UPEM for v in e["placement_origin"]]

            left = min(r[0] * scale - ox - dx for r in e["rectangles"])
            right = max((r[0] + r[2]) * scale - ox - dx for r in e["rectangles"])
            top = min(r[1] * scale + oy + dy for r in e["rectangles"])
            bottom = max((r[1] + r[3]) * scale + oy + dy for r in e["rectangles"])

            assert left >= 0 and bottom <= 0, (name, ppem, "uncompensatable offset")

            # Keep ordinary padding constant so V and ! isolate one axis each.
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

            e["strikes"][str(ppem)] = dict(
                type="png", offset=[dx, dy], width=w, height=h
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
    fb.save(FONT_PATH)

    return dict(
        family="Sbix Test",
        file="sbix.ttf",
        sha256=hashlib.sha256(FONT_PATH.read_bytes()).hexdigest(),
        description=(
            "One font, 27 per-glyph cases. "
            "Offsets up to ±32 strike pixels / ±512 font units; "
            "wrong signs move glyphs by up to 64 pixels at size 64."
        ),
        upem=UPEM,
        glyphs=entries,
        sample=SAMPLE,
        supported="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz! ",
        line_height=2.5,
        margin=1.5,
    )


if __name__ == "__main__":
    build()
    print(f"Generated {FONT_PATH}; verified placement at all three strikes.")
