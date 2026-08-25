import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import srt  # noqa: E402

SAMPLE = """﻿1
00:00:02,000 --> 00:00:04,500
Hello there.

2
00:00:05,000 --> 00:00:07,250
<i>Don't move.</i>

3
00:00:08,000 --> 00:00:10,000
- Who's there?
- It's me.
"""


def test_parse_basic():
    cues = srt.parse(SAMPLE)
    assert len(cues) == 3
    assert cues[0].start == 2000 and cues[0].end == 4500
    assert cues[2].text == "- Who's there?\n- It's me."


def test_roundtrip_preserves_timing():
    cues = srt.parse(SAMPLE)
    again = srt.parse(srt.dumps(cues))
    assert [(c.start, c.end, c.text) for c in cues] == [(c.start, c.end, c.text) for c in again]


def test_parse_without_indices_and_crlf():
    text = "00:00:01,000 --> 00:00:02,000\r\nFirst\r\n\r\n00:00:03,000 --> 00:00:04,000\r\nSecond\r\n"
    cues = srt.parse(text)
    assert [c.text for c in cues] == ["First", "Second"]


def test_parse_dot_milliseconds_and_short_fraction():
    cues = srt.parse("1\n00:00:01.5 --> 00:00:02.25\nHi\n")
    assert cues[0].start == 1500 and cues[0].end == 2250


def test_timestamp_formatting():
    assert srt.ms_to_ts(3_723_456) == "01:02:03,456"
    assert srt.ms_to_ts(-5) == "00:00:00,000"


def test_undress_and_dress_italics():
    body, skin = srt.undress("<i>Don't move.</i>")
    assert body == "Don't move." and skin.italic
    assert srt.dress("لا تتحرك.", skin) == "<i>لا تتحرك.</i>"


def test_undress_keeps_ass_position_tag():
    body, skin = srt.undress("{\\an8}TOKYO, 1998")
    assert body == "TOKYO, 1998"
    assert srt.dress("طوكيو، 1998", skin).startswith("{\\an8}")


def test_wrap_splits_long_line_in_two_balanced_parts():
    line = "و" * 30 + " " + "ب" * 30
    out = srt.wrap(line, max_line=42, max_lines=2)
    assert out.count("\n") == 1
    assert all(len(part) <= 42 for part in out.split("\n"))


def test_wrap_leaves_dialogue_dashes_alone():
    text = "- من هناك؟\n- أنا."
    assert srt.wrap(text) == text


def test_wrap_short_text_untouched():
    assert srt.wrap("مرحبا") == "مرحبا"


def test_decode_cp1256_arabic():
    data = "مرحبا".encode("cp1256")
    assert srt.decode(data) == "مرحبا"


def test_decode_strips_bom():
    assert srt.decode("﻿hi".encode("utf-8")) == "hi"


def test_looks_arabic():
    assert srt.looks_arabic(srt.parse("1\n00:00:01,000 --> 00:00:02,000\nمرحبا بك\n"))
    assert not srt.looks_arabic(srt.parse(SAMPLE))


def test_strip_hi():
    assert srt.strip_hi("[door creaks]\nMARGE: Get down!") == "Get down!"


def test_malformed_block_is_skipped_not_fatal():
    text = SAMPLE + "\nWEBVTT junk\n\n4\n00:00:11,000 --> 00:00:12,000\nEnd\n"
    cues = srt.parse(text)
    assert len(cues) == 4 and cues[-1].text == "End"
