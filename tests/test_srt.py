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


# -- stacked duplicates ----------------------------------------------------
# An .ass with several styled layers, flattened to SRT by ffmpeg, repeats each
# line on top of itself. One real episode: 5,390 cues, 346 unique lines.

def _c(i, start, end, text):
    return srt.Cue(i, start, end, text)


def test_overlapping_duplicates_are_counted():
    cues = [_c(1, 0, 2000, "Hello."), _c(2, 100, 2100, "Hello."), _c(3, 200, 2200, "Hello.")]
    assert srt.overlapping_duplicates(cues) == 2


def test_a_line_repeated_later_is_not_a_stacked_duplicate():
    cues = [_c(1, 0, 2000, "No!"), _c(2, 30000, 31000, "No!")]
    assert srt.overlapping_duplicates(cues) == 0


def test_collapse_merges_stacked_copies_into_one_spanning_cue():
    cues = [_c(1, 1000, 3000, "Hello."), _c(2, 1200, 3500, "Hello."), _c(3, 900, 2000, "Hello.")]
    out, merged = srt.collapse_duplicates(cues)
    assert merged == 2 and len(out) == 1
    assert out[0].start == 900 and out[0].end == 3500


def test_collapse_leaves_a_genuine_repeat_alone():
    cues = [_c(1, 0, 2000, "No!"), _c(2, 30000, 31000, "No!")]
    out, merged = srt.collapse_duplicates(cues)
    assert merged == 0 and len(out) == 2


def test_collapse_ignores_markup_when_matching():
    cues = [_c(1, 0, 2000, "<i>Hello.</i>"), _c(2, 100, 2100, "Hello.")]
    out, merged = srt.collapse_duplicates(cues)
    assert merged == 1 and out[0].text == "<i>Hello.</i>"


def test_collapse_keeps_different_lines_that_overlap():
    cues = [_c(1, 0, 2000, "Hello."), _c(2, 100, 2100, "Goodbye.")]
    out, merged = srt.collapse_duplicates(cues)
    assert merged == 0 and len(out) == 2


# -- markup that ffmpeg generates from .ass --------------------------------
# Seen in real output: {\an8} buried inside a <font> wrapper, and a font tag
# split across a line break by the wrapper.

def test_an_ass_tag_buried_inside_markup_is_lifted_to_the_front():
    body, skin = srt.undress(r'<font size="72"><b>{\an8}Please try again.</b></font>')
    assert r"{\an8}" not in body
    assert skin.ass_prefix == r"{\an8}"
    assert srt.dress("حاول مجددًا.", skin).startswith(r"{\an8}")


def test_several_ass_tags_are_all_lifted_and_deduplicated():
    body, skin = srt.undress(r"{\an8}{\pos(10,20)}Hi {\an8}there")
    assert body == "Hi there"
    assert skin.ass_prefix == r"{\an8}{\pos(10,20)}"


def test_wrapping_never_splits_a_font_tag():
    line = '<font size="20" color="#ffff00">' + "و" * 40 + " " + "ب" * 40 + "</font>"
    out = srt.wrap(line, max_line=42, max_lines=2)
    for part in out.split("\n"):
        # a broken tag shows as an unbalanced angle bracket on one line
        assert part.count("<") == part.count(">"), part


def test_wrapping_keeps_a_long_tag_intact_even_when_it_exceeds_the_width():
    line = '<font face="Some Very Long Font Name" size="72">Hi</font>'
    out = srt.wrap(line, max_line=20, max_lines=2)
    assert '<font face="Some Very Long Font Name" size="72">' in out


def test_font_tags_are_stripped_but_emphasis_is_kept():
    text = '<font face="Vesta" size="72" color="#ffeeee"><b><i>Hello</i></b></font>'
    assert srt.strip_style_tags(text) == "<b><i>Hello</i></b>"


def test_stripping_style_tags_leaves_plain_text_alone():
    assert srt.strip_style_tags("Just words.") == "Just words."
