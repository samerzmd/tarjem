"""CLI argument handling. Bad paths must explain themselves, not traceback."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli import main  # noqa: E402


def test_missing_file_exits_cleanly(tmp_path, capsys):
    assert main([str(tmp_path / "nope.srt")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_an_unfilled_placeholder_says_so(tmp_path, capsys):
    assert main([str(tmp_path / "<episode>.mkv"), "--from-video"]) == 2
    err = capsys.readouterr().err
    assert "no such file" in err and "placeholder" in err


def test_a_directory_is_rejected(tmp_path, capsys):
    assert main([str(tmp_path)]) == 2
    assert "not a file" in capsys.readouterr().err


def test_analyze_reports_repeats_without_calling_a_model(tmp_path, capsys):
    srt_file = tmp_path / "s.en.srt"
    body = "\n\n".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\n"
        + ("Yes." if i % 2 else f"Unique line {i}.")
        for i in range(1, 11)
    )
    srt_file.write_text(body, encoding="utf-8")

    assert main([str(srt_file), "--analyze"]) == 0
    err = capsys.readouterr().err
    assert "total cues" in err and "repeats" in err
