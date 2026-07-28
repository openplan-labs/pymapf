"""The promo film's voice-over: timing, assembly, and graceful absence.

These tests are about the *contract* between narration and film -- that scene
lengths follow the spoken lines, and that the assembled track is exactly as long
as the film. Both are properties that hold without any audio tooling installed,
so they run everywhere. The handful that need espeak-ng and ffmpeg skip when
those are missing, which is also the behaviour the script itself must have: a
checkout with no audio stack renders a silent film rather than failing.
"""

import sys
from os import path

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(path.join(ROOT, "scripts"))

import pytest

from narration import LEAD_IN, TAIL, Narrator, espeak_binary, ffmpeg_binary

needs_audio = pytest.mark.skipif(
    not (espeak_binary() and ffmpeg_binary()),
    reason="needs espeak-ng and ffmpeg",
)

SCENES = [("scene_a", 4.0), ("scene_b", 8.0), ("scene_c", 3.0)]


def test_a_narrator_reports_what_it_is_missing():
    narrator = Narrator()
    if narrator.available:
        assert narrator.why_unavailable() == "missing "
    else:
        assert "espeak-ng" in narrator.why_unavailable() or "ffmpeg" in narrator.why_unavailable()
    narrator.cleanup()


def test_scene_durations_stretch_to_fit_the_line():
    """The film follows the narration, not the other way round."""
    narrator = Narrator()
    clips = {"scene_a": ("", 9.0), "scene_b": ("", 1.0), "scene_c": ("", 0.0)}
    durations = narrator.scene_durations(SCENES, clips)

    # Long line: the scene grows to fit it, with room either side.
    assert durations[0] == pytest.approx(9.0 + LEAD_IN + TAIL)
    # Short line: the animation's own minimum wins.
    assert durations[1] == pytest.approx(8.0)
    # No line at all: the minimum, untouched.
    assert durations[2] == pytest.approx(3.0)
    narrator.cleanup()


def test_a_scene_without_a_line_keeps_its_own_length():
    narrator = Narrator()
    assert narrator.scene_durations(SCENES, {}) == [4.0, 8.0, 3.0]
    narrator.cleanup()


def test_no_line_is_ever_cut_off_by_its_scene():
    """The property that matters: every line finishes before its scene ends."""
    narrator = Narrator()
    clips = {name: ("", length) for name, length in
             [("scene_a", 12.0), ("scene_b", 0.4), ("scene_c", 3.0)]}
    durations = narrator.scene_durations(SCENES, clips)
    for (name, _), duration in zip(SCENES, durations):
        assert LEAD_IN + clips[name][1] <= duration + 1e-9
    narrator.cleanup()


@needs_audio
def test_synthesis_produces_audible_clips():
    narrator = Narrator()
    try:
        clips = narrator.synthesise({"one": "Multi agent path finding.", "two": ""})
        assert set(clips) == {"one"}          # empty lines are skipped
        wav, seconds = clips["one"]
        assert path.exists(wav)
        assert 0.5 < seconds < 10.0
    finally:
        narrator.cleanup()


@needs_audio
def test_the_track_is_exactly_as_long_as_the_film():
    """Built by concatenation so audio and video cannot drift apart."""
    import wave

    narrator = Narrator()
    try:
        lines = {"scene_a": "The first line.", "scene_b": "The second line."}
        clips = narrator.synthesise(lines)
        durations = narrator.scene_durations(SCENES, clips)
        track = narrator.build_track(SCENES, durations, clips)

        with wave.open(track, "rb") as handle:
            length = handle.getnframes() / float(handle.getframerate())
        assert length == pytest.approx(sum(durations), abs=0.15)
    finally:
        narrator.cleanup()


@needs_audio
def test_the_promo_script_imports_and_its_narration_covers_every_scene():
    """A scene added without a line would silently play under nothing."""
    import make_promo

    scene_names = {scene.__name__ for scene, _ in make_promo.SCENES}
    assert scene_names == set(make_promo.NARRATION)
    assert all(text.strip() for text in make_promo.NARRATION.values())


def test_scene_durations_are_positive_and_finite():
    import make_promo

    assert all(duration > 0 for _, duration in make_promo.SCENES)
    assert len(make_promo.SCENES) == len(make_promo.NARRATION)


@needs_audio
def test_muxing_refuses_a_track_that_would_truncate_the_film():
    """`-shortest` stops at whichever stream ends first.

    A short narration track therefore does not leave silence at the end -- it
    cuts the film off, and the result still plays. A sample-rate mismatch
    upstream once turned a 101.7-second film into 57 seconds with its last four
    scenes missing, and nothing reported an error. This makes it a crash.
    """
    import subprocess
    import tempfile

    narrator = Narrator()
    try:
        directory = tempfile.mkdtemp()
        film = path.join(directory, "film.mp4")
        # Ten seconds of video, against a two-second track.
        subprocess.run(
            [narrator.ffmpeg, "-y", "-f", "lavfi", "-i",
             "color=c=black:s=160x90:d=10", "-c:v", "libx264", "-t", "10", film],
            check=True, capture_output=True,
        )
        short = narrator._silence(2.0, "far-too-short")

        with pytest.raises(RuntimeError, match="truncate"):
            narrator.mux(film, short, path.join(directory, "out.mp4"))
    finally:
        narrator.cleanup()


@needs_audio
def test_a_matching_track_muxes_and_keeps_the_full_length():
    import subprocess
    import tempfile

    narrator = Narrator()
    try:
        directory = tempfile.mkdtemp()
        film = path.join(directory, "film.mp4")
        subprocess.run(
            [narrator.ffmpeg, "-y", "-f", "lavfi", "-i",
             "color=c=black:s=160x90:d=6", "-c:v", "libx264", "-t", "6", film],
            check=True, capture_output=True,
        )
        track = narrator._silence(6.0, "just-right")
        output = narrator.mux(film, track, path.join(directory, "out.mp4"))
        assert narrator.probe_seconds(output) == pytest.approx(6.0, abs=0.3)
    finally:
        narrator.cleanup()


@needs_audio
def test_clips_are_normalised_to_one_sample_rate():
    """espeak renders at 22050; the pipeline runs at 44100."""
    import wave

    narrator = Narrator()
    try:
        wav, _ = narrator.say("A short line.", "rate-check")
        with wave.open(wav, "rb") as handle:
            assert handle.getframerate() == 44100
            assert handle.getnchannels() == 1
    finally:
        narrator.cleanup()
