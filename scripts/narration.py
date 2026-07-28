"""Voice-over for the promo film: synthesis, timing and the final mux.

The reason this module exists rather than a hand-timed audio track is sync.
Scene lengths and narration lengths have to agree, and hand-tuning both is how a
film ends up with a sentence still playing over the next cut. So the dependency
runs one way: **the narration is synthesised first, and the scene durations are
derived from it**. A scene lasts as long as its line takes to say, plus lead-in
and tail, floored at whatever the animation itself needs to read.

Everything is offline and deterministic. ``espeak-ng`` is a formant synthesiser
-- it will never be mistaken for a person -- but it needs no model download and
no network, which is what makes this reproducible on a fresh checkout. The
post-processing chain does what it can: a band-pass to cut the buzz at the
extremes, gentle compression, a short room reflection so it does not sound
recorded inside a tin, and loudness normalisation to broadcast level.

    from narration import Narrator
    narrator = Narrator()
    clips = narrator.synthesise(LINES)      # {scene_name: (path, seconds)}
    narrator.mux(video, clips, timeline, output)
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import wave
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Narrator", "ffmpeg_binary", "espeak_binary"]

# Sample rate for the whole pipeline. espeak-ng renders at 22050, so every clip
# is resampled to this on the way out of say() -- not at the end. The concat
# demuxer reinterprets later inputs using the *first* file's parameters, so a
# mismatch here does not fail, it silently retimes the narration.
RATE = 44100

# The voice. Chosen by ear over the variants espeak-ng ships: +m3 is the least
# nasal of the male set, and dropping the pitch and rate from the defaults takes
# most of the chatter out of it.
VOICE = "en-us+m3"
WORDS_PER_MINUTE = 158
PITCH = 33
AMPLITUDE = 190
WORD_GAP = 3  # units of 10 ms inserted between words

# A little air on each side of every line, so a sentence never butts against a
# cut and the next one does not start before the scene has drawn.
LEAD_IN = 0.55
TAIL = 0.75


def ffmpeg_binary() -> Optional[str]:
    """Prefer a system ffmpeg, fall back to the one imageio ships."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def espeak_binary() -> Optional[str]:
    return shutil.which("espeak-ng") or shutil.which("espeak")


def _wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


class Narrator:
    """Synthesises narration and muxes it onto a rendered film.

    Args:
        voice, rate, pitch: passed through to espeak-ng.
        directory: where the intermediate wavs go. A temporary directory is
            created and cleaned up if this is not given.
    """

    def __init__(
        self,
        voice: str = VOICE,
        rate: int = WORDS_PER_MINUTE,
        pitch: int = PITCH,
        directory: Optional[str] = None,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.espeak = espeak_binary()
        self.ffmpeg = ffmpeg_binary()
        self._temporary = None
        if directory is None:
            self._temporary = tempfile.mkdtemp(prefix="pymapf-vo-")
            directory = self._temporary
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    @property
    def available(self) -> bool:
        """Can a voice-over be produced at all?"""
        return bool(self.espeak and self.ffmpeg)

    def why_unavailable(self) -> str:
        missing = []
        if not self.espeak:
            missing.append("espeak-ng (apt-get install espeak-ng)")
        if not self.ffmpeg:
            missing.append("ffmpeg (pip install imageio-ffmpeg)")
        return "missing " + " and ".join(missing)

    # ------------------------------------------------------------------
    # synthesis
    # ------------------------------------------------------------------
    def say(self, text: str, name: str) -> Tuple[str, float]:
        """Render one line to a wav, returning its path and duration.

        The clip is resampled to :data:`RATE` before it is returned, and that
        step is not optional. espeak-ng renders at 22050 Hz while the silence
        this module generates is at 44100, and ffmpeg's concat *demuxer* takes
        its stream parameters from the first input and reinterprets the rest --
        so every spoken clip would play at half its length, and the narration
        would slide progressively out of sync with the film. Normalising each
        clip on the way out makes the mismatch impossible rather than
        remembered.
        """
        raw = os.path.join(self.directory, "%s-raw.wav" % name)
        subprocess.run(
            [
                self.espeak,
                "-v", self.voice,
                "-s", str(self.rate),
                "-p", str(self.pitch),
                "-a", str(AMPLITUDE),
                "-g", str(WORD_GAP),
                "-w", raw,
                text,
            ],
            check=True,
            capture_output=True,
        )
        path = os.path.join(self.directory, "%s.wav" % name)
        self._run([
            self.ffmpeg, "-y", "-i", raw,
            "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", path,
        ])
        os.remove(raw)
        return path, _wav_seconds(path)

    def synthesise(self, lines: Dict[str, str]) -> Dict[str, Tuple[str, float]]:
        """Render every line. Returns ``{name: (path, seconds)}``."""
        clips = {}
        for name, body in lines.items():
            if not body:
                continue
            clips[name] = self.say(body, name)
        return clips

    def scene_durations(
        self,
        scenes: Sequence[Tuple[str, float]],
        clips: Dict[str, Tuple[str, float]],
    ) -> List[float]:
        """How long each scene must last to fit its line.

        The animation's own minimum wins when it is longer -- a scene that needs
        four seconds to draw does not get cut to two because the sentence is
        short.
        """
        durations = []
        for name, minimum in scenes:
            spoken = clips.get(name, (None, 0.0))[1]
            durations.append(max(minimum, spoken + LEAD_IN + TAIL if spoken else 0.0))
        return durations

    # ------------------------------------------------------------------
    # assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _rate_of(path: str) -> int:
        with wave.open(path, "rb") as handle:
            return handle.getframerate()

    def _silence(self, seconds: float, name: str) -> str:
        path = os.path.join(self.directory, "%s-pad.wav" % name)
        frames = max(1, int(seconds * RATE))
        with wave.open(path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(RATE)
            handle.writeframes(struct.pack("<%dh" % frames, *([0] * frames)))
        return path

    def build_track(
        self,
        scenes: Sequence[Tuple[str, float]],
        durations: Sequence[float],
        clips: Dict[str, Tuple[str, float]],
    ) -> str:
        """Lay every clip onto a silent bed at its scene's start time.

        Built by concatenation rather than mixing: each scene contributes
        lead-in silence, its line, then silence out to the scene's end, so the
        track is exactly as long as the film by construction and cannot drift.
        """
        pieces = []
        for (name, _), duration in zip(scenes, durations):
            clip = clips.get(name)
            if clip is None:
                pieces.append(self._silence(duration, "%s-only" % name))
                continue
            path, spoken = clip
            pieces.append(self._silence(LEAD_IN, "%s-in" % name))
            pieces.append(path)
            remaining = max(0.02, duration - LEAD_IN - spoken)
            pieces.append(self._silence(remaining, "%s-out" % name))

        # Every piece must already agree on rate and channels -- see say().
        rates = {self._rate_of(piece) for piece in pieces}
        if len(rates) != 1:
            raise RuntimeError(
                "narration pieces disagree on sample rate (%s); the concat "
                "demuxer would silently retime them" % sorted(rates)
            )

        listing = os.path.join(self.directory, "concat.txt")
        with open(listing, "w") as handle:
            for piece in pieces:
                handle.write("file '%s'\n" % os.path.abspath(piece))

        raw = os.path.join(self.directory, "voice-raw.wav")
        self._run([
            self.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listing,
            "-ar", str(RATE), "-ac", "1", raw,
        ])

        # Make a formant synthesiser listenable: trim the buzz above and below
        # the voice band, even the levels out, put it in a small room, and
        # normalise to a sane broadcast loudness.
        polished = os.path.join(self.directory, "voice.wav")
        self._run([
            self.ffmpeg, "-y", "-i", raw,
            "-af", (
                "highpass=f=95,lowpass=f=7600,"
                "equalizer=f=250:t=q:w=1.2:g=2.5,"
                "equalizer=f=2600:t=q:w=1.6:g=-3,"
                "acompressor=threshold=0.10:ratio=4:attack=8:release=200,"
                "aecho=0.86:0.85:26:0.14,"
                "loudnorm=I=-17:TP=-2:LRA=11"
            ),
            "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", polished,
        ])
        return polished

    def probe_seconds(self, path: str) -> Optional[float]:
        """Duration of a media file, read back from ffmpeg's own report."""
        report = subprocess.run(
            [self.ffmpeg, "-i", path], capture_output=True
        ).stderr.decode("utf-8", "replace")
        for line in report.splitlines():
            if "Duration:" in line:
                stamp = line.split("Duration:")[1].split(",")[0].strip()
                try:
                    hours, minutes, seconds = stamp.split(":")
                    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                except ValueError:
                    return None
        return None

    def mux(self, video: str, track: str, output: str, tolerance: float = 0.5) -> str:
        """Combine the silent render with the narration track.

        The length check is not paranoia. ``-shortest`` stops at whichever
        stream ends first, so a track that comes out short does not produce
        quiet gaps at the end -- it **truncates the film**, silently, and the
        result still plays. That is exactly what a sample-rate mismatch upstream
        caused here once: a 101.7-second film delivered as 57 seconds with its
        last four scenes simply gone, and nothing anywhere reporting an error.
        Comparing the two durations before muxing turns that class of mistake
        into a crash instead of a bad deliverable.
        """
        film = self.probe_seconds(video)
        voice = self.probe_seconds(track)
        if film and voice and abs(film - voice) > tolerance:
            raise RuntimeError(
                "narration track is %.2fs but the film is %.2fs; muxing with "
                "-shortest would truncate it to %.2fs"
                % (voice, film, min(film, voice))
            )
        self._run([
            self.ffmpeg, "-y", "-i", video, "-i", track,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", output,
        ])
        return output

    def _run(self, command: Sequence[str]) -> None:
        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                "%s failed:\n%s"
                % (command[0], result.stderr.decode("utf-8", "replace")[-2000:])
            )

    def cleanup(self) -> None:
        if self._temporary:
            shutil.rmtree(self._temporary, ignore_errors=True)
            self._temporary = None
