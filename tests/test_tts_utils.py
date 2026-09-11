"""Offline tests for the TTS worker helpers (no torch, no GPU, no network)."""
import base64
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts_utils import (  # noqa: E402
    MAX_CHUNK_CHARS,
    clamp_float,
    clamp_int,
    encode_mp3,
    parse_reference,
    probe_duration,
    read_wav_bytes,
    reference_suffix,
    split_text,
    write_wav,
)


class TestSplitText:
    def test_keeps_every_character_of_a_short_text(self):
        text = "Доброе утро, Кирилл. Сегодня пятница."
        assert split_text(text) == [text]

    def test_splits_on_sentence_boundaries(self):
        chunks = split_text("Первое предложение. " * 40, max_chars=100)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_no_text_is_lost(self):
        text = "А" * 30 + ". " + "Б" * 900 + "! " + "В" * 50 + "?"
        chunks = split_text(text, max_chars=120)
        joined = " ".join(chunks)
        for token in ("А" * 30, "Б" * 900, "В" * 50):
            assert token in joined.replace(" ", "")
        assert all(len(c) <= 120 for c in chunks)

    def test_oversized_word_gets_hard_cut_not_dropped(self):
        chunks = split_text("х" * 700, max_chars=100)
        assert "".join(chunks) == "х" * 700
        assert all(len(c) <= 100 for c in chunks)

    def test_brief_sized_text_fits_in_a_few_chunks(self):
        # ~1000 chars is the size of the morning brief (~35 s of speech). The
        # real brief file is not readable from a CI runner, so the shape is
        # reproduced here with lorem-style Russian sentences.
        sentence = "Сегодня в Москве облачно и небольшой дождь. "
        text = (sentence * 24).strip()  # ~1080 chars, 24 sentences
        chunks = split_text(text, MAX_CHUNK_CHARS)
        assert 3 <= len(chunks) <= 10
        assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

    def test_no_chunk_can_exceed_the_decoder_budget(self):
        # 240 chars ~ 14 s of speech; the decoder stops at ~40 s per call.
        for size in (60, 120, 240, 900):
            chunks = split_text("слово " * 400, max_chars=size)
            assert all(len(c) <= size for c in chunks)

    def test_empty_input(self):
        assert split_text("") == []
        assert split_text(None) == []


class TestClamp:
    def test_clamp_float(self):
        assert clamp_float("0.7", 0.5, 0.0, 1.0) == 0.7
        assert clamp_float(None, 0.5, 0.0, 1.0) == 0.5
        assert clamp_float("nope", 0.5, 0.0, 1.0) == 0.5
        assert clamp_float(9, 0.5, 0.0, 1.0) == 1.0
        assert clamp_float(float("nan"), 0.5, 0.0, 1.0) == 0.5

    def test_clamp_int(self):
        assert clamp_int("12", 5, 0, 10) == 10
        assert clamp_int(None, 5, 0, 10) == 5
        assert clamp_int("abc", 5, 0, 10) == 5


class TestReference:
    def test_plain_base64(self):
        payload, mime = parse_reference(base64.b64encode(b"RIFFdata").decode())
        assert payload == b"RIFFdata"
        assert mime is None

    def test_data_uri(self):
        payload, mime = parse_reference("data:audio/wav;base64," + base64.b64encode(b"RIFF").decode())
        assert payload == b"RIFF"
        assert mime == "audio/wav"

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_reference("not base64!!!")
        with pytest.raises(ValueError):
            parse_reference("")
        with pytest.raises(ValueError):
            parse_reference(None if False else "data:audio/wav;base64,")

    def test_none_passes_through(self):
        assert parse_reference(None) == (None, None)

    def test_suffix_sniffing(self):
        assert reference_suffix(None, b"RIFF....") == ".wav"
        assert reference_suffix(None, b"ID3\x04") == ".mp3"
        assert reference_suffix(None, b"\xff\xfb\x90") == ".mp3"
        assert reference_suffix("audio/flac", b"zzz") == ".flac"
        assert reference_suffix(None, b"\x00\x01") == ".audio"


class TestAudioIo:
    def test_wav_roundtrip_and_clipping(self, tmp_path):
        samples = [0.0, 0.5, -0.5, 2.0, -2.0, 0.25]
        path = write_wav(tmp_path / "a.wav", samples, 24000)
        with wave.open(path, "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 24000
            assert handle.getnframes() == len(samples)
        raw = read_wav_bytes(path)
        assert raw.startswith(b"RIFF")

    def test_mp3_encode_when_ffmpeg_available(self, tmp_path):
        path = write_wav(tmp_path / "b.wav", [0.1] * 24000, 24000)
        data = encode_mp3(path)
        if data is None:
            pytest.skip("ffmpeg not available in this environment")
        assert (data.startswith(b"ID3")
                or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")), data[:4]
        duration = probe_duration(tmp_path / "b.wav")
        assert duration == pytest.approx(1.0, abs=0.05)
