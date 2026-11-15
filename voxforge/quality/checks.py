"""Quality checks for one generated utterance (VF-018).

Pure functions: ASR words (from any recogniser) + the audio file + the text the engine
was asked to say -> an explainable report. No model, no GPU, no UI, so Playground, Studio
and tests share exactly the same rules.

Two kinds of evidence are kept apart:

- **audio** (measured on the waveform, reliable): silent output, duration per character,
  speaking rate, long inner silences, leading/trailing silence, level, clipping, abrupt end;
- **asr** (a recogniser's opinion, *not* proof of a TTS error): missing, repeated, extra
  and different words from a word-level alignment of the transcript against the text.

Every finding carries ``severity`` (major/minor) and ``confidence`` (high/medium/low).
ASR findings are downgraded when the recogniser itself was unsure (low word probability)
and only become high confidence when the audio agrees (e.g. missing last words AND the
clip is too short or ends abruptly). The verdict keeps "suspect" separate from "problem":

- ``ok``: nothing beyond low-confidence minor notes;
- ``suspect``: listen and decide; nothing is changed automatically;
- ``problem``: at least one major finding with high confidence; only this verdict allows
  automatic alternatives (``retake_recommended``).
"""

from __future__ import annotations

import difflib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from voxforge.benchmark.metrics import duration_check, normalize_for_cer, score_transcript
from voxforge.playground.rhythm import (
    FRAME_SECONDS,
    frame_db,
    pauses,
    read_pcm16_mono,
    split_sentences,
    syllables,
    voiced_mask,
)

CHECKER_VERSION = "vf018-3"
SCHEMA_VERSION = 1

# Eşikler (gerekçeleri docs/QUALITY_DIRECTOR.md'de).
ASR_UNSURE_PROB = 0.6  # kelime olasılığı bunun altındaysa ASR emin değil
ASR_GLOBAL_UNSURE_PROB = 0.65  # transkript ortalaması bunun altındaysa tüm ASR bulguları zayıf
NEAR_WORD_SIMILARITY = 0.75  # tek kelime, yakın yazım ("benle"/"benimle"): telaffuz/ASR belirsizliği
HIGH_CER = 0.15  # emin bir ASR ile toplam karakter hata oranı
SHORT_FUNCTION_WORD_CHARS = 3  # "ve", "de", "bir": tek başına düşmesi çoğu zaman ASR
RATE_FAST = 9.5  # hece/sn; VoxCPM2 mimo ~6-8 (VF-016 ölçümü)
RATE_SLOW = 3.5
MIN_VOICED_SECONDS = 0.25
LONG_SILENCE_SECONDS = 1.0  # duraklama istenmemişse cümle içi en uzun sessizlik
LONG_SILENCE_MARGIN = 0.6  # istenen duraklamanın üstüne tolerans
VERY_LONG_SILENCE_SECONDS = 2.5
LEADING_SILENCE_SECONDS = 0.8
TRAILING_SILENCE_SECONDS = 1.2
ABRUPT_END_SECONDS = 0.02
LOW_LEVEL_DBFS = -32.0
CLIP_LEVEL = 0.999
CLIP_FRACTION = 0.001
WORD_MARGIN_SECONDS = 0.1  # kelime zaman damgası çevresindeki tolerans
UNHEARD_SPEECH_SECONDS = 0.6  # ASR'ın hiçbir kelimeye bağlamadığı sesli süre
UNHEARD_SPEECH_HIGH_SECONDS = 1.0

SEVERITIES = ("major", "minor")
CONFIDENCES = ("high", "medium", "low")
VERDICT_LABELS = {
    "ok": "Başarılı",
    "suspect": "Şüpheli (dinleyip karar verin)",
    "problem": "Sorunlu (yüksek güven)",
    "unchecked": "Denetlenmedi",
}
CODE_LABELS = {
    "silent_output": "Sessiz/boş çıktı",
    "asr_empty": "ASR konuşma duymadı",
    "missing_words": "Eksik kelime",
    "truncated": "Kesilme şüphesi",
    "repeated_words": "Tekrarlanan kelime",
    "extra_words": "Metin dışı kelime",
    "wrong_word": "Farklı kelime",
    "pronunciation": "Telaffuz şüphesi",
    "too_short": "Aşırı kısa",
    "too_long": "Aşırı uzun",
    "rate_fast": "Anormal hızlı",
    "rate_slow": "Anormal yavaş",
    "long_silence": "Uzun sessizlik",
    "leading_silence": "Başta uzun sessizlik",
    "trailing_silence": "Sonda uzun sessizlik",
    "low_level": "Düşük ses seviyesi",
    "clipping": "Kırpılma (clipping)",
    "abrupt_end": "Ani bitiş",
    "asr_unsure": "ASR güveni düşük",
    "identity_drift": "Ses kimliği kayması şüphesi",
    "unheard_speech": "Tanınmayan konuşma",
    "high_cer": "Yüksek metin hatası",
}


@dataclass
class Finding:
    code: str
    severity: str
    confidence: str
    source: str  # audio | asr | asr+audio | speaker
    message: str
    sentence: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return CODE_LABELS.get(self.code, self.code)


@dataclass(frozen=True)
class Token:
    text: str
    prob: float | None = None
    start: float | None = None
    end: float | None = None
    sentence: int | None = None


# ---- tokens and alignment -----------------------------------------------------------------
def reference_tokens(spoken_text: str) -> tuple[list[Token], list[str]]:
    """Normalised words of the expected text, each tagged with its sentence index."""
    sentences = split_sentences(spoken_text) or [spoken_text]
    tokens = [
        Token(word, sentence=index)
        for index, sentence in enumerate(sentences)
        for word in normalize_for_cer(sentence).split()
    ]
    return tokens, sentences


def hypothesis_tokens(words: Iterable[dict[str, Any]]) -> list[Token]:
    """ASR words -> normalised tokens ("5" becomes "beş"; each piece keeps the word's probability)."""
    tokens: list[Token] = []
    for word in words:
        prob = word.get("prob", word.get("probability"))
        for piece in normalize_for_cer(str(word.get("word", ""))).split():
            tokens.append(Token(piece, None if prob is None else float(prob), word.get("start"), word.get("end")))
    return tokens


def align(reference: list[str], hypothesis: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Word-level Levenshtein with backtrace: ("equal"|"sub"|"del"|"ins", ref index, hyp index)."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            same = reference[i - 1] == hypothesis[j - 1]
            cost[i][j] = min(cost[i - 1][j - 1] + (0 if same else 1), cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1):
            ops.append(("equal" if reference[i - 1] == hypothesis[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def error_spans(ops: list[tuple[str, int | None, int | None]]) -> list[list[tuple[str, int | None, int | None]]]:
    spans: list[list[tuple[str, int | None, int | None]]] = []
    current: list[tuple[str, int | None, int | None]] = []
    for op in ops:
        if op[0] == "equal":
            if current:
                spans.append(current)
                current = []
        else:
            current.append(op)
    if current:
        spans.append(current)
    return spans


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _downgrade(confidence: str, steps: int = 1) -> str:
    return CONFIDENCES[min(len(CONFIDENCES) - 1, CONFIDENCES.index(confidence) + steps)]


def _upgrade(confidence: str) -> str:
    return CONFIDENCES[max(0, CONFIDENCES.index(confidence) - 1)]


# ---- audio ------------------------------------------------------------------------------------
@dataclass
class AudioFacts:
    duration: float
    voiced_seconds: float
    leading_silence: float
    trailing_silence: float
    pauses: list[tuple[float, float]]
    mean_dbfs: float | None
    peak_dbfs: float | None
    clip_fraction: float | None
    readable: bool = True
    mask: list[bool] = field(default_factory=list)  # 20 ms çerçeve: konuşma var mı


def audio_facts(path: Path) -> AudioFacts:
    try:
        samples, rate = read_pcm16_mono(Path(path))
    except (ValueError, OSError, EOFError):  # 16-bit olmayan/bozuk WAV: ses denetimleri atlanır
        return _unreadable()
    duration = len(samples) / float(rate) if rate else 0.0
    levels = frame_db(samples, rate)
    mask = voiced_mask(levels)
    peak = max((abs(v) for v in samples), default=0.0)
    if True in mask:
        first = mask.index(True)
        last = len(mask) - 1 - mask[::-1].index(True)
        voiced_levels = [level for level, on in zip(levels, mask) if on]
        mean_db = round(10 * math.log10(sum(10 ** (lv / 10) for lv in voiced_levels) / len(voiced_levels)), 2)
    else:
        first, last, mean_db = len(mask), len(mask) - 1, None
    # Sessizlik eşiği tepe seviyeye görelidir; neredeyse sessiz bir dosyada "konuşma" sayılmasın.
    voiced_seconds = sum(mask) * FRAME_SECONDS if peak > 10 ** (-50 / 20) else 0.0
    return AudioFacts(
        duration=round(duration, 3),
        voiced_seconds=round(voiced_seconds, 3),
        leading_silence=round(first * FRAME_SECONDS, 3),
        trailing_silence=round(max(0.0, duration - (last + 1) * FRAME_SECONDS), 3),
        pauses=pauses(mask),
        mean_dbfs=mean_db,
        peak_dbfs=round(20 * math.log10(peak), 2) if peak > 0 else None,
        clip_fraction=round(sum(1 for v in samples if abs(v) >= CLIP_LEVEL) / len(samples), 6) if samples else None,
        mask=mask if voiced_seconds else [],
    )


def _unreadable() -> AudioFacts:
    return AudioFacts(0.0, 0.0, 0.0, 0.0, [], None, None, None, readable=False)


# ---- report -----------------------------------------------------------------------------------
def evaluate(
    spoken_text: str,
    audio_path: Path,
    words: list[dict[str, Any]] | None = None,
    asr_text: str | None = None,
    intended_pauses: list[float] | None = None,
    asr_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quality report for one utterance.

    ``words`` are ASR words with ``word``/``start``/``end``/``prob``; ``None`` means the ASR
    was not run (audio checks only, verdict never better than what audio alone shows).
    ``intended_pauses`` are the pauses the engine inserted on purpose (narration pacing), so
    they are not reported as long silences and are excluded from the speaking rate.
    """
    facts = audio_facts(Path(audio_path))
    ref_tokens, sentences = reference_tokens(spoken_text)
    intended = [float(p) for p in (intended_pauses or []) if p]
    findings: list[Finding] = []
    metrics: dict[str, Any] = {"duration": facts.duration, "voiced_seconds": facts.voiced_seconds}

    findings += _audio_findings(facts, spoken_text, intended, metrics)
    asr_checked = words is not None
    if asr_checked:
        audio_codes = {f.code for f in findings}
        findings += _asr_findings(ref_tokens, hypothesis_tokens(words or []), facts, metrics, asr_text or "", spoken_text, audio_codes)

    verdict = _verdict(findings)
    problem_sentences = sorted({f.sentence for f in findings if f.sentence is not None and _counts(f)})
    return {
        "schema_version": SCHEMA_VERSION,
        "checker_version": CHECKER_VERSION,
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "retake_recommended": verdict == "problem",
        "asr_checked": asr_checked,
        "asr": {**(asr_info or {}), "text": asr_text} if asr_checked else {**(asr_info or {}), "text": None},
        "metrics": metrics,
        "sentences": sentences,
        "problem_sentences": problem_sentences,
        "findings": [{**asdict(f), "label": f.label} for f in findings],
    }


def _counts(finding: Finding) -> bool:
    """Findings that make an utterance at least suspect.

    Major findings always count. Minor ones count only when measured on the audio with at
    least medium confidence: a minor ASR difference (near word such as "benle"/"benimle") is
    a note, because Whisper often normalises colloquial forms (VF-018 GPU check).
    """
    if finding.severity == "major":
        return True
    return finding.source in ("audio", "speaker") and finding.confidence in ("high", "medium")


def _verdict(findings: list[Finding]) -> str:
    if any(f.severity == "major" and f.confidence == "high" for f in findings):
        return "problem"
    if any(_counts(f) for f in findings):
        return "suspect"
    return "ok"


def _audio_findings(facts: AudioFacts, spoken_text: str, intended: list[float], metrics: dict[str, Any]) -> list[Finding]:
    out: list[Finding] = []
    if not facts.readable:
        out.append(Finding("silent_output", "minor", "low", "audio", "Ses dosyası 16-bit PCM değil; ses denetimleri atlandı."))
        return out
    if facts.voiced_seconds < MIN_VOICED_SECONDS:
        out.append(Finding("silent_output", "major", "high", "audio", f"Çıktıda konuşma yok denecek kadar az ({facts.voiced_seconds} sn sesli).",
                           evidence={"voiced_seconds": facts.voiced_seconds, "peak_dbfs": facts.peak_dbfs}))
        return out
    check = duration_check(spoken_text, facts.duration)
    metrics.update(seconds_per_char=check["seconds_per_char"])
    if check["too_short"]:
        out.append(Finding("too_short", "major", "medium", "audio",
                           f"Metne göre çok kısa ({check['seconds_per_char']} sn/karakter); kelime düşmüş veya kesilmiş olabilir.",
                           evidence={"seconds_per_char": check["seconds_per_char"]}))
    if check["too_long"]:
        out.append(Finding("too_long", "major", "medium", "audio",
                           f"Metne göre çok uzun ({check['seconds_per_char']} sn/karakter); tekrar, metin dışı ses veya uzun sessizlik olabilir.",
                           evidence={"seconds_per_char": check["seconds_per_char"]}))
    speech = facts.duration - facts.leading_silence - facts.trailing_silence - sum(intended)
    count = syllables(normalize_for_cer(spoken_text))
    rate = round(count / speech, 2) if speech > 0.2 else None
    metrics.update(syllables=count, rate=rate, leading_silence=facts.leading_silence, trailing_silence=facts.trailing_silence)
    if rate is not None and count >= 6:
        if rate > RATE_FAST:
            out.append(Finding("rate_fast", "minor", "medium", "audio", f"Konuşma hızı {rate} hece/sn (> {RATE_FAST}); aceleci okunuyor olabilir.", evidence={"rate": rate}))
        elif rate < RATE_SLOW:
            out.append(Finding("rate_slow", "minor", "medium", "audio", f"Konuşma hızı {rate} hece/sn (< {RATE_SLOW}); uzatma veya boşluk olabilir.", evidence={"rate": rate}))
    allowed = (max(intended) + LONG_SILENCE_MARGIN) if intended else LONG_SILENCE_SECONDS
    lengths = [round(end - start, 3) for start, end in facts.pauses]
    metrics.update(longest_pause=max(lengths, default=0.0), pause_count=len(lengths))
    long_ones = [(s, e) for (s, e), length in zip(facts.pauses, lengths) if length > allowed]
    if long_ones:
        longest = max(e - s for s, e in long_ones)
        out.append(Finding("long_silence", "major" if longest > max(VERY_LONG_SILENCE_SECONDS, allowed + 1.0) else "minor", "medium", "audio",
                           f"{len(long_ones)} beklenmeyen uzun sessizlik (en uzun {longest:.2f} sn; izin verilen {allowed:.2f} sn).",
                           evidence={"spans": [[round(s, 2), round(e, 2)] for s, e in long_ones], "allowed": round(allowed, 2)}))
    if facts.leading_silence > LEADING_SILENCE_SECONDS:
        out.append(Finding("leading_silence", "minor", "low", "audio", f"Başta {facts.leading_silence} sn sessizlik."))
    if facts.trailing_silence > TRAILING_SILENCE_SECONDS:
        out.append(Finding("trailing_silence", "minor", "low", "audio", f"Sonda {facts.trailing_silence} sn sessizlik."))
    metrics.update(mean_dbfs=facts.mean_dbfs, peak_dbfs=facts.peak_dbfs, clip_fraction=facts.clip_fraction)
    if facts.mean_dbfs is not None and facts.mean_dbfs < LOW_LEVEL_DBFS:
        out.append(Finding("low_level", "minor", "medium", "audio", f"Konuşma seviyesi düşük (ortalama {facts.mean_dbfs} dBFS).", evidence={"mean_dbfs": facts.mean_dbfs}))
    if facts.clip_fraction is not None and facts.clip_fraction > CLIP_FRACTION:
        out.append(Finding("clipping", "minor", "medium", "audio", f"Örneklerin %{facts.clip_fraction * 100:.2f}'i tam ölçekte (kırpılma).", evidence={"clip_fraction": facts.clip_fraction}))
    metrics["abrupt_end"] = facts.trailing_silence < ABRUPT_END_SECONDS
    return out


def _asr_findings(
    ref: list[Token],
    hyp: list[Token],
    facts: AudioFacts,
    metrics: dict[str, Any],
    asr_text: str,
    spoken_text: str,
    audio_codes: set[str],
) -> list[Finding]:
    out: list[Finding] = []
    probs = [t.prob for t in hyp if t.prob is not None]
    mean_prob = round(sum(probs) / len(probs), 3) if probs else None
    metrics.update(score_transcript(spoken_text, asr_text or " ".join(t.text for t in hyp)))
    metrics.update(asr_mean_prob=mean_prob, reference_words=len(ref), heard_words=len(hyp))
    if not hyp:
        if facts.voiced_seconds >= MIN_VOICED_SECONDS:
            out.append(Finding("asr_empty", "major", "medium", "asr", "Ses var ama ASR hiçbir kelime çıkaramadı (anlaşılmaz konuşma veya gürültü)."))
        return out
    globally_unsure = mean_prob is not None and mean_prob < ASR_GLOBAL_UNSURE_PROB
    if globally_unsure:
        out.append(Finding("asr_unsure", "minor", "low", "asr",
                           f"ASR ortalama kelime güveni düşük ({mean_prob}); metin bulguları zayıf kanıttır.", evidence={"mean_prob": mean_prob}))
    ops = align([t.text for t in ref], [t.text for t in hyp])
    too_short = "too_short" in audio_codes
    too_long = "too_long" in audio_codes
    abrupt = bool(metrics.get("abrupt_end"))
    last_ref = len(ref) - 1
    word_ops = {"deleted": 0, "inserted": 0, "substituted": 0}
    for span in error_spans(ops):
        dels = [r for kind, r, _ in span if kind == "del"]
        ins = [h for kind, _, h in span if kind == "ins"]
        subs = [(r, h) for kind, r, h in span if kind == "sub"]
        word_ops["deleted"] += len(dels)
        word_ops["inserted"] += len(ins)
        word_ops["substituted"] += len(subs)
        ref_words = [ref[r].text for r in dels + [r for r, _ in subs]]
        hyp_words = [hyp[h].text for h in ins + [h for _, h in subs]]
        # Yalnız boşluk farkı ("bir şey" / "birşey") hata değildir.
        if ref_words and hyp_words and "".join(sorted_words(span, ref, hyp, "ref")) == "".join(sorted_words(span, ref, hyp, "hyp")):
            continue
        sentence = next((ref[r].sentence for r in dels + [r for r, _ in subs]), None)
        if sentence is None:
            sentence = _nearest_sentence(ops, span, ref)
        span_probs = [hyp[h].prob for h in ins + [h for _, h in subs] if hyp[h].prob is not None]
        if not span_probs:  # silinen kelime: komşu duyulan kelimelerin güveni
            span_probs = _neighbour_probs(ops, span, hyp)
        prob = min(span_probs) if span_probs else None
        unsure = prob is not None and prob < ASR_UNSURE_PROB
        evidence = {"expected": " ".join(ref_words) or None, "heard": " ".join(hyp_words) or None, "asr_prob": prob}

        if subs and not dels and not ins:
            similarity = _similarity(" ".join(ref[r].text for r, _ in subs), " ".join(hyp[h].text for _, h in subs))
            evidence["similarity"] = round(similarity, 3)
            if len(subs) == 1 and similarity >= NEAR_WORD_SIMILARITY:
                finding = Finding("pronunciation", "minor", "low" if unsure else "medium", "asr",
                                  f"'{evidence['expected']}' yerine '{evidence['heard']}' duyuldu (yakın; telaffuz veya ASR farkı).", sentence, evidence)
            else:
                finding = Finding("wrong_word", "major", "low" if unsure else "medium", "asr",
                                  f"'{evidence['expected']}' yerine '{evidence['heard']}' duyuldu.", sentence, evidence)
        elif dels and not ins and len(dels) >= len(subs):
            trailing = max(dels) == last_ref
            short_function = len(dels) == 1 and not subs and len(ref[dels[0]].text) <= SHORT_FUNCTION_WORD_CHARS
            confidence = "medium"
            if short_function or unsure:
                confidence = "low"
            code, message = "missing_words", f"Eksik: '{evidence['expected']}'."
            if trailing:
                code, message = "truncated", f"Son kelimeler duyulmadı: '{evidence['expected']}' (kesilme şüphesi)."
                if too_short or abrupt:
                    confidence = "high" if not unsure else "medium"
            elif len(dels) >= 3 and too_short:
                confidence = _upgrade(confidence)
            finding = Finding(code, "major" if not short_function else "minor", confidence,
                              "asr+audio" if confidence == "high" else "asr", message, sentence, evidence)
        elif ins and not dels and len(ins) >= len(subs):
            repeated = _is_repeat(ins, hyp, ref, ops)
            confidence = "low" if unsure else "medium"
            if repeated:
                if not unsure and (too_long or (len(ins) >= 2 and (prob or 0) >= 0.8)):
                    confidence = "high"
                finding = Finding("repeated_words", "major", confidence, "asr+audio" if too_long and confidence == "high" else "asr",
                                  f"Tekrar duyuldu: '{evidence['heard']}'.", sentence, evidence)
            else:
                if len(ins) == 1:
                    confidence = "low"
                elif too_long and not unsure:
                    confidence = "high"
                finding = Finding("extra_words", "major" if len(ins) >= 2 else "minor", confidence, "asr",
                                  f"Metinde olmayan: '{evidence['heard']}'.", sentence, evidence)
        else:
            finding = Finding("wrong_word", "major", "low" if unsure else "medium", "asr",
                              f"'{evidence['expected'] or ''}' yerine '{evidence['heard'] or ''}' duyuldu.", sentence, evidence)
        if globally_unsure:
            finding.confidence = "low"
        out.append(finding)
    metrics["word_ops"] = word_ops
    cer = metrics.get("cer")
    if isinstance(cer, (int, float)) and cer > HIGH_CER and not globally_unsure and not any(f.severity == "major" and f.source.startswith("asr") for f in out):
        out.append(Finding("high_cer", "major", "medium", "asr", f"Duyulan metin beklenenden belirgin farklı (CER {cer:.3f}).", evidence={"cer": cer}))
    unheard = _unheard_speech(facts, hyp)
    metrics["unheard_speech_seconds"] = unheard
    if unheard > UNHEARD_SPEECH_SECONDS:
        confident = unheard >= UNHEARD_SPEECH_HIGH_SECONDS and not globally_unsure
        out.append(Finding(
            "unheard_speech", "major", "high" if confident else "medium", "asr+audio",
            f"{unheard:.2f} sn sesli bölüm hiçbir tanınan kelimeye denk gelmiyor (tekrar, metin dışı ses veya anlaşılmaz konuşma).",
            evidence={"seconds": unheard},
        ))
    if abrupt and not any(f.code == "truncated" for f in out):
        out.append(Finding("abrupt_end", "minor", "low", "audio", "Ses sonda sessizlik olmadan bitiyor (yalnız destekleyici kanıt)."))
    return out


def _unheard_speech(facts: AudioFacts, hyp: list[Token]) -> float:
    """Voiced time outside every ASR word span (+margin).

    Whisper tends to drop a repeated phrase from the transcript (VF-018: an injected repeat
    gave CER 0), but the audio is still there: on real VoxCPM2 output this was 0.06 s for a
    clean take and 1.28 s with a 1.4 s repeat.
    """
    if not facts.mask or not hyp or any(t.start is None or t.end is None for t in hyp):
        return 0.0
    covered = [False] * len(facts.mask)
    for token in hyp:
        first = max(0, int((float(token.start) - WORD_MARGIN_SECONDS) / FRAME_SECONDS))
        last = min(len(covered), int((float(token.end) + WORD_MARGIN_SECONDS) / FRAME_SECONDS) + 1)
        for index in range(first, last):
            covered[index] = True
    return round(sum(1 for voiced, hit in zip(facts.mask, covered) if voiced and not hit) * FRAME_SECONDS, 2)


def sorted_words(span: list[tuple[str, int | None, int | None]], ref: list[Token], hyp: list[Token], side: str) -> list[str]:
    if side == "ref":
        return [ref[r].text for _, r, _ in span if r is not None]
    return [hyp[h].text for _, _, h in span if h is not None]


def _nearest_sentence(ops: list[tuple[str, int | None, int | None]], span: list[Any], ref: list[Token]) -> int | None:
    index = ops.index(span[0])
    for kind, r, _ in reversed(ops[:index]):
        if r is not None:
            return ref[r].sentence
    for kind, r, _ in ops[index:]:
        if r is not None:
            return ref[r].sentence
    return None


def _neighbour_probs(ops: list[tuple[str, int | None, int | None]], span: list[Any], hyp: list[Token]) -> list[float]:
    start, stop = ops.index(span[0]), ops.index(span[-1])
    found = []
    for kind, _, h in reversed(ops[:start]):
        if h is not None:
            found.append(hyp[h].prob)
            break
    for kind, _, h in ops[stop + 1 :]:
        if h is not None:
            found.append(hyp[h].prob)
            break
    return [p for p in found if p is not None]


def _is_repeat(ins: list[int], hyp: list[Token], ref: list[Token], ops: list[tuple[str, int | None, int | None]]) -> bool:
    """Inserted words that repeat what was just said (or is about to be said)."""
    words = [hyp[h].text for h in ins]
    first, size = min(ins), len(ins)
    before = [t.text for t in hyp[max(0, first - size) : first]]
    after = [t.text for t in hyp[first + size : first + 2 * size]]
    if words == before or words == after:
        return True
    neighbours = {t.text for t in hyp[max(0, first - 3) : first]} | {t.text for t in hyp[first + size : first + size + 3]}
    return all(word in neighbours for word in words)


def summarize(report: dict[str, Any] | None) -> str:
    """One line for lists/tables."""
    if not report:
        return VERDICT_LABELS["unchecked"]
    metrics = report.get("metrics") or {}
    parts = [report.get("verdict_label") or VERDICT_LABELS.get(report.get("verdict", ""), "?")]
    if metrics.get("cer") is not None:
        parts.append(f"CER {metrics['cer']:.3f}")
    major = [f["label"] for f in report.get("findings", []) if f["severity"] == "major" and f["confidence"] != "low"]
    if major:
        parts.append(", ".join(dict.fromkeys(major)))
    if not report.get("asr_checked"):
        parts.append("ASR yok")
    return " · ".join(parts)


def describe_report(report: dict[str, Any] | None) -> str:
    """Markdown explanation: verdict, measurements, then each finding with its evidence source."""
    if not report:
        return "_Kalite denetimi yapılmadı._"
    metrics = report.get("metrics") or {}
    lines = [f"**Kalite: {report.get('verdict_label')}** · denetleyici `{report.get('checker_version')}`"]
    measured = []
    if metrics.get("cer") is not None:
        measured.append(f"CER {metrics['cer']:.3f} (eksik {metrics.get('deletions')}, fazla {metrics.get('insertions')}, farklı {metrics.get('substitutions')} karakter)")
    if metrics.get("rate") is not None:
        measured.append(f"hız {metrics['rate']} hece/sn")
    if metrics.get("seconds_per_char") is not None:
        measured.append(f"{metrics['seconds_per_char']} sn/karakter")
    if metrics.get("longest_pause"):
        measured.append(f"en uzun iç sessizlik {metrics['longest_pause']} sn")
    if metrics.get("mean_dbfs") is not None:
        measured.append(f"seviye {metrics['mean_dbfs']} dBFS")
    if metrics.get("asr_mean_prob") is not None:
        measured.append(f"ASR güveni {metrics['asr_mean_prob']}")
    if measured:
        lines.append("- Ölçüm: " + " · ".join(measured))
    asr = report.get("asr") or {}
    if report.get("asr_checked"):
        lines.append(f"- Duyulan ({asr.get('model') or 'ASR'}): _{asr.get('text') or '—'}_")
    else:
        lines.append(f"- ASR çalıştırılmadı{': ' + asr['error'] if asr.get('error') else ''}; yalnız ses ölçümleri.")
    sources = {"audio": "ses ölçümü", "asr": "ASR görüşü", "asr+audio": "ASR + ses ölçümü", "speaker": "konuşmacı benzerliği"}
    confidence = {"high": "yüksek", "medium": "orta", "low": "düşük"}
    for item in report.get("findings") or []:
        where = f" · cümle {item['sentence'] + 1}" if item.get("sentence") is not None else ""
        lines.append(
            f"- {'⚠️' if item['severity'] == 'major' else '·'} **{item['label']}** ({confidence[item['confidence']]} güven, "
            f"{sources.get(item['source'], item['source'])}{where}): {item['message']}"
        )
    if not report.get("findings"):
        lines.append("- Bulgu yok.")
    if report.get("asr_checked"):
        lines.append("- _ASR farkları kesin TTS hatası değildir; yalnız yüksek güvenli sorunlar otomatik alternatif başlatır._")
    return "\n".join(lines)
