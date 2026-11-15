"""Read local book documents without extracting archive members to disk."""

from __future__ import annotations

import io
import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.entities import html5
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.parsers.expat import errors as xml_errors
from zipfile import BadZipFile, ZipFile

FORMATS = (".txt", ".docx", ".epub")
MAX_DOCUMENT_BYTES = 32 * 1024**2
WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
CHAPTER_LABEL = r"(?:bölüm|chapter|kısım|part)"
CHAPTER_NUMBER = r"(?:\d+|[ivxlcdm]+)\.?"
TITLE_SEPARATOR = r"\s*[:—–-]\s*\S.*"
HEADING = re.compile(
    rf"(?:#{{1,2}}\s+.+|{CHAPTER_LABEL}(?:\s+{CHAPTER_NUMBER}(?:{TITLE_SEPARATOR})?|{TITLE_SEPARATOR})"
    rf"|{CHAPTER_NUMBER}\s+{CHAPTER_LABEL}(?:{TITLE_SEPARATOR})?)",
    re.IGNORECASE,
)
HTML_ENTITY = re.compile(r"&([A-Za-z][A-Za-z0-9]+;)")


@dataclass(frozen=True)
class Chapter:
    title: str
    text: str
    has_heading: bool = False


@dataclass(frozen=True)
class Book:
    extension: str
    original: bytes
    chapters: tuple[Chapter, ...]


def _chapters(blocks: list[tuple[bool, str]], fallback: str, separator: str) -> list[Chapter]:
    chapters: list[Chapter] = []
    title = fallback
    body: list[str] = []
    has_heading = False
    for heading, block in blocks:
        if heading and block.strip():
            if has_heading or any(line.strip() for line in body):
                chapters.append(Chapter(title, separator.join(body), has_heading))
            body = []
            title = re.sub(r"^#{1,2}\s+", "", block.strip())
            has_heading = True
        else:
            body.append(block)
    if has_heading or any(line.strip() for line in body):
        chapters.append(Chapter(title, separator.join(body), has_heading))
    return chapters


def _xml(archive: ZipFile, member: str, *, html: bool = False) -> ET.Element:
    entry = archive.getinfo(member)
    if entry.file_size > MAX_DOCUMENT_BYTES:
        raise ValueError("Belgedeki XML bölümü çok büyük.")
    data = archive.read(entry)
    if b"<!ENTITY" in data.replace(b"\x00", b""):
        raise ValueError("XML varlık tanımları desteklenmiyor.")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        if not html or exc.code != xml_errors.codes[xml_errors.XML_ERROR_UNDEFINED_ENTITY]:
            raise

    def numeric_entity(match: re.Match[str]) -> str:
        value = html5.get(match[1])
        return "".join(f"&#{ord(char)};" for char in value) if value else match[0]

    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return ET.fromstring(HTML_ENTITY.sub(numeric_entity, data.decode(encoding)))


def _docx(archive: ZipFile, fallback: str) -> list[Chapter]:
    body = _xml(archive, "word/document.xml").find(f"{WORD}body")
    if body is None:
        raise ValueError("DOCX belge gövdesi bulunamadı.")
    heading_styles = {"heading1", "heading2", "title"}
    if "word/styles.xml" in archive.namelist():
        for style in _xml(archive, "word/styles.xml").iter(f"{WORD}style"):
            outline = style.find(f"{WORD}pPr/{WORD}outlineLvl")
            name = style.find(f"{WORD}name")
            if (
                (outline is not None and outline.get(f"{WORD}val") in {"0", "1"})
                or (name is not None and name.get(f"{WORD}val", "").lower() in {"heading 1", "heading 2", "title"})
            ):
                heading_styles.add(style.get(f"{WORD}styleId", "").lower())
    blocks = []
    breaks = {f"{WORD}tab": "\t", f"{WORD}br": "\n", f"{WORD}cr": "\n"}
    for paragraph in body.iter(f"{WORD}p"):
        text = "".join(
            (node.text or "") if node.tag == f"{WORD}t" else breaks.get(node.tag, "")
            for node in paragraph.iter()
        )
        style = paragraph.find(f"{WORD}pPr/{WORD}pStyle")
        outline = paragraph.find(f"{WORD}pPr/{WORD}outlineLvl")
        heading = (
            (style is not None and style.get(f"{WORD}val", "").lower() in heading_styles)
            or (outline is not None and outline.get(f"{WORD}val") in {"0", "1"})
            or HEADING.fullmatch(text.strip()) is not None
        )
        blocks.append((bool(text.strip()) and heading, text))
    return _chapters(blocks, fallback, "\n\n")


def _tag(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _html_text(node: ET.Element) -> str:
    pieces = [node.text or ""]
    for child in node:
        tag = _tag(child)
        if tag == "br":
            pieces.append("\n")
        elif tag not in {"script", "style", "nav"}:
            pieces.append(_html_text(child))
        pieces.append(child.tail or "")
    return "".join(pieces)


def _html_blocks(node: ET.Element) -> list[tuple[bool, str]]:
    tag = _tag(node)
    if tag in {"head", "nav", "script", "style"} or node.get("hidden") is not None:
        return []
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "dt", "dd"}:
        return [(tag in {"h1", "h2"}, _html_text(node))]
    blocks = [(False, node.text)] if node.text and node.text.strip() else []
    for child in node:
        if _tag(child) in {"a", "em", "strong", "span", "i", "b", "br"}:
            inline = ("\n" if _tag(child) == "br" else _html_text(child)) + (child.tail or "")
            if blocks and not blocks[-1][0]:
                blocks[-1] = (False, blocks[-1][1] + inline)
            else:
                blocks.append((False, inline))
        else:
            blocks.extend(_html_blocks(child))
            if child.tail and child.tail.strip():
                blocks.append((False, child.tail))
    return blocks


def _epub(archive: ZipFile, fallback: str) -> list[Chapter]:
    container = _xml(archive, "META-INF/container.xml")
    rootfile = next(iter(container.findall(".//{*}rootfile")), None)
    if rootfile is None:
        raise ValueError("EPUB paket belgesi bulunamadı.")
    package_path = rootfile.attrib["full-path"]
    package = _xml(archive, package_path)
    manifest = {item.attrib["id"]: item for item in package.findall("./{*}manifest/{*}item")}
    result = []
    for itemref in package.findall("./{*}spine/{*}itemref"):
        item = manifest[itemref.attrib["idref"]]
        if itemref.get("linear") == "no" or "nav" in item.get("properties", "").split():
            continue
        if item.get("media-type") != "application/xhtml+xml":
            continue
        url = urlsplit(item.attrib["href"])
        href = unquote(url.path)
        if url.scheme or url.netloc or href.startswith("/"):
            raise ValueError("EPUB bölüm adresi yerel olmalı.")
        member = posixpath.normpath(posixpath.join(posixpath.dirname(package_path), href))
        document = _xml(archive, member, html=True)
        title_node = document.find("./{*}head/{*}title")
        title = _html_text(title_node).strip() if title_node is not None else ""
        result.extend(_chapters(_html_blocks(document), title or f"{fallback} · Bölüm {len(result) + 1}", "\n\n"))
    return result


def read_book(path: Path) -> Book:
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in FORMATS:
        raise ValueError("TXT, DOCX veya EPUB dosyası seçin.")
    with path.open("rb") as handle:
        original = handle.read(MAX_DOCUMENT_BYTES + 1)
    if len(original) > MAX_DOCUMENT_BYTES:
        raise ValueError("Kaynak dosya 32 MiB sınırını aşıyor.")
    try:
        if extension == ".txt":
            encoding = "utf-16" if original.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            text = original.decode(encoding)
            if "\x00" in text:
                raise ValueError("TXT dosyasını UTF-8 veya BOM’lu UTF-16 olarak kaydedin.")
            blocks = [(bool(HEADING.fullmatch(line.strip())), line) for line in text.splitlines()]
            chapters = _chapters(blocks, path.stem, "\n")
        else:
            with ZipFile(io.BytesIO(original)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 4 * MAX_DOCUMENT_BYTES:
                    raise ValueError("Açılmış belge 128 MiB sınırını aşıyor.")
                chapters = (_docx if extension == ".docx" else _epub)(archive, path.stem)
    except (BadZipFile, ET.ParseError, KeyError, RuntimeError, UnicodeError) as exc:
        raise ValueError(f"{extension.upper()} belgesi okunamadı: {exc}") from exc
    if not chapters:
        raise ValueError("Belgede seslendirilecek metin bulunamadı.")
    return Book(extension, original, tuple(chapters))
