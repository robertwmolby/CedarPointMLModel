from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from bs4 import Tag
from urllib.parse import urljoin


@dataclass(slots=True)
class CoasterPage:
    """
    Holds information about roller coasters scraped from a web site.
    """
    id: int
    name: str
    url: str
    amusement_park: str
    coaster_type: str
    design: str
    status: str

    country: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    length: Optional[float] = None        # meters (or your unit)
    height: Optional[float] = None
    drop: Optional[float] = None
    inversion_count: Optional[int] = None
    speed: Optional[float] = None
    vertical_angle: Optional[float] = None
    duration: Optional[float] = None      # seconds
    restraints: Optional[str] = None
    g_force: Optional[float] = None
    intensity: Optional[str] = None

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _text_or_none(node: Tag | None) -> Optional[str]:
        return node.get_text(strip=True) if node else None

    @staticmethod
    def _anchor_href(node: Tag | None, base: Optional[str] = None) -> Optional[str]:
        if not node:
            return None
        href = node.get("href")
        if href and base:
            return urljoin(base, href)
        return href

    @staticmethod
    def _to_int(s: Optional[str]) -> Optional[int]:
        if s is None or s == "":
            return None
        try:
            return int(s)
        except ValueError:
            return None

    @staticmethod
    def _to_float(s: Optional[str]) -> Optional[float]:
        if s is None or s == "":
            return None
        try:
            # remove common decorators like units/commas
            clean = (
                s.replace(",", "")
                 .replace("°", "")
                 .replace("%", "")
                 .replace("mph", "")
                 .replace("km/h", "")
                 .replace("m", "")
                 .replace("s", "")
                 .strip()
            )
            return float(clean)
        except ValueError:
            return None

    @staticmethod
    def _coaster_id_from_href(href: str) -> Optional[int]:
        # e.g. "/rollercoaster/12345.htm" -> 12345
        if not href:
            return None
        stem = href.rsplit("/", 1)[-1].removesuffix(".htm")
        try:
            return int(stem)
        except ValueError:
            return None

    # ---- main constructor -------------------------------------------------

    @classmethod
    def from_table_row(cls, row: Tag, *, base_url: Optional[str] = None) -> "CoasterPage":
        """
        Parse a <tr> like:
        td[1]=name(link), td[2]=park(link), td[3]=type(link), td[4]=design(link), td[5]=status(link or text)
        """
        cells = row.find_all("td")
        if len(cells) < 6:
            raise ValueError("Expected at least 6 <td> cells in coaster row")

        # Required bits
        name_a = cells[1].find("a")
        url = cls._anchor_href(name_a, base=base_url) or ""
        cid = cls._coaster_id_from_href(name_a.get("href", "")) or -1
        name = cls._text_or_none(name_a) or ""

        park_a = cells[2].find("a")
        park = cls._text_or_none(park_a) or ""

        type_a = cells[3].find("a")
        ctype = cls._text_or_none(type_a) or ""

        design_a = cells[4].find("a")
        design = cls._text_or_none(design_a) or ""

        # Status may be link or plain text or empty
        status_link = cells[5].find("a")
        status = (cls._text_or_none(status_link) or cls._text_or_none(cells[5]) or "Unknown")

        return cls(
            id=cid,
            name=name,
            url=url,
            amusement_park=park,
            coaster_type=ctype,
            design=design,
            status=status,
        )
