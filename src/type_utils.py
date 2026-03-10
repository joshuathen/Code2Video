from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

@dataclass
class Section:
    id: str
    title: str
    lecture_lines: List[str]
    animations: List[str]


@dataclass
class TeachingOutline:
    topic: str
    target_audience: str
    sections: List[Dict[str, Any]]


@dataclass
class SectionOutline:
    id: str
    title: str
    content: str
    example: str