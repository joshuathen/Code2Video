import os


def _get_prompt3_code_with_content(
    regenerate_note,
    section,
    base_class,
    section_content_block,
    setup_layout_title,
    setup_layout_lines,
):
    return f"""
You are an expert Manim animator using Manim Community Edition v0.19.0. 
Please generate a high-quality Manim class based on the following teaching script.
{regenerate_note}

1. Basic Requirements:
- Use the provided TeachingScene base class without modification.
- Each lecture line must have a matching color with its corresponding animation elements.
- Apply ONLY color changes to lecture lines - no scaling, translation, or Transform animations.

2. Visual Anchor System (MANDATORY):
- Use 6x6 grid system (A1-F6) for precise positioning.
- Pay attention to the positioning of elements to avoid occlusions (e.g., labels and formulas).
- All labels must be positioned within 1 grid unit of their corresponding objects
- Grid layout (right side only):
```
lecture |  A1  A2  A3  A4  A5  A6
        |  B1  B2  B3  B4  B5  B6
        |  C1  C2  C3  C4  C5  C6
        |  D1  D2  D3  D4  D5  D6
        |  E1  E2  E3  E4  E5  E6
        |  F1  F2  F3  F4  F5  F6
```

3. POSITIONING METHODS:
- Point example: self.place_at_grid(obj, 'B2', scale_factor=0.8)
- Area example: self.place_in_area(obj, 'A1', 'C3', scale_factor=0.7)
- NEVER use .to_edge(), .move_to(), or manual positioning!

{section_content_block}

5. STRUCTURE FOR CODE:
Use the following comment format to indicate which block corresponds to which line:
```python
# === Animation for Lecture Line 1 ===

6. EXAMPLE STRUCTURE:
```python
from manim import *

{base_class}

class {section.id.title().replace('_', '')}Scene(TeachingScene):
    def construct(self):
        self.setup_layout({setup_layout_title}, {setup_layout_lines})
        
        # rest of animation code
        # === Animation for Lecture Line 1 ===
        ...

        # === Animation for Lecture Line 2 ===
        ...
```

7. MANDATORY CONSTRAINTS:
- Colors: Use light, distinguishable hexadecimal colors.
- Scaling: Maintain appropriate font sizes and object scales for readability.
- Consistency: Do not apply any animation to the lecture lines except for color changes; The lecture lines and title's size and position must remain unchanged.
- Assets: If provided, MUST use the elements in the Animation Description formatted as [Asset: XXX/XXX.png] (abstract path).
- Simplicity: Avoid 3D functions, complex panels, or external dependencies except for filenames in Animation Description.
"""


def get_prompt3_code(regenerate_note, section, base_class):
    section_content_block = f"""
4. TEACHING CONTENT:
- Title: {section.title}
- Lecture Lines: {section.lecture_lines}
- Animation Description: {'; '.join(section.animations)}
"""
    return _get_prompt3_code_with_content(
        regenerate_note=regenerate_note,
        section=section,
        base_class=base_class,
        section_content_block=section_content_block,
        setup_layout_title=repr(section.title),
        setup_layout_lines=repr(section.lecture_lines),
    )


def get_prompt3_code_mas(regenerate_note, section, base_class):
    section_content_block = f"""
4. TEACHING CONTENT:
- You are working on section id: {section.id}
- Use the available read tools to fetch the current outline, storyboard, render context, code, and issues before you write code.
- Do not assume the prompt contains the latest lecture lines or animation descriptions; retrieve them from shared state.
"""
    return _get_prompt3_code_with_content(
        regenerate_note=regenerate_note,
        section=section,
        base_class=base_class,
        section_content_block=section_content_block,
        setup_layout_title='"SECTION_TITLE_HERE"',
        setup_layout_lines='["LECTURE_LINE_1", "LECTURE_LINE_2"]',
    )


def get_regenerate_note(attempt, MAX_REGENERATE_TRIES):
    return f"""    
**IMPORTANT NOTE:** This is attempt {attempt}/{MAX_REGENERATE_TRIES} to generate working code.
The previous attempts failed to run correctly. Please:
1. Use only basic, well-tested Manim functions
2. Avoid complex animations that might cause errors
3. Use simple, reliable Manim patterns
"""
