from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        title = "Practical Application: Your 5-Minute Strategy"
        lines = [
            "Prioritize high-frequency consonants and avoid repeating letters.",
            "Use top-tier openers like SLATE, CRANE, or TRACE.",
            "Focus on gaining information to solve the puzzle faster."
        ]
        self.setup_layout(title, lines)

        # Pre-create common colors
        COLOR_CONSONANT = "#FFFFFF"
        COLOR_NO_REPEAT = "#FF0000"
        COLOR_CHEAT_SHEET = "#D4AF37"
        COLOR_SOLVED = "#6AAA64"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Consonants Icons
        consonants = ["S", "T", "R", "L", "N"]
        consonant_icons = VGroup()
        for char in consonants:
            circ = Circle(radius=0.4, color=COLOR_CONSONANT)
            txt = Text(char, font_size=32, color=COLOR_CONSONANT)
            consonant_icons.add(VGroup(circ, txt))
        
        consonant_icons.arrange_in_grid(rows=2, cols=3, buff=0.3)
        self.place_in_area(consonant_icons, 'B1', 'C3', scale_factor=0.8)
        
        # No-repeat sign
        no_repeat_circle = Circle(radius=0.8, color=COLOR_NO_REPEAT, stroke_width=8)
        no_repeat_slash = Line(
            start=no_repeat_circle.point_at_angle(135 * DEGREES),
            end=no_repeat_circle.point_at_angle(-45 * DEGREES),
            color=COLOR_NO_REPEAT,
            stroke_width=8
        )
        
        # Representative "Repeated Letter" visual (e.g. 'E' and 'E')
        ee_text = Text("E ... E", color=COLOR_NO_REPEAT, font_size=36)
        no_repeat_group = VGroup(no_repeat_circle, no_repeat_slash, ee_text)
        self.place_in_area(no_repeat_group, 'B4', 'C6', scale_factor=0.8)
        
        self.play(FadeIn(consonant_icons))
        self.play(Create(no_repeat_circle), Create(no_repeat_slash), Write(ee_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Golden Cheat Sheet Box
        cheat_sheet_bg = Rectangle(width=2.5, height=2.2, color=COLOR_CHEAT_SHEET, stroke_width=4)
        cheat_sheet_title = Text("CHEAT SHEET", font_size=20, color=COLOR_CHEAT_SHEET).next_to(cheat_sheet_bg.get_top(), DOWN, buff=0.1)
        words = VGroup(
            Text("SLATE", font_size=24, color=WHITE),
            Text("CRANE", font_size=24, color=WHITE),
            Text("TRACE", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.2).next_to(cheat_sheet_title, DOWN, buff=0.2)
        
        cheat_sheet = VGroup(cheat_sheet_bg, cheat_sheet_title, words)
        self.place_in_area(cheat_sheet, 'E1', 'F3', scale_factor=0.8)
        
        self.play(FadeIn(cheat_sheet))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Magnifying Glass zooming into a Solved Grid
        # Create a simplified 3x3 grid section for the zoom
        grid_rows = VGroup()
        for i in range(3):
            row = VGroup(*[Square(side_length=0.4, fill_opacity=1, fill_color=COLOR_SOLVED if i == 2 else GRAY) for _ in range(5)]).arrange(RIGHT, buff=0.05)
            grid_rows.add(row)
        grid_rows.arrange(DOWN, buff=0.05)
        
        # Final "Solved" state label
        solved_text = Text("SOLVED", font_size=20, color=COLOR_SOLVED).next_to(grid_rows, DOWN, buff=0.1)
        solved_ui = VGroup(grid_rows, solved_text)
        self.place_in_area(solved_ui, 'E4', 'F6', scale_factor=0.8)
        
        # Magnifying Glass
        mag_circle = Circle(radius=0.5, color=WHITE, stroke_width=4)
        mag_handle = Line(mag_circle.point_at_angle(-45 * DEGREES), mag_circle.point_at_angle(-45 * DEGREES) + (0.4 * DOWN + 0.4 * RIGHT), color=WHITE, stroke_width=6)
        magnifying_glass = VGroup(mag_circle, mag_handle)
        
        # Position magnifying glass over the grid center using same grid area
        self.place_in_area(magnifying_glass, 'E4', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(solved_ui))
        self.play(FadeIn(magnifying_glass))
        self.play(magnifying_glass.animate.scale(1.5), run_time=1.5)
        self.play(magnifying_glass.animate.scale(1/1.5), run_time=1)
        
        self.wait(3)
