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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout(
            "The Intuition: 'Getting Close' without Touching",
            [
                "Turbo the snail crawls along function f of x.",
                "At x equals one, there is a missing point.",
                "Turbo looks ahead to see where he's heading.",
                "A limit describes this expected value, even if unreachable.",
                "It's about the journey toward that gap in space."
            ]
        )

        # Visual anchor and colors
        TURBO_COLOR = "#FFD700"  # Yellow
        GAP_COLOR = "#FF0000"    # Red
        LIMIT_COLOR = "#FFD700"  # Yellow
        GLOW_COLOR = "#00FFFF"   # Bright Blue/Cyan
        LINE_COLOR = "#FFFFFF"   # White

        # === Animation for Lecture Line 1 ===
        # Draw a white line segment with a gap in the middle. Place Turbo at the start.
        # We use grid points C1 to C2 and C4 to C5 to visually create a gap at C3.
        line_left = Line(self.grid['C1'], self.grid['C2'], color=LINE_COLOR)
        line_right = Line(self.grid['C4'], self.grid['C5'], color=LINE_COLOR)
        turbo = Dot(color=TURBO_COLOR)
        self.place_at_grid(turbo, 'C1', scale_factor=1.2)
        
        self.play(
            self.lecture[0].animate.set_color(TURBO_COLOR),
            Create(line_left),
            Create(line_right),
            FadeIn(turbo)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A small red circle appears at the gap, labeled 'x=1' above it (B3).
        gap_marker = Dot(color=GAP_COLOR)
        self.place_at_grid(gap_marker, 'C3', scale_factor=1.0)
        label_x1 = Text("x=1", font_size=20, color=GAP_COLOR)
        self.place_at_grid(label_x1, 'B3')
        
        self.play(
            self.lecture[1].animate.set_color(GAP_COLOR),
            FadeIn(gap_marker),
            Write(label_x1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Move Turbo along the line until he reaches the edge of the gap (C2).
        self.play(
            self.lecture[2].animate.set_color(TURBO_COLOR),
            turbo.animate.move_to(self.grid['C2'])
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A semi-transparent yellow circle fills the gap (C3), labeled 'Limit = 2' nearby (D3).
        limit_fill = Dot(color=LIMIT_COLOR, fill_opacity=0.5)
        self.place_at_grid(limit_fill, 'C3', scale_factor=2.0)
        label_limit = Text("Limit = 2", font_size=20, color=LIMIT_COLOR)
        self.place_at_grid(label_limit, 'D3')
        
        self.play(
            self.lecture[3].animate.set_color(LIMIT_COLOR),
            FadeIn(limit_fill),
            Write(label_limit)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The line segment behind Turbo glows bright blue to emphasize the path.
        glow_line = Line(self.grid['C1'], self.grid['C2'], color=GLOW_COLOR, stroke_width=8)
        
        self.play(
            self.lecture[4].animate.set_color(GLOW_COLOR),
            Create(glow_line)
        )
        self.wait(2)
