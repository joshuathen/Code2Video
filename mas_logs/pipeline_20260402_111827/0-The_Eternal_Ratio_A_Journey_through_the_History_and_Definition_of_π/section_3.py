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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Archimedes trapped the circle between two squares.',
            'He increased the side counts to regular hexagons.',
            'He eventually used polygons with ninety-six sides.',
            'This squeezed pi between two narrow decimal values.',
            'The gap between the shapes nearly disappears.'
        ]
        self.setup_layout("Ancient Origins: The Polygon Approximation", lecture_lines)

        # Positioning Constants
        CIRCLE_RADIUS = 1.8
        
        # Initialize Mobjects
        # Circle
        circle = Circle(radius=CIRCLE_RADIUS, color="#FFFFFF", stroke_width=2)
        
        # Squares (n=4)
        in_4 = RegularPolygon(n=4, radius=CIRCLE_RADIUS, color="#0000FF", stroke_width=4).rotate(PI/4)
        out_4 = RegularPolygon(n=4, radius=CIRCLE_RADIUS / np.cos(PI/4), color="#FF0000", stroke_width=4).rotate(PI/4)
        
        # Hexagons (n=6)
        hex_color = "#00FF00"
        in_6 = RegularPolygon(n=6, radius=CIRCLE_RADIUS, color=hex_color, stroke_width=3)
        out_6 = RegularPolygon(n=6, radius=CIRCLE_RADIUS / np.cos(PI/6), color=hex_color, stroke_width=3)
        
        # 96-sided polygons
        in_96 = RegularPolygon(n=96, radius=CIRCLE_RADIUS, color=hex_color, stroke_width=2)
        out_96 = RegularPolygon(n=96, radius=CIRCLE_RADIUS / np.cos(PI/96), color=hex_color, stroke_width=2)
        
        # Issue 31: Primary geometric construction scale
        # Grouping all geometric objects to apply consistent scaling and positioning
        construction_group = VGroup(circle, in_4, out_4, in_6, out_6, in_96, out_96)
        self.place_in_area(construction_group, 'B2', 'E5', scale_factor=0.85)

        # === Animation for Lecture Line 1 ===
        # Archimedes trapped the circle between two squares.
        self.lecture[0].set_color(YELLOW)
        self.play(Create(circle), Create(in_4), Create(out_4))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # He increased the side counts to regular hexagons.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
            ReplacementTransform(in_4, in_6),
            ReplacementTransform(out_4, out_6)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # He eventually used polygons with ninety-six sides.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            ReplacementTransform(in_6, in_96),
            ReplacementTransform(out_6, out_96),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This squeezed pi between two narrow decimal values.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        pi_range = Text("3.1408 < π < 3.1429", font_size=36, color=YELLOW)
        # Issue 29 & 30: Center formula horizontally and reduce scale to 0.75
        self.place_in_area(pi_range, 'F1', 'F6', scale_factor=0.75)
        self.play(FadeIn(pi_range))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The gap between the shapes nearly disappears.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(
            in_96.animate.set_stroke(opacity=0.3, width=1),
            out_96.animate.set_stroke(opacity=0.3, width=1),
            circle.animate.set_stroke(width=4, color="#FFFFFF"),
            run_time=2
        )
        self.wait(3)
