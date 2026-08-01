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

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Prerequisite: The Magic of 'Almost Zero'"
        lines = [
            "We approximate complex curves using tiny straight intervals.",
            "Increasing these intervals makes the approximation much more accurate.",
            "As intervals shrink toward zero, the true shape emerges."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CIRCLE = "#FFFFFF"
        COLOR_POLYGON = "#FF00FF"
        COLOR_LIMIT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Display a white circle #FFFFFF and a magenta hexagon #FF00FF inside it.
        self.lecture[0].set_color(COLOR_CIRCLE)
        
        circle = Circle(radius=1.8, color=COLOR_CIRCLE)
        # Shift all animation elements from columns 2-5 to columns 3-6 to balance the composition. (Issue 26)
        # Specifically placing in B3-E6 (Issue 24)
        self.place_in_area(circle, "B3", "E6", scale_factor=0.8)
        
        hexagon = RegularPolygon(n=6, color=COLOR_POLYGON).replace(circle)
        # The polygon should be slightly smaller to look "inscribed" visually or exactly match
        hexagon.scale(0.98) 

        self.play(Create(circle), run_time=1.5)
        self.play(Create(hexagon), run_time=1.5)
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Smoothly increase the polygon sides from 6 to 12, 24, and 48 until it approximates the circle.
        self.lecture[1].set_color(COLOR_POLYGON)
        
        poly_12 = RegularPolygon(n=12, color=COLOR_POLYGON).replace(circle).scale(0.98)
        poly_24 = RegularPolygon(n=24, color=COLOR_POLYGON).replace(circle).scale(0.99)
        poly_48 = RegularPolygon(n=48, color=COLOR_POLYGON).replace(circle).scale(0.99)

        self.play(Transform(hexagon, poly_12), run_time=1.0)
        self.play(Transform(hexagon, poly_24), run_time=1.0)
        self.play(Transform(hexagon, poly_48), run_time=1.0)
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Highlight the vanishing gap with a yellow #FFFF00 arrow and the label 'Limit'.
        self.lecture[2].set_color(COLOR_LIMIT)
        
        # Place 'Limit' label at 'A6' with scale_factor=0.6 (Issue 25)
        limit_label = Text("Limit", font_size=24, color=COLOR_LIMIT)
        self.place_at_grid(limit_label, 'A6', scale_factor=0.6)
        
        # Calculate arrow points relative to the circle's new position
        # Circle center is the center of B3-E6
        c_pos = circle.get_center()
        
        # We want to point to the top-right-ish edge of the circle
        # Vector from center to A6 label
        l_pos = limit_label.get_center()
        dir_vec = l_pos - c_pos
        dir_norm = dir_vec / np.linalg.norm(dir_vec)
        
        # Arrow points from near the label to the edge of the circle
        arrow_start = l_pos + DOWN * 0.3
        arrow_end = c_pos + (dir_norm * 1.5) # radius is approx 1.8 * 0.8 = 1.44
        
        limit_arrow = Arrow(
            start=arrow_start, 
            end=arrow_end, 
            color=COLOR_LIMIT, 
            buff=0.05,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15 # Adhering to L020 logic indirectly
        )

        self.play(Create(limit_arrow))
        self.play(Write(limit_label))
        
        # Use Indicate for the highlight (L004)
        self.play(Indicate(limit_label, color=COLOR_LIMIT), Indicate(limit_arrow, color=COLOR_LIMIT))
        
        self.wait(2.0)
