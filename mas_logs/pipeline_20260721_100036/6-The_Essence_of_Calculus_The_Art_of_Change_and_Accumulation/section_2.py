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
        # Data from storyboard
        title = "Prerequisite: The Power of Approximation"
        lecture_lines = [
            "Complex shapes are hard to measure directly.",
            "We can approximate them using many simpler pieces.",
            "As pieces become infinitely small, the approximation becomes exact."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show a circle (#00FFFF) with a triangle (#FFFF00) inside it.
        # Change lecture line 1 color to #FFFF00.
        
        circle = Circle(radius=2, color="#00FFFF")
        # Fix for Issue 29: Position circle to avoid crowding lecture text
        self.place_in_area(circle, 'B3', 'E6', scale_factor=1.0)
        
        triangle = RegularPolygon(n=3, radius=2, color="#FFFF00", fill_opacity=0.3)
        self.place_in_area(triangle, 'B3', 'E6', scale_factor=1.0)
        
        self.play(
            Create(circle),
            FadeIn(triangle),
            self.lecture[0].animate.set_color("#FFFF00")
        )
        self.wait(1.5)
        
        # === Animation for Lecture Line 2 ===
        # Replace the triangle with a hexagon (#FF8C00) then a dodecagon (#00FF00).
        # Change lecture line 2 color to #FFFF00.
        
        hexagon = RegularPolygon(n=6, radius=2, color="#FF8C00", fill_opacity=0.3)
        self.place_in_area(hexagon, 'B3', 'E6', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(triangle, hexagon),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)
        
        dodecagon = RegularPolygon(n=12, radius=2, color="#00FF00", fill_opacity=0.3)
        self.place_in_area(dodecagon, 'B3', 'E6', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(hexagon, dodecagon)
        )
        self.wait(1.5)
        
        # === Animation for Lecture Line 3 ===
        # As sides increase, polygon color blends into the circle color.
        # Label: 'n -> Infinity'. Change lecture line 3 color to #FFFF00.
        
        # Create label
        n_label = MathTex("n \\to \\infty", font_size=36, color=WHITE)
        # Fix for Issue 30: Centered label positioning
        self.place_in_area(n_label, 'F4', 'F5', scale_factor=0.8)
        
        # Intermediate polygons for smooth-ish transition to simulate "n -> infinity"
        # We use a few steps to keep performance high and follow instructions
        poly_24 = RegularPolygon(n=24, radius=2, color="#00CCFF", fill_opacity=0.3)
        self.place_in_area(poly_24, 'B3', 'E6', scale_factor=1.0)
        
        poly_48 = RegularPolygon(n=48, radius=2, color="#00EEFF", fill_opacity=0.3)
        self.place_in_area(poly_48, 'B3', 'E6', scale_factor=1.0)
        
        # The limit is effectively the filled circle matching the perimeter's color
        circle_fill = Circle(radius=2, color="#00FFFF", fill_opacity=0.3)
        self.place_in_area(circle_fill, 'B3', 'E6', scale_factor=1.0)
        
        self.play(
            Write(n_label),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.wait(1)
        
        self.play(ReplacementTransform(dodecagon, poly_24), run_time=1)
        self.play(ReplacementTransform(poly_24, poly_48), run_time=1)
        self.play(ReplacementTransform(poly_48, circle_fill), run_time=1)
        
        self.wait(3)
