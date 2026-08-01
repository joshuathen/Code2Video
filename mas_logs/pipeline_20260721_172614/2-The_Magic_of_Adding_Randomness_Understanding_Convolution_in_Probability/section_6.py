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
        # Setup Section 6
        title = "The Result: Why Shapes Change"
        lines = [
            "Convolution generally smooths out the resulting distribution shape.",
            "Adding two uniform blocks creates a smoother triangle.",
            "More variables lead toward the famous bell-shaped curve."
        ]
        self.setup_layout(title, lines)

        # Define Colors
        COLOR_SQUARE = "#00FF00"
        COLOR_TRIANGLE = "#FFFF00"
        COLOR_GAUSSIAN = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Convolution generally smooths out the resulting distribution shape.
        self.lecture[0].set_color(COLOR_SQUARE)
        
        sq1 = Rectangle(height=1, width=1, fill_opacity=0.5, color=COLOR_SQUARE)
        sq2 = Rectangle(height=1, width=1, fill_opacity=0.5, color=COLOR_SQUARE)
        
        # Fix for Issue 36: Position squares further right to avoid lecture notes
        self.place_at_grid(sq1, 'B4')
        self.place_at_grid(sq2, 'B5')
        
        self.play(FadeIn(sq1), FadeIn(sq2))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Adding two uniform blocks creates a smoother triangle.
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color(COLOR_TRIANGLE)
        )
        
        # Triangle distribution representing convolution of two uniforms
        triangle = Polygon(
            [-1, -0.5, 0], [0, 0.5, 0], [1, -0.5, 0], 
            fill_opacity=0.5, color=COLOR_TRIANGLE, stroke_width=4
        )
        # Fix for Issue 35: Position triangle further right
        self.place_in_area(triangle, 'B4', 'C5')
        
        self.play(
            ReplacementTransform(VGroup(sq1, sq2), triangle)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # More variables lead toward the famous bell-shaped curve.
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color(COLOR_GAUSSIAN)
        )
        
        # Briefly show a third square being added
        sq3 = Rectangle(height=1, width=1, fill_opacity=0.5, color=COLOR_SQUARE)
        self.place_at_grid(sq3, "B6")
        self.play(FadeIn(sq3))
        self.wait(1)
        
        # Create the Gaussian (Bell Curve)
        gaussian = FunctionGraph(
            lambda x: np.exp(-x**2),
            x_range=[-2, 2],
            color=COLOR_GAUSSIAN
        )
        # Fix for Issue 34: Position Gaussian further right to avoid lecture notes
        self.place_in_area(gaussian, 'B4', 'D6', scale_factor=0.8)
        
        # Transform the components into the Gaussian distribution
        self.play(
            ReplacementTransform(VGroup(triangle, sq3), gaussian)
        )
        self.wait(1)
        
        # Horizontal expansion animation
        self.play(
            gaussian.animate.stretch(1.4, 0)
        )
        self.wait(1)
        
        # Pulse effect on the peak to emphasize smoothness
        self.play(
            gaussian.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        # Final pause
        self.wait(3)
