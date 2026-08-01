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
        # Data from storyboard
        title_text = "The Power Rule: Expanding Dimensions"
        lecture_lines = [
            "The power rule calculates derivatives for functions like x squared.",
            "Imagine a square growing by a tiny amount of area.",
            "Growth happens along the boundary of the existing shape.",
            "For x squared, the rate of growth is two x.",
            "Power rule: drop the exponent and subtract one."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        BLUE_SQ = "#0000FF"
        WHITE_TXT = "#FFFFFF"
        GREEN_STRIP = "#00FF00"
        YELLOW_LBL = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # The power rule calculates derivatives for functions like x squared.
        self.lecture[0].set_color(BLUE_SQ)
        
        # Initialize mobjects but don't add to scene yet
        square = Square(side_length=2.0, fill_opacity=0.8, fill_color=BLUE_SQ, stroke_color=WHITE)
        self.place_at_grid(square, 'D3')
        
        label_x_left = MathTex("x", color=WHITE_TXT)
        self.place_at_grid(label_x_left, 'D2')
        
        label_x_bottom = MathTex("x", color=WHITE_TXT)
        self.place_at_grid(label_x_bottom, 'E3')
        
        area_text = MathTex("x^2", color=WHITE_TXT)
        self.place_at_grid(area_text, 'D3')
        
        self.play(
            Create(square),
            Write(label_x_left),
            Write(label_x_bottom),
            Write(area_text),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine a square growing by a tiny amount of area.
        self.lecture[1].set_color(GREEN_STRIP)
        
        # Top strip
        top_strip = Rectangle(width=2.0, height=0.2, fill_opacity=0.8, fill_color=GREEN_STRIP, stroke_color=WHITE)
        self.place_at_grid(top_strip, 'C3')
        top_strip.shift(UP * 0.1) # Position precisely above the square
        
        # Right strip
        right_strip = Rectangle(width=0.2, height=2.0, fill_opacity=0.8, fill_color=GREEN_STRIP, stroke_color=WHITE)
        self.place_at_grid(right_strip, 'D4')
        right_strip.shift(RIGHT * 0.1) # Position precisely to the right of the square
        
        self.play(
            FadeIn(top_strip),
            FadeIn(right_strip),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Growth happens along the boundary of the existing shape.
        self.lecture[2].set_color(YELLOW_LBL)
        
        label_strip_top = MathTex("x", color=YELLOW_LBL)
        self.place_at_grid(label_strip_top, 'B3')
        
        label_strip_right = MathTex("x", color=YELLOW_LBL)
        self.place_at_grid(label_strip_right, 'D5')
        
        self.play(
            Write(label_strip_top),
            Write(label_strip_right),
            run_time=1.0
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # For x squared, the rate of growth is two x.
        self.lecture[3].set_color(WHITE_TXT)
        
        # Create objects here to avoid any chance of early appearance
        target_2x = MathTex("2x", color=WHITE_TXT)
        self.place_at_grid(target_2x, 'B5', scale_factor=1.0)
        
        formula = MathTex(r"\frac{d}{dx}(x^2) = 2x", color=WHITE_TXT)
        self.place_in_area(formula, 'A2', 'A5', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(VGroup(label_strip_top, label_strip_right), target_2x),
            FadeIn(formula),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Power rule: drop the exponent and subtract one.
        self.lecture[4].set_color(WHITE_TXT)
        
        # Visual highlight of the connection
        self.play(
            Flash(target_2x, color=WHITE_TXT, flash_radius=0.4),
            Flash(top_strip, color=GREEN_STRIP, flash_radius=0.4),
            Flash(right_strip, color=GREEN_STRIP, flash_radius=0.4),
            run_time=1.5
        )
        self.wait(2)
