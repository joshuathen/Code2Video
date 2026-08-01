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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Transformation: The Dual Number Line", [
            "Imagine two parallel lines representing input and output.",
            "Arrows map points from the input to the output.",
            "Watch a tiny change, dx, on the input line.",
            "This results in a corresponding change, dy, on output.",
            "The ratio dy over dx is the local stretch."
        ])

        # Colors
        COLOR_INPUT = WHITE
        COLOR_OUTPUT = WHITE
        COLOR_ARROW = "#00FFFF" # Cyan
        COLOR_DX = "#00FFFF" # Cyan
        COLOR_DY = "#FF00FF" # Magenta
        COLOR_RATIO = "#FFFF00" # Yellow

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Number lines - B1 to B6 and E1 to E6
        input_line = Line(self.grid["B1"], self.grid["B6"], color=COLOR_INPUT)
        output_line = Line(self.grid["E1"], self.grid["E6"], color=COLOR_OUTPUT)
        
        input_label = Text("Input X", font_size=20, color=COLOR_INPUT)
        # Fix for Issue 28: Move from A1 to A2
        self.place_at_grid(input_label, "A2", scale_factor=0.8)
        
        output_label = Text("Output Y", font_size=20, color=COLOR_OUTPUT)
        # Fix for Issue 29: Move from D1 to D2
        self.place_at_grid(output_label, "D2", scale_factor=0.8)

        self.play(Create(input_line), Create(output_line), Write(input_label), Write(output_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Points mapping f(x) = sin(x) at x=0
        # Grid B3 is our x=0 point on input line
        points_x_rel = np.linspace(-1.5, 1.5, 7)
        arrows = VGroup()
        for px in points_x_rel:
            start = self.grid["B3"] + np.array([px, 0, 0])
            end = self.grid["E3"] + np.array([np.sin(px), 0, 0])
            arrow = Arrow(start, end, buff=0, color=COLOR_ARROW, stroke_width=2, tip_length=0.1)
            arrows.add(arrow)
            
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # dx at x=0 (B3)
        dx_val = 0.5 
        dx_line = Line(self.grid["B3"], self.grid["B3"] + np.array([dx_val, 0, 0]), color=COLOR_DX, stroke_width=8)
        dx_label = MathTex("dx", font_size=24, color=COLOR_DX).next_to(dx_line, UP, buff=0.1)
        
        self.play(Create(dx_line), Write(dx_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # dy = sin(dx)
        dy_val = np.sin(dx_val)
        dy_line = Line(self.grid["E3"], self.grid["E3"] + np.array([dy_val, 0, 0]), color=COLOR_DY, stroke_width=8)
        dy_label = MathTex("dy", font_size=24, color=COLOR_DY).next_to(dy_line, DOWN, buff=0.1)
        
        mapping_line = Arrow(
            self.grid["B3"] + np.array([dx_val, 0, 0]), 
            self.grid["E3"] + np.array([dy_val, 0, 0]), 
            color=COLOR_ARROW, buff=0, stroke_width=2, tip_length=0.1
        )
        
        self.play(Create(dy_line), Write(dy_label), Create(mapping_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        ratio_text = MathTex(r"\text{Ratio } \frac{dy}{dx} \approx 1", font_size=28, color=COLOR_RATIO)
        # Fix for Issue 30: Use place_in_area instead of place_at_grid
        self.place_in_area(ratio_text, "F3", "F4", scale_factor=1.0)
        
        # Pulse animation
        self.play(
            dx_line.animate.set_stroke(width=12),
            dy_line.animate.set_stroke(width=12),
            Write(ratio_text)
        )
        self.play(
            dx_line.animate.set_stroke(width=8),
            dy_line.animate.set_stroke(width=8),
        )
        self.wait(2)
        
        self.lecture[4].set_color(WHITE)
        self.wait(2)
