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
        # Updated Title and Lecture Lines as per Stage-3 prompt
        title = "Prerequisite: The 2D World of Numbers"
        lines = [
            "We expand the number line into a two-dimensional plane.",
            "Multiplying by i rotates numbers by ninety degrees.",
            "This geometry lets the unit circle trace out pi."
        ]
        self.setup_layout(title, lines)

        # Colors
        REAL_COLOR = "#FFFFFF"
        IMAG_COLOR = "#FF00FF"
        CIRCLE_COLOR = "#FFFF00"

        # Define the coordinate system origin center (Area B2 to E5)
        # Using the grid to determine the center point for the drawing origin
        tl = self.grid["B2"]
        br = self.grid["E5"]
        origin_point = (tl + br) / 2
        unit_size = 0.7  # Scale factor for axes

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(REAL_COLOR)
        
        # Draw a horizontal real axis from -3 to 3 with tick marks (#FFFFFF)
        real_axis = Line(
            start=origin_point + LEFT * 3 * unit_size, 
            end=origin_point + RIGHT * 3 * unit_size, 
            color=REAL_COLOR
        )
        
        ticks = VGroup()
        for x in range(-3, 4):
            tick = Line(
                start=origin_point + RIGHT * x * unit_size + UP * 0.1,
                end=origin_point + RIGHT * x * unit_size + DOWN * 0.1,
                color=REAL_COLOR,
                stroke_width=2
            )
            ticks.add(tick)
        
        real_label = Text("Real", font_size=16, color=REAL_COLOR)
        # Issue 29: Position real_label at D6 with scale 0.7
        self.place_at_grid(real_label, "D6", scale_factor=0.7)
        
        self.play(Create(real_axis), Create(ticks))
        self.play(FadeIn(real_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(IMAG_COLOR)
        
        # Animate a 90-degree rotation of a vector from the real axis to the vertical imaginary axis 'i' (#FF00FF)
        imag_axis = Line(
            start=origin_point + DOWN * 2.5 * unit_size, 
            end=origin_point + UP * 2.5 * unit_size, 
            color=IMAG_COLOR
        )
        
        i_label = Text("i", font_size=24, color=IMAG_COLOR)
        # Issue 30: Scale i_label to 0.8
        self.place_at_grid(i_label, "A3", scale_factor=0.8)
        
        # Rotation vector (starts at 1 on real axis)
        vec_start_point = origin_point + RIGHT * unit_size
        rotation_vector = Arrow(
            start=origin_point, 
            end=vec_start_point, 
            buff=0, 
            color=REAL_COLOR, 
            stroke_width=4
        )
        
        self.play(Create(imag_axis), FadeIn(i_label))
        self.play(Create(rotation_vector))
        self.wait(0.5)
        # Rotate 90 degrees
        self.play(
            Rotate(rotation_vector, angle=PI/2, about_point=origin_point),
            rotation_vector.animate.set_color(IMAG_COLOR),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(CIRCLE_COLOR)
        
        # Draw a Unit Circle (#FFFF00) centered at the origin
        unit_circle = Circle(radius=unit_size, color=CIRCLE_COLOR, stroke_width=4)
        # Centering the circle using the grid-derived origin
        self.place_in_area(unit_circle, "B2", "E5")
        
        # Label for Pi - showing relationship to the circle
        pi_symbol = Text("\u03c0", font_size=28, color=CIRCLE_COLOR)
        # Issue 31: Scale pi_symbol to 0.8
        self.place_at_grid(pi_symbol, "C5", scale_factor=0.8)
        
        self.play(Create(unit_circle))
        self.play(FadeIn(pi_symbol))
        self.wait(3)
