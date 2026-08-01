from manim import *
import numpy as np

# Fix for KeyError: 'iπ' - Manim's config.get_dir() fails if the input file path 
# contains curly braces because it attempts to format the path.
config.media_dir = "media"
if hasattr(config, "input_file") and config.input_file:
    config.input_file = str(config.input_file).replace("{", "").replace("}", "")

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
        # Initial layout setup
        self.setup_layout(
            "Prerequisite: The Geometry of 'i'",
            [
                "We usually think of i as the square root of -1.",
                "Geometrically, multiplying by i causes a 90-degree rotation.",
                "It moves us from the real to the imaginary axis."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight line 1 and show the standard definition of i
        self.lecture[0].set_color(YELLOW)
        
        # Use Text to avoid FileNotFoundError: 'latex'
        formula_i = Text("i = √-1", color=YELLOW)
        
        # Anchor formula_i to grid A3
        self.place_at_grid(formula_i, 'A3', scale_factor=1.2)
        
        self.play(Write(formula_i))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition to geometric interpretation
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # FIX: Use NumberPlane instead of ComplexPlane to avoid the hardcoded 
        # LaTeX dependency for the 'i' unit in ComplexPlane's y-axis labels.
        # We also pass label_constructor=Text to add_coordinates.
        complex_plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.4}
        ).add_coordinates(label_constructor=Text)
        
        self.place_in_area(complex_plane, 'B1', 'F6', scale_factor=0.9)
        
        origin = complex_plane.c2p(0, 0)
        start_pos = complex_plane.c2p(1, 0)
        
        # Elements to animate rotation
        dot = Dot(start_pos, color=BLUE)
        vec = Vector(start_pos - origin, color=BLUE).shift(origin)
        
        # Anchor rotation label to grid C4
        rotation_label = Text("90°", color=BLUE)
        self.place_at_grid(rotation_label, 'C4', scale_factor=0.8)
        
        # Create arc path for the rotation
        arc = Arc(
            radius=complex_plane.get_x_unit_size(), 
            start_angle=0, 
            angle=PI/2, 
            arc_center=origin, 
            color=BLUE
        )

        self.play(Create(complex_plane))
        self.play(FadeIn(dot), GrowArrow(vec))
        self.play(
            Rotate(dot, angle=PI/2, about_point=origin),
            Rotate(vec, angle=PI/2, about_point=origin),
            Create(arc),
            Write(rotation_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Emphasize the shift from real to imaginary axis
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        # Visual markers for axes labeled using grid system
        real_axis_label = Text("Real Axis", font_size=20, color=GREEN)
        imag_axis_label = Text("Imaginary Axis", font_size=20, color=GREEN)
        
        self.place_at_grid(real_axis_label, 'F5', scale_factor=0.8)
        self.place_at_grid(imag_axis_label, 'B4', scale_factor=0.8)
        
        self.play(
            Write(real_axis_label),
            Write(imag_axis_label),
            complex_plane.x_axis.animate.set_color(GREEN),
            complex_plane.y_axis.animate.set_color(GREEN)
        )
        self.wait(2)
