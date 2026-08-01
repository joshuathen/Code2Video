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
        # Setup layout with title and lecture lines
        title_text = "The Sequential Challenge"
        lecture_lines = [
            "We can apply two linear transformations in sequence.",
            "First, matrix A rotates our character.",
            "Then, matrix B stretches the result."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Asset Preparation ===
        # Load the "Cyber-Cat" asset
        cat = ImageMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png").scale(0.3)
        # Initial state: Rotated 90 degrees CCW (as if from the previous step)
        cat.rotate(90 * DEGREES)

        # Coordinate System
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "color": GREY_C},
        ).add_coordinates(label_constructor=Text, font_size=20)
        
        # Fix: Issue 28 - Axes positioning
        self.place_in_area(axes, "C2", "F6", scale_factor=0.8)
        
        # Matrix B (Green)
        matrix_b = Text(
            "B = [[2, 0], [0, 1]]", 
            color="#00FF00",
            font_size=24
        )
        # Fix: Issue 26 - Matrix B positioning
        self.place_in_area(matrix_b, "A1", "A3", scale_factor=0.7)
        
        # Composition Formula (White)
        formula = Text(
            "B(A(v))", 
            color="#FFFFFF",
            font_size=24
        )
        # Fix: Issue 27 - Formula positioning
        self.place_in_area(formula, "A4", "A6", scale_factor=0.8)

        # Initial Position of Cat (relative to axes, starts at rotated position)
        cat.move_to(axes.c2p(-1, 1))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00")) 
        self.play(Create(axes), FadeIn(cat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        # Visual feedback for rotation
        self.play(cat.animate.scale(1.2), run_time=0.3)
        self.play(cat.animate.scale(1/1.2), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        self.play(Write(matrix_b), Write(formula))
        self.wait(0.5)

        # Execute stretching transformation (Matrix B)
        # Cat is at (-1, 1). Stretch factor 2 in x means it moves to (-2, 1).
        self.play(
            cat.animate.stretch(2, dim=0, about_point=axes.c2p(0, 0)),
            run_time=2,
            rate_func=smooth
        )
        self.wait(2)
