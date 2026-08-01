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
        # Setup layout with specific title and lecture lines
        self.setup_layout(
            "Defining the Eigen-Relationship", 
            [
                "The relationship is defined as A v equals lambda v.", 
                "The eigenvector v stays on its original line.", 
                "The eigenvalue lambda scales that vector's length."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Equation: A v = λ v
        # Characters: [0]:A, [1]:v, [2]:'=', [3]:λ, [4]:v
        eqn = VGroup(
            Text("A"), Text("v"), Text("="), MarkupText("λ"), Text("v")
        ).arrange(RIGHT, buff=0.2)
        
        # Initial state: White, placed in A1-B6 to use top space (Issue 30)
        eqn.set_color(WHITE)
        self.place_in_area(eqn, 'A1', 'B6', scale_factor=1.2)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            Write(eqn),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Logic: Highlight A (Blue) and λ (Gold). 
        # Show a vector v (Green) in the middle C1-D6 area (Issue 31).
        
        # Setup vector group relative to local origin to be placed in area
        # Span line: 5 units wide
        span_line = Line(LEFT * 2.0, RIGHT * 3.0, color=GRAY, stroke_opacity=0.3)
        # Vector v: 1 unit long, starts at local -1.0 so tail aligns with grid C2 after placement
        vector_v = Arrow(LEFT * 1.0, ORIGIN, buff=0, color="#00FF00", stroke_width=6)
        
        vector_visualization = VGroup(span_line, vector_v)
        # Move group to centered C1-D6 area (Issue 31)
        self.place_in_area(vector_visualization, 'C1', 'D6', scale_factor=1.0)
        
        # Place label 'v' at grid point C2 (Issue 32)
        v_label = Text("v", color="#00FF00", font_size=24)
        self.place_at_grid(v_label, 'C2', scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color("#00FF00"),
            eqn[0].animate.set_color("#0000FF"), # Matrix A to Blue
            eqn[3].animate.set_color("#FFD700"), # Lambda to Gold
            Create(span_line),
            GrowArrow(vector_v),
            FadeIn(v_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Logic: Animate the green vector 'v' scaling by factor 3.
        
        # The tail of vector_v is fixed at self.grid["C2"]'s x-coordinate (1.5) 
        # and visualization's y-coordinate (-0.3).
        v_tail_pos = vector_v.get_start()
        vector_v_scaled = Arrow(
            start=v_tail_pos, 
            end=v_tail_pos + RIGHT * 3.0, 
            buff=0, 
            color="#00FF00", 
            stroke_width=6
        )
        
        # Position label '3v' at grid point C4 (Issue 32)
        v_label_3v = Text("3v", color="#FFD700", font_size=24)
        self.place_at_grid(v_label_3v, 'C4', scale_factor=0.8)

        self.play(
            self.lecture[2].animate.set_color("#FFD700"),
            Transform(vector_v, vector_v_scaled),
            Transform(v_label, v_label_3v),
            run_time=2
        )
        self.wait(2)
