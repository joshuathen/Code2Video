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
        self.setup_layout(
            "The Sequence: Two Transformations",
            [
                "What if we apply two transformations in a row?",
                "First, we apply matrix A to our vector.",
                "Then, we apply matrix B to that result.",
                "This sequence is called a composition of functions.",
                "The output of A becomes the input for B."
            ]
        )

        # Colors
        YELLOW_A = "#FFFFE0"
        BLUE_B = "#ADD8E6"
        
        # Central origin for the coordinate system visuals (D4 on the right side grid)
        plane_center = self.grid["D4"]
        
        # Coordinate system plane
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=3,
            y_length=3,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        ).move_to(plane_center)
        
        # Define vector positions relative to the plane center
        v_coords = np.array([1, 1, 0])
        Av_coords = np.array([-1, 1, 0])   # Result of 90-degree CCW rotation
        BAv_coords = np.array([-1, 2, 0])  # Result of vertical stretch by 2
        
        # Initial Vector and Label
        vec_v = Arrow(plane_center, plane_center + v_coords, buff=0, color=WHITE)
        # Issue 31 Fix: Label 'v' at B6
        v_label = Text("v", slant=ITALIC, color=WHITE)
        self.place_at_grid(v_label, 'B6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        # Introduce the grid and the initial vector
        self.lecture[0].set_color(WHITE)
        self.play(Create(plane), GrowArrow(vec_v), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # First transformation: Matrix A (90-degree rotation)
        self.lecture[1].set_color(YELLOW_A)
        
        vec_Av = Arrow(plane_center, plane_center + Av_coords, buff=0, color=YELLOW_A)
        # Issue 32 Fix: Label 'A(v)' at C3
        Av_label = Text("A(v)", color=YELLOW_A)
        self.place_at_grid(Av_label, 'C3', scale_factor=0.8)
        
        self.play(
            plane.animate.apply_matrix([[0, -1], [1, 0]]),
            Transform(vec_v, vec_Av),
            FadeIn(Av_label),
            FadeOut(v_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Second transformation: Matrix B (Vertical stretch by 2)
        self.lecture[2].set_color(BLUE_B)
        
        vec_BAv = Arrow(plane_center, plane_center + BAv_coords, buff=0, color=BLUE_B)
        # Issue 30 Fix: Label 'B(A(v))' in area A3 to B4
        BAv_label = Text("B(A(v))", color=BLUE_B)
        self.place_in_area(BAv_label, 'A3', 'B4', scale_factor=0.7)
        
        self.play(
            plane.animate.apply_matrix([[1, 0], [0, 2]]),
            Transform(vec_v, vec_BAv),
            FadeIn(BAv_label),
            FadeOut(Av_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Define composition - output of A is input for B
        self.lecture[3].set_color(WHITE)
        
        # Visual markers for start, middle, and end states
        dot_start = Dot(plane_center + v_coords, color=WHITE)
        dot_mid = Dot(plane_center + Av_coords, color=YELLOW_A)
        dot_end = Dot(plane_center + BAv_coords, color=BLUE_B)
        
        self.play(FadeIn(dot_start), FadeIn(dot_mid), FadeIn(dot_end))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the sequence flow with trails
        self.lecture[4].set_color(WHITE)
        
        # Trail arrows indicating the step-by-step movement
        trail1 = CurvedArrow(plane_center + v_coords, plane_center + Av_coords, color=YELLOW_A, angle=-PI/2)
        trail2 = CurvedArrow(plane_center + Av_coords, plane_center + BAv_coords, color=BLUE_B, angle=PI/4)
        
        self.play(Create(trail1))
        self.play(Create(trail2))
        self.wait(2)
        
        # Final cleanup of sequence markers
        self.play(
            FadeOut(dot_start), FadeOut(dot_mid), FadeOut(dot_end),
            FadeOut(trail1), FadeOut(trail2)
        )
        self.wait(1)
