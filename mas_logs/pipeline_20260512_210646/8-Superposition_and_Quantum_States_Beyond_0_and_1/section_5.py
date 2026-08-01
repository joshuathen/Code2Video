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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with teaching script lines
        lecture_lines = [
            'Measuring a quantum state triggers a transformation.',
            'The state vector collapses into a definite outcome.',
            'It may snap to the vertical zero state.',
            'Or it can snap to the horizontal one.',
            'Probability amplitudes determine the final observed state.'
        ]
        self.setup_layout("The Moment of Truth: Measurement", lecture_lines)

        # Colors
        COLOR_EYE = "#00BFFF"
        COLOR_VECTOR = "#FFFF00"
        COLOR_ZERO = "#FF0000"
        COLOR_ONE = "#00FF00"
        COLOR_TEXT = "#FFFFFF"

        # Positioning constants
        origin_pos = self.grid["D2"]
        y_axis_end = self.grid["B2"]
        x_axis_end = self.grid["D4"]
        diagonal_end = self.grid["C3"]
        eye_pos = "A3"

        # Coordinate elements
        y_axis = Arrow(origin_pos, y_axis_end, buff=0, color=WHITE)
        x_axis = Arrow(origin_pos, x_axis_end, buff=0, color=WHITE)
        
        label_0 = Text("|0>", font_size=20, color=WHITE)
        self.place_at_grid(label_0, "A2")
        
        label_1 = Text("|1>", font_size=20, color=WHITE)
        self.place_at_grid(label_1, "D5")
        
        axes_group = VGroup(y_axis, x_axis, label_0, label_1)

        # === Animation for Lecture Line 1 ===
        # Stage: Measurement_Eye
        self.lecture[0].set_color(COLOR_EYE)
        
        # Eye Icon
        eye_center = self.grid[eye_pos]
        eye_outer = ArcBetweenPoints(
            eye_center + LEFT*0.4, 
            eye_center + RIGHT*0.4, 
            radius=0.6, color=COLOR_EYE
        )
        eye_lower = ArcBetweenPoints(
            eye_center + RIGHT*0.4, 
            eye_center + LEFT*0.4, 
            radius=0.6, color=COLOR_EYE
        )
        pupil = Circle(radius=0.15, color=COLOR_EYE, fill_opacity=1).move_to(eye_center)
        eye_icon = VGroup(eye_outer, eye_lower, pupil)

        # Vector |psi>
        vector_psi = Arrow(origin_pos, diagonal_end, buff=0, color=COLOR_VECTOR)
        label_psi = Text("|psi>", font_size=18, color=COLOR_VECTOR).next_to(diagonal_end, UR, buff=0.1)

        self.play(Create(axes_group))
        self.play(Create(vector_psi), Write(label_psi))
        self.play(FadeIn(eye_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Stage: Wavefunction_Collapse
        self.lecture[1].set_color(COLOR_VECTOR)
        
        # Vibration/Flicker
        for _ in range(4):
            self.play(vector_psi.animate.rotate(0.1, about_point=origin_pos), run_time=0.08)
            self.play(vector_psi.animate.rotate(-0.2, about_point=origin_pos), run_time=0.08)
            self.play(vector_psi.animate.rotate(0.1, about_point=origin_pos), run_time=0.08)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Stage: Snap_To_Zero
        self.lecture[2].set_color(COLOR_ZERO)
        
        # Snap to vertical |0> axis
        target_vector_0 = Arrow(origin_pos, y_axis_end, buff=0, color=COLOR_ZERO)
        self.play(
            ReplacementTransform(vector_psi, target_vector_0),
            label_psi.animate.next_to(y_axis_end, LEFT, buff=0.2).set_color(COLOR_ZERO),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Stage: Snap_To_One
        self.lecture[3].set_color(COLOR_ONE)
        
        # Reset to diagonal
        vector_psi_reset = Arrow(origin_pos, diagonal_end, buff=0, color=COLOR_VECTOR)
        self.play(
            ReplacementTransform(target_vector_0, vector_psi_reset),
            label_psi.animate.next_to(diagonal_end, UR, buff=0.1).set_color(COLOR_VECTOR),
            run_time=0.5
        )
        
        # Snap to horizontal |1> axis
        target_vector_1 = Arrow(origin_pos, x_axis_end, buff=0, color=COLOR_ONE)
        self.play(
            ReplacementTransform(vector_psi_reset, target_vector_1),
            label_psi.animate.next_to(x_axis_end, DOWN, buff=0.2).set_color(COLOR_ONE),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Stage: Probabilistic_Result
        self.lecture[4].set_color(COLOR_TEXT)
        
        result_text = Text("Measurement forces a definite state", font_size=24, color=COLOR_TEXT)
        # Fix for Issue 38 and 39: Move to E2-E5 and use scale factor 0.65
        self.place_in_area(result_text, "E2", "E5", scale_factor=0.65)
        
        self.play(Write(result_text))
        self.wait(2)
