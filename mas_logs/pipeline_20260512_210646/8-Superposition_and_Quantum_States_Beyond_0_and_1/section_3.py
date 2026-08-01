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
        # Initialize Scene with exact prompt lecture lines
        lecture_lines = [
            'Superposition is written as a linear combination.',
            'Alpha and beta are the probability amplitudes.',
            'The squared amplitudes must sum to one.',
            'We project the state vector onto the axes.',
            'Quarky follows the diagonal path of the vector.'
        ]
        self.setup_layout("The Recipe of Superposition", lecture_lines)

        # Colors
        COLOR_CYAN = "#00FFFF"
        COLOR_ORANGE = "#FFA500"
        COLOR_LIME = "#7FFF00"
        COLOR_GRAY = "#888888"
        COLOR_YELLOW = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Stage: Superposition_Equation
        # Equation: |ψ⟩ = α|0⟩ + β|1⟩
        eq_parts = VGroup(
            Text("|ψ⟩", color=COLOR_CYAN),
            Text("=", color=COLOR_CYAN),
            Text("α", color=COLOR_CYAN),
            Text("|0⟩", color=COLOR_CYAN),
            Text("+", color=COLOR_CYAN),
            Text("β", color=COLOR_CYAN),
            Text("|1⟩", color=COLOR_CYAN)
        ).arrange(RIGHT, buff=0.15)
        
        # Issue 35 fix: A2 to A5 instead of A1 to A6
        self.place_in_area(eq_parts, "A2", "A5", scale_factor=1.0)
        
        self.play(
            self.lecture[0].animate.set_color(COLOR_CYAN),
            FadeIn(eq_parts),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Stage: Alpha_Beta_Labels
        # Highlight α and β with labels 'Probability Amplitudes'
        label_text = Text("Probability Amplitudes", color=COLOR_ORANGE, font_size=20)
        # Issue 33 fix: B3 to B5, scale 0.6
        self.place_in_area(label_text, "B3", "B5", scale_factor=0.6)
        
        # Create small lines/arrows pointing to alpha and beta
        arrow_alpha = Line(label_text.get_top() + LEFT*0.4, eq_parts[2].get_bottom(), color=COLOR_ORANGE, stroke_width=2)
        arrow_beta = Line(label_text.get_top() + RIGHT*0.4, eq_parts[5].get_bottom(), color=COLOR_ORANGE, stroke_width=2)
        
        self.play(
            self.lecture[1].animate.set_color(COLOR_ORANGE),
            Write(label_text),
            Create(arrow_alpha),
            Create(arrow_beta),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Stage: Normalization_Formula
        # Formula: |α|² + |β|² = 1
        norm_parts = VGroup(
            Text("|α|²", color=COLOR_LIME),
            Text("+", color=COLOR_LIME),
            Text("|β|²", color=COLOR_LIME),
            Text("=", color=COLOR_LIME),
            Text("1", color=COLOR_LIME)
        ).arrange(RIGHT, buff=0.15)
        
        # Issue 34 fix: F2 to F5, scale 0.8
        self.place_in_area(norm_parts, "F2", "F5", scale_factor=0.8)
        
        self.play(
            self.lecture[2].animate.set_color(COLOR_LIME),
            FadeIn(norm_parts),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Stage: Component_Projections
        # Vector visualization (shifted slightly up to avoid overlapping norm_parts)
        origin = self.grid["E2"]
        axis_0_end = self.grid["E5"]
        axis_1_end = self.grid["C2"]
        vector_tip = self.grid["C4"]
        
        axis_0 = Line(origin, axis_0_end, color=WHITE)
        axis_1 = Line(origin, axis_1_end, color=WHITE)
        axis_label_0 = Text("|0⟩", font_size=18).next_to(axis_0_end, RIGHT, buff=0.1)
        axis_label_1 = Text("|1⟩", font_size=18).next_to(axis_1_end, UP, buff=0.1)
        
        vector_psi = Arrow(start=origin, end=vector_tip, buff=0, color=COLOR_CYAN, stroke_width=4)
        vector_label = Text("|ψ⟩", color=COLOR_CYAN, font_size=18).next_to(vector_tip, UR, buff=0.05)
        
        # Projections
        proj_h = DashedLine(vector_tip, [origin[0], vector_tip[1], 0], color=COLOR_GRAY)
        proj_v = DashedLine(vector_tip, [vector_tip[0], origin[1], 0], color=COLOR_GRAY)
        
        self.play(
            self.lecture[3].animate.set_color(COLOR_GRAY),
            Create(axis_0),
            Create(axis_1),
            Write(axis_label_0),
            Write(axis_label_1),
            FadeIn(vector_psi),
            Write(vector_label),
            run_time=1.5
        )
        self.play(Create(proj_h), Create(proj_v), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Stage: Quarky_Path
        # Yellow circle 'Quarky' moves along the diagonal vector
        quarky = Circle(radius=0.12, color=COLOR_YELLOW, fill_opacity=1.0)
        quarky.move_to(origin)
        quarky_label = Text("Quarky", color=COLOR_YELLOW, font_size=16).next_to(quarky, DOWN, buff=0.1)
        
        self.play(
            self.lecture[4].animate.set_color(COLOR_YELLOW),
            FadeIn(quarky),
            FadeIn(quarky_label),
            run_time=0.5
        )
        
        # Follow the vector
        self.play(
            quarky.animate.move_to(vector_tip),
            quarky_label.animate.next_to(vector_tip, DOWN, buff=0.1),
            run_time=2,
            rate_func=linear
        )
        self.wait(2)
