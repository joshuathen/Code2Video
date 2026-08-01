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
        # Setup the scene with updated lecture lines
        lecture_lines = [
            'Quantum state Psi combines basic states zero and one.',
            "Amplitudes alpha and beta determine each state's influence.",
            'These amplitudes are complex numbers describing the system.',
            'Squaring these amplitudes reveals the probability of each outcome.',
            'Quark the Cat exists as a combination of paths.'
        ]
        self.setup_layout("Defining the Quantum State (Psi)", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Equation |ψ⟩ = α|0⟩ + β|1⟩ fades in center screen (#FFFFFF).
        # Fix: Moved to A2-B5, scale 1.1 (Issue 44)
        main_eq = VGroup(
            Text("|ψ⟩", font_size=32), 
            Text("=", font_size=32), 
            Text("α", font_size=32), 
            Text("|0⟩", font_size=32), 
            Text("+", font_size=32), 
            Text("β", font_size=32), 
            Text("|1⟩", font_size=32)
        ).arrange(RIGHT, buff=0.15).set_color(WHITE)
        
        self.place_in_area(main_eq, "A2", "B5", scale_factor=1.1)
        
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            Write(main_eq)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The variables α (#FF00FF) and β (#00FFFF) grow and pulse.
        alpha_part = main_eq[2]
        beta_part = main_eq[5]
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(
            alpha_part.animate.scale(1.3).set_color("#FF00FF"),
            beta_part.animate.scale(1.3).set_color("#00FFFF"),
            Flash(alpha_part, color="#FF00FF"),
            Flash(beta_part, color="#00FFFF")
        )
        self.wait(0.5)
        self.play(
            alpha_part.animate.scale(1/1.3),
            beta_part.animate.scale(1/1.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The text 'Complex Amplitudes' (#B0E0E6) appears below the equation.
        # Fix: Moved to D3-D4, scale 0.7 (Issue 45)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        complex_label = Text("Complex Amplitudes", font_size=28, color="#B0E0E6")
        self.place_in_area(complex_label, "D3", "D4", scale_factor=0.7)
        self.play(FadeIn(complex_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The formula |α|² + |β|² = 1 appears and glows green (#00FF00).
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        prob_eq = Text("|α|² + |β|² = 1", font_size=28, color="#00FF00")
        self.place_in_area(prob_eq, "E2", "E5", scale_factor=1.0)
        
        self.play(FadeIn(prob_eq))
        self.play(Indicate(prob_eq, color="#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Fork in a path appears with a white circle moving along both branches.
        # Fix: cat_eq moved to F2-F5, scale 0.8 (Issue 46)
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        cat_eq = Text(
            "|Quark⟩ ≈ 0.7|Left⟩ + 0.7|Right⟩",
            font_size=24,
            color=WHITE
        )
        self.place_in_area(cat_eq, "F2", "F5", scale_factor=0.8)

        # Path visualization (Fork) in Row C
        start_p = self.grid["C1"]
        junction = self.grid["C2"]
        end_u = self.grid["C5"] + UP * 0.3
        end_d = self.grid["C5"] + DOWN * 0.3

        stem = Line(start_p, junction, color=GRAY)
        branch_u = Line(junction, end_u, color=GRAY)
        branch_d = Line(junction, end_d, color=GRAY)
        fork_group = VGroup(stem, branch_u, branch_d)

        dot = Dot(color=WHITE)
        dot.move_to(start_p)

        self.play(Create(fork_group), Write(cat_eq))
        
        # Animation: Move dot to junction, then split and move along both branches
        self.play(dot.animate.move_to(junction), run_time=0.8)
        
        dot_u = dot.copy()
        dot_d = dot.copy()
        self.add(dot_u, dot_d)
        self.remove(dot)
        
        self.play(
            MoveAlongPath(dot_u, branch_u),
            MoveAlongPath(dot_d, branch_d),
            run_time=2,
            rate_func=linear
        )
        
        self.wait(2)
        
        # Reset colors and finish
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
