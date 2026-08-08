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
        title = "Defining Superposition: The Linear Combination"
        lines = [
            "Superposition is a linear combination of base states.",
            "The formula is |ψ⟩ = α|0⟩ + β|1⟩.",
            "α and β are called probability amplitudes.",
            "Their squared values must always sum to one.",
            "This allows particles to exist in multiple places."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        COLOR_FORMULA = "#ADD8E6"
        COLOR_COEFF = "#FFFF00"
        COLOR_QUARKY = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Intro to linear combination
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The formula is |ψ⟩ = α|0⟩ + β|1⟩.
        # Break formula into parts for easy coloring
        formula = MathTex(
            r"|\psi\rangle", "=", r"\alpha", r"|0\rangle", "+", r"\beta", r"|1\rangle",
            font_size=36, color=COLOR_FORMULA
        )
        self.place_in_area(formula, 'A2', 'A5')
        
        self.play(
            Write(formula),
            self.lecture[1].animate.set_color(COLOR_FORMULA)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # α and β are called probability amplitudes.
        label_alpha = Text("alpha", font_size=16, color=COLOR_COEFF)
        label_beta = Text("beta", font_size=16, color=COLOR_COEFF)
        
        # Position labels exactly one unit away (B-row for A-row formula)
        label_alpha.move_to(self.grid['B2'] + UP*0.3)
        label_beta.move_to(self.grid['B5'] + UP*0.3)

        self.play(
            formula[2].animate.set_color(COLOR_COEFF),
            formula[5].animate.set_color(COLOR_COEFF),
            FadeIn(label_alpha),
            FadeIn(label_beta),
            self.lecture[2].animate.set_color(COLOR_COEFF)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Their squared values must always sum to one.
        sum_formula = MathTex(
            r"|\alpha|^2 + |\beta|^2 = 1",
            font_size=32, color=WHITE
        )
        self.place_in_area(sum_formula, 'F3', 'F4')
        
        self.play(
            FadeIn(sum_formula, shift=UP),
            self.lecture[3].animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This allows particles to exist in multiple places.
        
        # Rooms
        room_l = Square(side_length=1.5, color=WHITE, stroke_width=2)
        room_r = Square(side_length=1.5, color=WHITE, stroke_width=2)
        label_l = Text("Left Room", font_size=18, color=WHITE)
        label_r = Text("Right Room", font_size=18, color=WHITE)
        
        # Positioning for vertical balance and to avoid occlusion
        self.place_at_grid(room_l, 'C2')
        self.place_at_grid(room_r, 'C5')
        self.place_at_grid(label_l, 'D2')
        self.place_at_grid(label_r, 'D5')
        
        # Quarky representation: A glowing circle
        def create_quarky(opacity=1.0):
            # Using basic shapes to avoid external asset dependency if not explicitly provided
            body = Circle(radius=0.4, fill_opacity=opacity, fill_color=COLOR_QUARKY, stroke_width=0)
            eye_l = Dot(radius=0.05, color=BLACK).move_to(body.get_center() + LEFT*0.1 + UP*0.1)
            eye_r = Dot(radius=0.05, color=BLACK).move_to(body.get_center() + RIGHT*0.1 + UP*0.1)
            # Simple smile
            smile = Arc(radius=0.15, start_angle=-TAU/4 - 0.5, angle=1, color=BLACK).move_to(body.get_center() + DOWN*0.1)
            return VGroup(body, eye_l, eye_r, smile)

        quarky_l = create_quarky(opacity=0.7)
        quarky_r = create_quarky(opacity=0.7)
        
        # Applying scale factor of 0.8 as per Issue 30/31 logic
        self.place_at_grid(quarky_l, 'C2', scale_factor=0.8)
        self.place_at_grid(quarky_r, 'C5', scale_factor=0.8)
        
        self.play(
            Create(room_l), Create(room_r),
            Write(label_l), Write(label_r),
            FadeIn(quarky_l), FadeIn(quarky_r),
            self.lecture[4].animate.set_color(COLOR_QUARKY)
        )
        self.wait(1)
        
        # Adjusting opacities to show shifting superposition
        self.play(
            quarky_l[0].animate.set_fill(opacity=0.9),
            quarky_r[0].animate.set_fill(opacity=0.3),
            run_time=1.5
        )
        self.wait(0.5)
        self.play(
            quarky_l[0].animate.set_fill(opacity=0.5),
            quarky_r[0].animate.set_fill(opacity=0.5),
            run_time=1.5
        )
        self.wait(2)
