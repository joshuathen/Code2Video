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
        # Define content
        title = "Prerequisite: The Gatekeeper (Reynolds Number)"
        lines = [
            "The Reynolds number predicts when turbulence begins.",
            "It balances inertial forces against viscous resistance.",
            "High inertia creates chaotic, turbulent motion.",
            "High viscosity keeps flow smooth and steady.",
            "Large Reynolds numbers signal a turbulent state."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        formula_color = "#FFFFFF"
        scale_color = "#A9A9A9"
        inertia_color = "#00FFFF"
        viscosity_color = "#FFA500"
        chaotic_color = "#D3D3D3"
        smooth_color = "#ADD8E6"
        highlight_color = YELLOW
        
        # Pre-create mobjects
        # Fixed scale factor 1.0 (Issue 29)
        formula = MathTex(r"Re = \frac{\rho \cdot u \cdot L}{\mu}", color=formula_color)
        self.place_in_area(formula, 'A4', 'B6', scale_factor=1.0)
        
        # Scale components
        pivot = Triangle(color=scale_color, fill_opacity=1).scale(0.15)
        self.place_at_grid(pivot, 'E5')
        pivot_point = pivot.get_top()
        
        beam = Line(LEFT, RIGHT, color=scale_color, stroke_width=6).scale(1.0)
        beam.move_to(pivot_point + UP * 0.05)
        
        left_pan = Line(beam.get_start(), beam.get_start() + DOWN*0.3, color=scale_color)
        right_pan = Line(beam.get_end(), beam.get_end() + DOWN*0.3, color=scale_color)
        
        # Scale group for rotation
        scale_group = VGroup(beam, left_pan, right_pan)
        
        # Updated positions (Issue 27, 28)
        inertia_label = Text("Inertia", color=inertia_color, font_size=20)
        self.place_at_grid(inertia_label, 'D4')
        
        viscosity_label = Text("Viscosity", color=viscosity_color, font_size=20)
        self.place_at_grid(viscosity_label, 'D6')
        
        turbulent_state_label = Text("TURBULENT", color=RED, font_size=24, weight=BOLD)
        self.place_at_grid(turbulent_state_label, 'F5')

        # Chaotic swirls for background
        swirls = VGroup(*[
            Arc(radius=0.1 + 0.2*np.random.rand(), 
                start_angle=np.random.rand()*TAU, 
                angle=TAU*0.7, 
                color=chaotic_color, 
                stroke_width=1)
            for _ in range(15)
        ])
        self.place_in_area(swirls, 'C4', 'E6', scale_factor=1.0)

        # Smooth lines for background
        smooth_lines = VGroup(*[
            Line(LEFT, RIGHT, color=smooth_color, stroke_width=1).scale(0.8)
            for _ in range(8)
        ]).arrange(DOWN, buff=0.15)
        self.place_in_area(smooth_lines, 'C4', 'E6', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # L1: "The Reynolds number predicts when turbulence begins."
        self.play(self.lecture[0].animate.set_color(highlight_color))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L2: "It balances inertial forces against viscous resistance."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(highlight_color)
        )
        self.play(FadeIn(pivot), Create(scale_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L3: "High inertia creates chaotic, turbulent motion."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color)
        )
        self.play(FadeIn(inertia_label))
        # Tip scale toward Inertia (left)
        self.play(
            Rotate(scale_group, angle=25 * DEGREES, about_point=pivot_point),
            inertia_label.animate.shift(UP * 0.2),
            FadeIn(swirls)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # L4: "High viscosity keeps flow smooth and steady."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(highlight_color)
        )
        self.play(FadeIn(viscosity_label))
        # Tip scale toward Viscosity (right)
        # 25 degrees left -> 20 degrees right (total -45 degrees)
        self.play(
            Rotate(scale_group, angle=-45 * DEGREES, about_point=pivot_point),
            inertia_label.animate.shift(DOWN * 0.4),
            viscosity_label.animate.shift(UP * 0.2),
            ReplacementTransform(swirls, smooth_lines)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # L5: "Large Reynolds numbers signal a turbulent state."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(highlight_color)
        )
        # Highlight 'Re' in formula
        re_part = formula[0][0:2] # 'Re'
        self.play(re_part.animate.set_color(highlight_color).scale(1.2))
        
        # Heavy tip to Inertia (left)
        # 20 degrees right -> 45 degrees left (total 65 degrees)
        self.play(
            Rotate(scale_group, angle=65 * DEGREES, about_point=pivot_point),
            inertia_label.animate.shift(UP * 0.4),
            viscosity_label.animate.shift(DOWN * 0.4),
            ReplacementTransform(smooth_lines, swirls),
            FadeIn(turbulent_state_label)
        )
        self.wait(2)
        
        # Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
