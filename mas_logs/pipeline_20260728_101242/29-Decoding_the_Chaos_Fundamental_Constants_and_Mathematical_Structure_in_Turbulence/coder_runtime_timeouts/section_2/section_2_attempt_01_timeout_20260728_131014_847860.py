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
        self.setup_layout("Prerequisite: The Gateway Number (Reynolds Number)", [
            "The Reynolds number compares inertial forces to viscous forces.",
            "High Reynolds numbers signify inertia dominating the fluid's stickiness.",
            "Beyond a critical threshold, flow inevitably becomes turbulent."
        ])

        # === Animation for Lecture Line 1 ===
        # Display formula Re = rho*v*L / mu (#FFFFFF). Highlight rho*v*L as 'Inertia' (#90EE90) and mu as 'Viscosity' (#FFB6C1).
        
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        re_formula = MathTex(
            "Re", "=", "{ \\rho v L", "\\over", "\\mu }",
            font_size=48, color=WHITE
        )
        self.place_in_area(re_formula, "B3", "C4", scale_factor=1.2)
        
        # Color specific parts
        # Re [0], = [1], \rho v L [2], \over [3], \mu [4]
        inertia_part = re_formula[2]
        viscosity_part = re_formula[4]
        
        inertia_label = Text("Inertia", font_size=20, color="#90EE90")
        viscosity_label = Text("Viscosity", font_size=20, color="#FFB6C1")
        
        self.place_at_grid(inertia_label, "A3", scale_factor=1.0)
        inertia_label.next_to(inertia_part, UP, buff=0.2)
        
        self.place_at_grid(viscosity_label, "D3", scale_factor=1.0)
        viscosity_label.next_to(viscosity_part, DOWN, buff=0.2)

        self.play(Write(re_formula))
        self.play(
            inertia_part.animate.set_color("#90EE90"),
            inertia_label.animate.set_color("#90EE90"),
            run_time=0.5
        )
        self.play(FadeIn(inertia_label))
        self.play(
            viscosity_part.animate.set_color("#FFB6C1"),
            viscosity_label.animate.set_color("#FFB6C1"),
            run_time=0.5
        )
        self.play(FadeIn(viscosity_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a fast-moving shape with chaotic wakes behind it (#ADD8E6). Label: 'High Re: Inertia Wins' (#FFFFFF).
        
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        
        high_re_shape = Circle(radius=0.2, color="#ADD8E6", fill_opacity=0.8)
        self.place_at_grid(high_re_shape, "B2")
        
        high_re_label = Text("High Re: Inertia Wins", font_size=18, color=WHITE)
        self.place_at_grid(high_re_label, "A2", scale_factor=1.0)
        
        # Create wake particles
        wakes = VGroup(*[
            Dot(point=high_re_shape.get_center(), radius=0.03, color="#ADD8E6", fill_opacity=0.5)
            for _ in range(20)
        ])
        
        def update_wake(group, dt):
            # Move existing particles back and add randomness
            for particle in group:
                particle.shift(LEFT * 1.5 * dt + UP * (np.random.rand() - 0.5) * 0.5 * dt)
                particle.set_opacity(max(0, particle.get_opacity() - 0.8 * dt))
                if particle.get_opacity() <= 0:
                    particle.move_to(high_re_shape.get_center())
                    particle.set_opacity(0.5)

        wakes.add_updater(update_wake)
        
        self.play(FadeIn(high_re_shape), FadeIn(high_re_label))
        self.add(wakes)
        self.play(high_re_shape.animate.shift(RIGHT * 3), run_time=3, rate_func=linear)
        wakes.remove_updater(update_wake)
        self.play(FadeOut(high_re_shape), FadeOut(wakes), FadeOut(high_re_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a small circle moving slowly in straight lines (#ADD8E6). Label: 'Low Re: Viscosity Wins' (#FFFFFF).
        
        self.play(self.lecture[2].animate.set_color("#ADD8E6"))
        
        low_re_shape = Circle(radius=0.15, color="#ADD8E6", fill_opacity=0.8)
        self.place_at_grid(low_re_shape, "E2")
        
        low_re_label = Text("Low Re: Viscosity Wins", font_size=18, color=WHITE)
        self.place_at_grid(low_re_label, "D2", scale_factor=1.0)
        
        # Straight path indicator
        path_line = Line(start=self.grid["E2"], end=self.grid["E5"], color="#ADD8E6", stroke_width=1, stroke_opacity=0.3)
        
        self.play(FadeIn(low_re_shape), FadeIn(low_re_label), Create(path_line))
        self.play(low_re_shape.animate.move_to(self.grid["E5"]), run_time=4, rate_func=linear)
        self.wait(1)
