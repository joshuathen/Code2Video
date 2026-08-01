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
        self.setup_layout(
            "Prerequisite: The Gateway Number (Reynolds Number)",
            [
                "The Reynolds number tracks the transition to turbulence.",
                "Increasing the ratio of inertia shatters smooth flow.",
                "High Reynolds numbers turn stability into chaotic particles."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Re Slider
        re_tracker = ValueTracker(1)
        slider_line = Line(LEFT, RIGHT, color=WHITE).scale(2)
        # Resolved Issue 30: Position shifted to column 2-6
        self.place_in_area(slider_line, "B2", "B6", scale_factor=0.8)
        
        slider_dot = Dot(slider_line.get_left(), color="#FFFFFF")
        slider_dot.add_updater(lambda d: d.move_to(slider_line.point_from_proportion(re_tracker.get_value()/100000 if re_tracker.get_value() > 1 else 0)))
        
        # Label placed at B1 to avoid overlap
        re_label = Text("Re:", color=WHITE, font_size=28)
        self.place_at_grid(re_label, "B1", scale_factor=0.7)
        re_label.next_to(slider_line, LEFT, buff=0.2)
        
        re_val = always_redraw(lambda: DecimalNumber(re_tracker.get_value(), num_decimal_places=0, color=WHITE, mob_class=Text).scale(0.7).next_to(slider_line, RIGHT))
        
        self.add(slider_line, slider_dot, re_label, re_val)
        self.play(re_tracker.animate.set_value(50000), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        
        # Ratio Visualization
        num_text = Text("Inertial Forces", color="#ADD8E6", font_size=24)
        den_text = Text("Viscous Forces", color="#ADD8E6", font_size=24)
        div_line = Line(LEFT, RIGHT, color="#ADD8E6").match_width(den_text)
        fraction = VGroup(num_text, div_line, den_text).arrange(DOWN, buff=0.1)
        
        ratio_formula = VGroup(
            Text("Re", color="#ADD8E6", font_size=28),
            Text("=", color="#ADD8E6", font_size=28),
            fraction
        ).arrange(RIGHT, buff=0.3)
        
        # Resolved Issue 31: Position shifted to column 2-6
        self.place_in_area(ratio_formula, "C2", "D6", scale_factor=0.8)
        
        self.play(Write(ratio_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F08080"))
        
        # Flow visual - Initial Laminar Line
        laminar_line = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        # Resolved Issue 32: Position shifted to column 2-6
        self.place_in_area(laminar_line, "E2", "F6", scale_factor=0.8)
        
        # Create particles for shatter effect
        num_particles = 40
        particles = VGroup(*[Dot(radius=0.04, color="#FFFFFF") for _ in range(num_particles)])
        for p in particles:
            p.move_to(laminar_line.point_from_proportion(np.random.random()))

        self.play(Create(laminar_line))
        self.wait(0.5)
        
        # Move Re to high value and shatter
        self.play(
            re_tracker.animate.set_value(100000),
            FadeOut(laminar_line, shift=UP*0.1),
            FadeIn(particles),
            run_time=1.5
        )
        
        # Turbulent motion for particles
        def update_turbulent_particles(mob, dt):
            for p in mob:
                # Random walk + drift to right
                p.shift(RIGHT * 1.5 * dt + np.array([
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(-0.5, 0.5),
                    0
                ]) * dt * 5)
                # Wrap around within the grid area roughly (E2-F6 starts at x=1.5, ends at x=5.5)
                if p.get_x() > 5.6:
                    p.set_x(1.4)

        particles.add_updater(update_turbulent_particles)
        self.play(ratio_formula.animate.set_color("#F08080"))
        self.wait(3)
        particles.remove_updater(update_turbulent_particles)
