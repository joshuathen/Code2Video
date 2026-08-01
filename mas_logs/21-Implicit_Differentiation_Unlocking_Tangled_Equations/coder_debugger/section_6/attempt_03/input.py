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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with lecture lines
        lecture_lines = [
            'Space probes follow elliptical paths around the sun.',
            'Implicit differentiation helps find their exact trajectory.',
            'We calculate velocity without messy square root formulas.'
        ]
        self.setup_layout("Application: The Solar Orbit", lecture_lines)

        # Orbit parameters
        a, b = 1.8, 1.1
        orbit_color = "#444444"
        probe_color = "#FFFFFF"
        velocity_color = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create Ellipse
        ellipse = Ellipse(width=2*a, height=2*b, color=orbit_color)
        self.place_in_area(ellipse, 'B2', 'E6')
        orbit_center = ellipse.get_center()
        
        # Add a "Sun" at one focus
        c = np.sqrt(a**2 - b**2)
        sun = Dot(color=YELLOW, radius=0.15).move_to(orbit_center + np.array([-c, 0, 0]))
        sun_glow = Dot(color=YELLOW, radius=0.25, fill_opacity=0.3).move_to(sun.get_center())
        
        # Space Probe
        probe = Dot(color=probe_color, radius=0.08)
        # Orbit tracker
        t_tracker = ValueTracker(0)
        
        def update_probe(p):
            t = t_tracker.get_value()
            pos = orbit_center + np.array([a * np.cos(t), b * np.sin(t), 0])
            p.move_to(pos)
            
        probe.add_updater(update_probe)
        
        self.add(ellipse, sun, sun_glow, probe)
        # Initial orbit animation
        self.play(t_tracker.animate.set_value(TAU), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN)
        )
        
        # Velocity Vector (Tangent)
        velocity_vector = Arrow(
            start=ORIGIN, end=RIGHT, color=velocity_color, 
            buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.3
        )
        
        def update_velocity(v):
            t = t_tracker.get_value()
            pos = orbit_center + np.array([a * np.cos(t), b * np.sin(t), 0])
            # Tangent direction is derivative of (a cos t, b sin t) -> (-a sin t, b cos t)
            direction = np.array([-a * np.sin(t), b * np.cos(t), 0])
            # Normalize and scale
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction) * 1.2
            v.put_start_and_end_on(pos, pos + direction)
            
        velocity_vector.add_updater(update_velocity)
        
        self.add(velocity_vector)
        # Orbit with velocity vector
        self.play(t_tracker.animate.set_value(2*TAU), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(TEAL)
        )
        
        # Fixed FileNotFoundError: [Errno 2] No such file or directory: 'latex'
        # Replaced MathTex with Text to avoid reliance on external LaTeX distribution
        formula = Text(
            "dy/dx = -(b^2 x) / (a^2 y)", 
            color=TEAL
        )
        self.place_at_grid(formula, 'A4', scale_factor=0.9)
        
        # Labeling the components for clarity using Text instead of MathTex
        eqn_label = Text("x^2/a^2 + y^2/b^2 = 1", color=orbit_color, font_size=24)
        self.place_at_grid(eqn_label, 'A2', scale_factor=0.8)
        
        self.play(Write(eqn_label), FadeIn(formula))
        
        # Flash the velocity vector to emphasize trajectory
        self.play(Flash(velocity_vector, color=velocity_color, num_lines=12, flash_radius=0.5))
        
        # Final orbit rotation
        self.play(t_tracker.animate.set_value(3*TAU), run_time=4, rate_func=linear)
        self.wait(2)