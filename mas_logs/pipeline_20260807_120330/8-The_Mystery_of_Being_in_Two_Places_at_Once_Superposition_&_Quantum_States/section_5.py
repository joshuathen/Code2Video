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
        # Data from shared state
        title = "Visualizing the Qubit: The Bloch Sphere"
        lecture_lines = [
            "A qubit is visualized on a 3D globe.",
            "We call this representation the Bloch Sphere.",
            "North and South poles represent base states.",
            "The equator represents equal superpositions of both.",
            "This sphere allows for complex quantum computations."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_GLOBE = "#FFFFFF"
        COLOR_NAME = "#ADD8E6"
        COLOR_POLES = "#FFFFE0"
        COLOR_EQUATOR = "#FF00FF"
        COLOR_COMPUTATION = "#90EE90"

        # === Animation for Lecture Line 1 ===
        # A qubit is visualized on a 3D globe.
        self.lecture[0].set_color(COLOR_GLOBE)
        
        # Asset integration: globe.svg
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg]
        globe_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        globe_asset.set_color(COLOR_GLOBE).set_stroke(width=1)
        
        # We still need the vertical axis for context of poles
        # Radius estimate for the SVG when scaled to fit area
        vertical_axis = Line(UP * 1.5, DOWN * 1.5, color=COLOR_GLOBE, stroke_width=1).set_stroke(opacity=0.3)
        
        self.sphere_group = VGroup(globe_asset, vertical_axis)
        # Issue 35 fix: place_in_area('C2', 'F5', scale_factor=0.9)
        self.place_in_area(self.sphere_group, 'C2', 'F5', scale_factor=0.9)
        
        self.play(
            Create(globe_asset), 
            Create(vertical_axis), 
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We call this representation the Bloch Sphere.
        self.lecture[1].set_color(COLOR_NAME)
        self.bloch_label = Text("Bloch Sphere", font_size=24, color=COLOR_NAME)
        # Issue 36 fix: place_in_area('A2', 'A5')
        self.place_in_area(self.bloch_label, 'A2', 'A5')
        
        self.play(Write(self.bloch_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # North and South poles represent base states.
        self.lecture[2].set_color(COLOR_POLES)
        
        # Radius of the scaled sphere group for positioning
        # sphere_group was scaled by 0.9 in place_in_area. 
        # Base height was approx 3.0 (UP*1.5 to DOWN*1.5). Scaled height ~ 2.7.
        # scaled_radius ~ 1.35
        s_center = self.sphere_group.get_center()
        scaled_radius = 1.35 
        
        state_0 = MathTex(r"|0\rangle", color=COLOR_POLES, font_size=36)
        state_1 = MathTex(r"|1\rangle", color=COLOR_POLES, font_size=36)
        
        # Positioning poles relative to center
        state_0.move_to(s_center + UP * (scaled_radius + 0.3))
        state_1.move_to(s_center + DOWN * (scaled_radius + 0.3))
        
        pole_n = Dot(s_center + UP * scaled_radius, color=COLOR_POLES, radius=0.06)
        pole_s = Dot(s_center + DOWN * scaled_radius, color=COLOR_POLES, radius=0.06)
        
        self.play(
            FadeIn(pole_n), FadeIn(pole_s),
            Write(state_0), Write(state_1)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The equator represents equal superpositions of both.
        self.lecture[3].set_color(COLOR_EQUATOR)
        
        # A solid equator highlight using an Ellipse
        # width ~ 2 * scaled_radius
        equator_highlight = Ellipse(width=scaled_radius * 2, height=0.6, color=COLOR_EQUATOR, stroke_width=2)
        equator_highlight.move_to(s_center)
        
        # A point (Dot) on the equator
        theta_tracker = ValueTracker(0)
        equator_dot = Dot(color=COLOR_EQUATOR, radius=0.08)
        equator_dot.add_updater(lambda d: d.move_to(
            s_center + np.array([
                scaled_radius * np.cos(theta_tracker.get_value()),
                0.3 * np.sin(theta_tracker.get_value()),
                0
            ])
        ))
        
        self.play(
            Create(equator_highlight),
            FadeIn(equator_dot)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This sphere allows for complex quantum computations.
        self.lecture[4].set_color(COLOR_COMPUTATION)
        
        # Move the point to show rotation / state change
        self.play(
            theta_tracker.animate.set_value(TAU * 1.5),
            run_time=3,
            rate_func=linear
        )
        
        # Highlight computation complexity by color shifting key elements
        self.play(
            self.sphere_group.animate.set_color(COLOR_COMPUTATION),
            equator_highlight.animate.set_color(COLOR_COMPUTATION),
            equator_dot.animate.set_color(COLOR_COMPUTATION),
            self.bloch_label.animate.set_color(COLOR_COMPUTATION),
            state_0.animate.set_color(COLOR_COMPUTATION),
            state_1.animate.set_color(COLOR_COMPUTATION)
        )
        self.wait(2)
