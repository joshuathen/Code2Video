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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "The Wave Equation: Modeling Vibration"
        lines = [
            "The wave equation models oscillations like a plucked string.",
            "Acceleration at a point depends on local spatial tension.",
            "This creates disturbances that propagate as traveling waves.",
            "Energy moves through the medium without displacing the matter.",
            "Hyperbolic equations describe this dynamic, time-dependent behavior."
        ]
        self.setup_layout(title, lines)

        # Helper for wave pulse
        def pulse_func(x, t, center=3.0, width=0.5, amp=0.5):
            # Traveling pulse f(x - ct)
            c = 1.5
            pos = center + c * t
            # Simple wrapping for continuous movement in demo
            pos = (pos % 7.0) - 0.5 
            return amp * np.exp(-((x - pos) ** 2) / (2 * width ** 2))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg]
        # Load string asset and adjust it to fit the horizontal span of the grid
        string_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg", color=WHITE)
        string_asset.width = self.grid["D6"][0] - self.grid["D1"][0]
        self.place_in_area(string_asset, "D1", "D6", scale_factor=1.0)
        
        # Formula: d^2u/dt^2 = c^2 d^2u/dx^2
        formula = MathTex(r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}", color=WHITE)
        # Fix Issue 32: Vertical centering and area placement
        self.place_in_area(formula, 'A1', 'B6', scale_factor=1.0)
        
        self.play(DrawBorderThenFill(string_asset), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Brown arc pulling the string
        arc = ArcBetweenPoints(self.grid["C3"] + LEFT*0.2, self.grid["C3"] + RIGHT*0.2, angle=TAU/4, color="#A52A2A")
        
        # Define pulled string shape
        pulled_path = VMobject(color=WHITE)
        pulled_path.set_points_as_corners([
            self.grid["D1"],
            self.grid["D2"],
            self.grid["C3"], # Pulled up point
            self.grid["D4"],
            self.grid["D5"],
            self.grid["D6"]
        ]).make_smooth()

        self.play(Create(arc))
        self.play(Transform(string_asset, pulled_path))
        self.wait(1)
        self.play(FadeOut(arc))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Traveling pulse simulation
        time_tracker = ValueTracker(0)
        
        def get_wave_path_points():
            points = []
            start_x = self.grid["D1"][0]
            end_x = self.grid["D6"][0]
            y_base = self.grid["D1"][1]
            for x in np.linspace(start_x, end_x, 60):
                rel_x = (x - start_x) / (end_x - start_x) * 6.0 
                y_offset = pulse_func(rel_x, time_tracker.get_value())
                points.append([x, y_base + y_offset, 0])
            return points

        wave_line = VMobject(color="#00FFFF")
        wave_line.set_points_as_corners(get_wave_path_points())
        wave_line.add_updater(lambda m: m.set_points_as_corners(get_wave_path_points()))
        
        # Yellow dot on the string at x near Grid D4 (x=3.5)
        dot_x_coord = self.grid["D4"][0]
        dot_rel_x = (dot_x_coord - self.grid["D1"][0]) / (self.grid["D6"][0] - self.grid["D1"][0]) * 6.0
        
        dot = Dot(color="#FFFF00")
        dot.add_updater(lambda m: m.move_to([
            dot_x_coord, 
            self.grid["D1"][1] + pulse_func(dot_rel_x, time_tracker.get_value()), 
            0
        ]))

        # Fix Issue 33 & 34: Grouping and constraining to right-side area C1:F6
        wave_animation_group = VGroup(wave_line, dot)
        # Note: Since coordinates are derived from the grid, they are already constrained.
        
        self.play(FadeOut(string_asset), Create(wave_line), FadeIn(dot))
        self.play(time_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Dashed line to show dot only moves vertically
        v_line = DashedLine(self.grid["C4"], self.grid["E4"], color=GRAY, stroke_opacity=0.3)
        self.add(v_line)
        self.play(time_tracker.animate.set_value(8), run_time=4, rate_func=linear)
        self.remove(v_line)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Green vectors showing curvature at the wave peak
        def get_peak_pos():
            t = time_tracker.get_value()
            c = 1.5
            center = 3.0
            pos_rel = ((center + c * t) % 7.0) - 0.5
            x_coord = self.grid["D1"][0] + (pos_rel / 6.0) * (self.grid["D6"][0] - self.grid["D1"][0])
            # Clamp to horizontal grid bounds
            x_coord = np.clip(x_coord, self.grid["D1"][0], self.grid["D6"][0])
            y_coord = self.grid["D1"][1] + pulse_func(pos_rel, t)
            return [x_coord, y_coord, 0]

        peak_vector = Arrow(start=DOWN*0.3, end=UP*0.3, color="#00FF00", buff=0)
        peak_vector.add_updater(lambda m: m.move_to(get_peak_pos()))
        
        # Ensure the vector is also part of the constrained group
        wave_animation_group.add(peak_vector)
        
        self.play(GrowArrow(peak_vector))
        self.play(time_tracker.animate.set_value(12), run_time=4, rate_func=linear)
        
        self.wait(2)
