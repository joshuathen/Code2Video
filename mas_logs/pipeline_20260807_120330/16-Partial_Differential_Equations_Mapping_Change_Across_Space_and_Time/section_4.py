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
        title = "The Wave Equation: Energy in Motion"
        lines = [
            "The Wave Equation governs vibrations and energy propagation.",
            "Unlike heat, spatial curvature here determines local acceleration.",
            "String tension pulls curves back, creating traveling oscillations."
        ]
        self.setup_layout(title, lines)

        # Colors
        cyan = "#00FFFF"
        vector_color = "#FFD700" # Golden yellow for acceleration
        string_color = "#AAAAAA" # Grey string

        # === Animation for Lecture Line 1 ===
        # The Wave Equation governs vibrations and energy propagation.
        self.lecture[0].set_color(cyan)
        
        wave_eq = MathTex(r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u", color=cyan)
        # Fix for Issue 40: Adjusted scale_factor to 1.0 for better spacing
        self.place_in_area(wave_eq, "B2", "B5", scale_factor=1.0)
        
        self.play(Write(wave_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Unlike heat, spatial curvature here determines local acceleration.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(vector_color)

        # Asset Integration (Issue 27)
        # Load the string SVG asset to represent the physical medium
        string_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg"
        string_asset = SVGMobject(string_path).set_color(string_color)
        # Place string asset across Row D
        self.place_in_area(string_asset, "D1", "D6", scale_factor=1.5)

        # Setup for dynamic wave animation
        x_start = self.grid["D1"][0]
        x_end = self.grid["D6"][0]
        y_eq = self.grid["D1"][1] # Base line for the string
        
        time_tracker = ValueTracker(0)
        c_speed = 1.2
        width = 0.4
        amplitude = 0.5

        def wave_func(x, t):
            # A localized sine wave pulse moving from left to right
            cycle_len = x_end - x_start + 2.0
            pos = x_start - 1.0 + (t * c_speed % cycle_len)
            envelope = np.exp(-((x - pos)**2) / (2 * width**2))
            return amplitude * np.sin(2 * PI * (x - pos) / width) * envelope

        # Dynamic string mobject
        wave_string = always_redraw(lambda: ParametricFunction(
            lambda x: np.array([x, y_eq + wave_func(x, time_tracker.get_value()), 0]),
            t_range=[x_start, x_end],
            color=string_color,
            stroke_width=3
        ))
        
        # Acceleration Label - Fix for Issue 39: Positioned in Area E4 to F6
        accel_label = MathTex(r"\text{Acceleration: } \frac{\partial^2 u}{\partial t^2}", color=vector_color)
        self.place_in_area(accel_label, 'E4', 'F6', scale_factor=0.8)

        self.play(
            FadeIn(string_asset),
            FadeIn(accel_label)
        )
        # Transition from static asset to dynamic wave string
        self.play(ReplacementTransform(string_asset, wave_string))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # String tension pulls curves back, creating traveling oscillations.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)

        # Dynamic acceleration vectors showing restoration force
        num_vectors = 8
        
        def get_vectors():
            vg = VGroup()
            t = time_tracker.get_value()
            cycle_len = x_end - x_start + 2.0
            pos = x_start - 1.0 + (t * c_speed % cycle_len)
            
            for i in range(num_vectors):
                sample_x = pos - 0.5 + i * 0.15
                if sample_x < x_start or sample_x > x_end:
                    continue
                u_val = wave_func(sample_x, t)
                if abs(u_val) < 0.05:
                    continue
                
                start_pt = np.array([sample_x, y_eq + u_val, 0])
                # Acceleration points toward equilibrium (y_eq)
                end_pt = np.array([sample_x, y_eq + u_val * 0.2, 0]) 
                arrow = Arrow(
                    start_pt, end_pt, 
                    color=vector_color, 
                    buff=0, 
                    stroke_width=2.5, 
                    max_tip_length_to_length_ratio=0.25
                )
                vg.add(arrow)
            return vg

        vector_group = always_redraw(get_vectors)
        self.add(vector_group)

        # Animate the traveling wave pulse
        self.play(time_tracker.animate.set_value(6), run_time=6, rate_func=linear)
        
        self.wait(2)
