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
        self.setup_layout("The Harmonic Blueprint (The Formula)", [
            "The Fourier Series combines sines of different frequencies.",
            "n=1 is the fundamental, the base pitch we hear.",
            "Higher values of n represent overtones or harmonics.",
            "Each harmonic adds detail and texture to the sound.",
            "Together, they form the complete blueprint for any signal."
        ])

        # Colors
        COLOR_L1 = "#FFFF00"
        COLOR_L2 = "#00FFFF"
        COLOR_L3 = "#FFA500"
        COLOR_L4 = "#FFA500"
        COLOR_L5 = "#FFFFFF"
        COLOR_FUNDAMENTAL = "#D3D3D3"
        COLOR_OVERTONE = "#FFA500"

        # Time Tracker for animations
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda t, dt: t.increment_value(dt))

        # Helper function for vibrating strings
        def get_string(n, color, length=4, amplitude=0.4):
            res = 50
            points = [np.array([x, 0, 0]) for x in np.linspace(-length/2, length/2, res)]
            mobj = VMobject(color=color).set_points_as_corners(points)
            mobj.n_val = n
            mobj.amp = amplitude
            mobj.len = length
            mobj.base_center = np.array([0,0,0])
            
            def string_updater(m):
                t = time_tracker.get_value()
                new_points = []
                for x_rel in np.linspace(-m.len/2, m.len/2, res):
                    y = m.amp * np.sin(m.n_val * PI * (x_rel + m.len/2) / m.len) * np.cos(m.n_val * t * 4)
                    new_points.append(m.base_center + np.array([x_rel, y, 0]))
                m.set_points_as_corners(new_points)
            
            mobj.add_updater(string_updater)
            return mobj

        # === Animation for Lecture Line 1 ===
        # Display the Fourier series formula \sum b_n \sin(n \omega x) in #FFFF00
        formula = MathTex(r"\sum b_n \sin(n \omega x)", color=COLOR_L1)
        self.place_in_area(formula, "B1", "B6", scale_factor=1.2)
        
        self.play(self.lecture[0].animate.set_color(COLOR_L1))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight n and label it "Harmonic Number" in #00FFFF.
        # Show a string vibrating at n=1 (Fundamental) in #D3D3D3.
        
        n_label = Text("Harmonic Number", font_size=20, color=COLOR_L2)
        self.place_at_grid(n_label, "A4", scale_factor=0.8)
        
        string_n1 = get_string(1, COLOR_FUNDAMENTAL, length=4.0)
        self.place_in_area(string_n1, "C2", "C6") # Fixed Issue 32
        string_n1.base_center = string_n1.get_center()
        
        label_n1 = Text("n=1 (Fundamental)", font_size=18, color=COLOR_FUNDAMENTAL)
        self.place_at_grid(label_n1, "C1", scale_factor=0.5) # Fixed Issue 33

        self.play(self.lecture[1].animate.set_color(COLOR_L2))
        self.play(
            formula.animate.set_color_by_tex("n", COLOR_L2),
            Write(n_label),
            Create(string_n1),
            Write(label_n1)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Add vibrations for n=3 and n=5 (Overtones) in #FFA500.
        
        string_n3 = get_string(3, COLOR_OVERTONE, length=4.0, amplitude=0.3)
        self.place_in_area(string_n3, "D2", "D6") # Fixed Issue 32
        string_n3.base_center = string_n3.get_center()
        
        label_n3 = Text("n=3", font_size=18, color=COLOR_OVERTONE)
        self.place_at_grid(label_n3, "D1", scale_factor=0.8)

        string_n5 = get_string(5, COLOR_OVERTONE, length=4.0, amplitude=0.2)
        self.place_in_area(string_n5, "E2", "E6") # Fixed Issue 32
        string_n5.base_center = string_n5.get_center()
        
        label_n5 = Text("n=5", font_size=18, color=COLOR_OVERTONE)
        self.place_at_grid(label_n5, "E1", scale_factor=0.8)

        self.play(self.lecture[2].animate.set_color(COLOR_L3))
        self.play(
            Create(string_n3),
            Write(label_n3),
            Create(string_n5),
            Write(label_n5)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Flash the higher n values to emphasize their role as "detail".
        self.play(self.lecture[3].animate.set_color(COLOR_L4))
        self.play(
            Flash(string_n3, color=COLOR_OVERTONE, flash_radius=1.5, line_stroke_width=3),
            Flash(string_n5, color=COLOR_OVERTONE, flash_radius=1.5, line_stroke_width=3),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Together, they form the complete blueprint for any signal.
        # Showing the sum of harmonics as a "Blueprint" string.
        
        blueprint_string = get_string(1, COLOR_L5, length=4.0)
        def blueprint_updater(m):
            t = time_tracker.get_value()
            res = 50
            new_points = []
            for x_rel in np.linspace(-m.len/2, m.len/2, res):
                # Harmonic superposition: b1*sin(1x) + b3*sin(3x) + b5*sin(5x)
                y1 = 0.5 * np.sin(1 * PI * (x_rel + m.len/2) / m.len) * np.cos(1 * t * 4)
                y3 = 0.15 * np.sin(3 * PI * (x_rel + m.len/2) / m.len) * np.cos(3 * t * 4)
                y5 = 0.08 * np.sin(5 * PI * (x_rel + m.len/2) / m.len) * np.cos(5 * t * 4)
                y = y1 + y3 + y5
                new_points.append(m.base_center + np.array([x_rel, y, 0]))
            m.set_points_as_corners(new_points)

        # Update the default updater with the composite one
        blueprint_string.remove_updater(blueprint_string.updaters[0])
        blueprint_string.add_updater(blueprint_updater)
        
        self.place_in_area(blueprint_string, "F2", "F6") # Fixed Issue 32
        blueprint_string.base_center = blueprint_string.get_center()
        
        label_blueprint = Text("Complete Blueprint (Sum)", font_size=18, color=COLOR_L5)
        self.place_at_grid(label_blueprint, "F1", scale_factor=0.5) # Fixed Issue 34

        self.play(self.lecture[4].animate.set_color(COLOR_L5))
        self.play(
            Create(blueprint_string),
            Write(label_blueprint)
        )
        self.wait(3)
