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
        lecture_lines = [
            "A laser beam splits into two separate paths.",
            "Reference beams provide a stable phase baseline.",
            "Object beams carry complex amplitude $E = A e^{i\\phi}$.",
            "Rotating phasors represent the wave's intensity and phase.",
            "Interference patterns encode this data into a map."
        ]
        self.setup_layout("Recording the Hologram: The Interference Map", lecture_lines)

        # Colors
        laser_color = "#00FF00"
        ref_color = "#FFFFFF"
        obj_color = "#FFFF00"
        result_color = "#00FFFF"

        # Tracker for global phase rotation
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        # Show a single laser beam #00FF00 splitting into two paths using a beam splitter.
        self.lecture[0].set_color(laser_color)
        
        splitter = Triangle(color=GREY, fill_opacity=0.7).scale(0.3).rotate(-PI/2)
        self.place_at_grid(splitter, "B2")
        
        # Beam entering splitter from off-screen left (preserving gap at Col 1)
        laser_in = Line(start=self.grid["B2"] + LEFT*1.5, end=self.grid["B2"], color=laser_color)
        
        self.play(Create(laser_in), FadeIn(splitter))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The Reference Beam #FFFFFF travels straight; show its phasor as a static rotating vector.
        self.lecture[1].set_color(ref_color)
        
        # Path: B2 down to E2, then right to E5 (Mirror at E2)
        ref_path1 = Line(self.grid["B2"], self.grid["E2"], color=ref_color)
        ref_path2 = Line(self.grid["E2"], self.grid["E5"], color=ref_color)
        
        plate = Rectangle(height=0.8, width=0.1, color=WHITE, fill_opacity=0.2)
        self.place_at_grid(plate, "E5")
        
        # Phasor at D2 (balance vertical segment - Issue 39)
        ref_phasor_base = Circle(radius=0.3, color=GREY_D)
        self.place_at_grid(ref_phasor_base, "D2", scale_factor=0.8)
        # Anchor the label/arrow at the midpoint of the phasor circle
        ref_phasor_arrow = Arrow(start=self.grid["D2"], end=self.grid["D2"] + RIGHT*0.24, buff=0, color=ref_color)
        # Persistent rotation via updater
        ref_phasor_arrow.add_updater(lambda m: m.set_angle(time_tracker.get_value() * 2 * PI))
        
        self.play(
            Create(ref_path1),
            Create(ref_path2),
            FadeIn(plate),
            FadeIn(ref_phasor_base),
            FadeIn(ref_phasor_arrow)
        )
        # Advance time to show rotation
        self.play(time_tracker.animate.set_value(1), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The Object Beam #FFFF00 hits a 3D statue; show its phasor's phase changing based on distance.
        self.lecture[2].set_color(obj_color)
        
        # Path: B2 right to B5 (Statue), then down to E5 (Plate)
        obj_path1 = Line(self.grid["B2"], self.grid["B5"], color=obj_color)
        statue = Star(n=5, color=obj_color, fill_opacity=0.8).scale(0.3)
        self.place_at_grid(statue, "B5")
        obj_path2 = Line(self.grid["B5"], self.grid["E5"], color=obj_color)
        
        # Phasor at B4 (middle of beam path - Issue 38)
        obj_phasor_base = Circle(radius=0.3, color=GREY_D)
        self.place_at_grid(obj_phasor_base, "B4", scale_factor=0.8)
        obj_phasor_arrow = Arrow(start=self.grid["B4"], end=self.grid["B4"] + RIGHT*0.24, buff=0, color=obj_color)
        # Object phasor has a different phase (PI/3)
        obj_phasor_arrow.add_updater(lambda m: m.set_angle(time_tracker.get_value() * 2 * PI + PI/3))
        
        self.play(
            Create(obj_path1),
            FadeIn(statue),
            Create(obj_path2),
            FadeIn(obj_phasor_base),
            FadeIn(obj_phasor_arrow)
        )
        self.play(time_tracker.animate.increment_value(1), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The two beams meet on a plate; show the phasors adding together: E = A * e^{i\phi} #00FFFF.
        self.lecture[3].set_color(result_color)
        
        # Phasor Addition at D4 area
        math_pos = self.grid["D4"]
        comp_ref = Arrow(start=math_pos, end=math_pos + RIGHT*0.25, buff=0, color=ref_color)
        comp_obj = Arrow(start=math_pos + RIGHT*0.25, end=math_pos + RIGHT*0.25 + UP*0.2, buff=0, color=obj_color)
        sum_arrow = Arrow(start=math_pos, end=math_pos + RIGHT*0.25 + UP*0.2, buff=0, color=result_color)
        
        # Math formula positioned at F4-F6 (Issue 37)
        formula = MathTex(r"E = A e^{i\phi}", color=result_color)
        self.place_in_area(formula, "F4", "F6", scale_factor=0.8)
        
        self.play(
            Create(comp_ref),
            Create(comp_obj),
            Write(formula)
        )
        self.play(Create(sum_arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The plate transforms into a 'wood grain' interference pattern #FFFFFF.
        self.lecture[4].set_color(WHITE)
        
        # Interference Pattern (Wood Grain Effect)
        grain_group = VGroup()
        for i in range(12):
            offset_y = (i - 6) * 0.06
            line = Arc(radius=0.1 + np.random.rand()*0.1, start_angle=0, angle=PI, color=WHITE, stroke_width=1)
            line.move_to(self.grid["E5"] + UP*offset_y)
            grain_group.add(line)
            
        self.play(
            plate.animate.set_fill(color=WHITE, opacity=0.7),
            FadeIn(grain_group),
            FadeOut(sum_arrow),
            FadeOut(comp_ref),
            FadeOut(comp_obj)
        )
        # Final rotation beat
        self.play(time_tracker.animate.increment_value(1), run_time=1, rate_func=linear)
        self.wait(2)
