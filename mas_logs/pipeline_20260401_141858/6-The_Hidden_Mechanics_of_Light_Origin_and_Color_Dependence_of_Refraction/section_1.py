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

class Section1Scene(TeachingScene):
    def construct(self):
        # Alignment check: 5 Lecture Lines and 5 Animation Steps
        lecture_lines = [
            "Light is an oscillating electromagnetic field moving through space.",
            "Refractive index n is the ratio of c over v.",
            "Imagine light as a runner moving onto sandy ground.",
            "Changing speed causes the runner's path to bend.",
            "A laser beam demonstrates this refraction entering glass."
        ]
        
        self.setup_layout("Prerequisite: Light as an Electromagnetic Interaction", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        wave_tracker = ValueTracker(0)
        def get_wave():
            w = FunctionGraph(
                lambda x: 0.3 * np.sin(4 * x - 10 * wave_tracker.get_value()),
                x_range=[-1.5, 1.5],
                color="#00FFFF"
            )
            return w

        wave = always_redraw(get_wave)
        em_label = Text("Electromagnetic Field", font_size=20, color="#00FFFF")
        wave_group = VGroup(wave, em_label).arrange(DOWN, buff=0.4)
        
        # Issue 34 Fix: self.place_in_area(wave_group, 'A1', 'C6', scale_factor=0.8)
        self.place_in_area(wave_group, 'A1', 'C6', scale_factor=0.8)
        
        self.play(Create(wave), Write(em_label))
        self.play(wave_tracker.animate.increment_value(1.0), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        n_formula = Text("n = c / v", color="#FFFFFF")
        # Issue 35 Fix: self.place_at_grid(n_formula, 'B3', scale_factor=1.5)
        self.place_at_grid(n_formula, 'B3', scale_factor=1.5)
        
        c_label = Text("c: Vacuum", font_size=18, color="#AAAAAA")
        v_label = Text("v: Medium", font_size=18, color="#AAAAAA")
        self.place_at_grid(c_label, 'A3', scale_factor=1.0) # Within 1 unit (above)
        self.place_at_grid(v_label, 'C3', scale_factor=1.0) # Within 1 unit (below)
        
        self.play(Write(n_formula), Write(c_label), Write(v_label))
        self.play(wave_tracker.animate.increment_value(0.5), run_time=1, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Clear previous animations to avoid occlusion
        self.play(FadeOut(wave_group), FadeOut(n_formula), FadeOut(c_label), FadeOut(v_label))
        
        pavement_rect = Rectangle(width=5, height=2, color=GREY, fill_opacity=0.3, stroke_width=0)
        sand_rect = Rectangle(width=5, height=2, color=YELLOW, fill_opacity=0.3, stroke_width=0)
        surfaces = VGroup(pavement_rect, sand_rect).arrange(DOWN, buff=0)
        self.place_in_area(surfaces, 'D1', 'F6', scale_factor=1.0)
        
        runner = Dot(color="#FFD700").move_to(surfaces.get_top() + LEFT * 1.5)
        self.play(FadeIn(surfaces), FadeIn(runner))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        
        # Calculate bending path
        boundary_point = surfaces.get_center() + LEFT * 0.5
        end_point = surfaces.get_bottom() + RIGHT * 0.2
        
        self.play(runner.animate.move_to(boundary_point), run_time=1.2, rate_func=linear)
        self.play(runner.animate.move_to(end_point), run_time=1.8, rate_func=linear)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.play(FadeOut(surfaces), FadeOut(runner))
        
        glass_block = Rectangle(width=4, height=3, color=BLUE_A, fill_opacity=0.2)
        incident_ray = Line(start=UP*2 + LEFT*2, end=ORIGIN, color=RED)
        refracted_ray = Line(start=ORIGIN, end=DOWN*2 + RIGHT*0.8, color=RED)
        laser_animation_group = VGroup(glass_block, incident_ray, refracted_ray)
        
        # Issue 36 Fix: self.place_in_area(laser_animation_group, 'A1', 'F6', scale_factor=0.85)
        self.place_in_area(laser_animation_group, 'A1', 'F6', scale_factor=0.85)
        
        self.play(Create(glass_block))
        self.play(Create(incident_ray))
        self.play(Create(refracted_ray))
        self.wait(2)
