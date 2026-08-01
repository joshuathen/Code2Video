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
        # Title and Lecture Lines
        title = "Application: From Eyes to Fiber Optics"
        lines = [
            "This simple law governs how lenses focus light.",
            "It also explains how fiber optics carry internet signals.",
            "Light's logic of least time shapes our modern world."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)

        # Assets and Objects
        lens_color = "#FFFFFF"
        ray_color = "#FFFF00"
        
        # Eye Asset integration (Issue 31)
        eye_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/eye.svg").set_color(WHITE)
        self.place_at_grid(eye_icon, 'B6', scale_factor=0.6)

        # Draw Lens
        # Approximate a convex lens using two arcs
        lens_left = ArcBetweenPoints(
            start=self.grid['A3'] + DOWN*0.2, 
            end=self.grid['C3'] + UP*0.2, 
            angle=-TAU/8, 
            color=lens_color
        )
        lens_right = ArcBetweenPoints(
            start=self.grid['C3'] + UP*0.2, 
            end=self.grid['A3'] + DOWN*0.2, 
            angle=-TAU/8, 
            color=lens_color
        )
        lens = VGroup(lens_left, lens_right)
        
        # Rays
        focus_point = self.grid['B6']
        
        # Ray 1: Top
        ray1_in = Line(self.grid['A1'], self.grid['A3'], color=ray_color)
        ray1_out = Line(self.grid['A3'], focus_point, color=ray_color)
        
        # Ray 2: Middle
        ray2_in = Line(self.grid['B1'], self.grid['B3'], color=ray_color)
        ray2_out = Line(self.grid['B3'], focus_point, color=ray_color)
        
        # Ray 3: Bottom
        ray3_in = Line(self.grid['C1'], self.grid['C3'], color=ray_color)
        ray3_out = Line(self.grid['C3'], focus_point, color=ray_color)
        
        rays = VGroup(ray1_in, ray2_in, ray3_in, ray1_out, ray2_out, ray3_out)

        self.play(FadeIn(eye_icon), Create(lens))
        self.play(
            Create(ray1_in), Create(ray2_in), Create(ray3_in),
            run_time=1
        )
        self.play(
            Create(ray1_out), Create(ray2_out), Create(ray3_out),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Clear previous visuals and highlight second line
        self.play(
            FadeOut(lens), FadeOut(rays), FadeOut(eye_icon),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Draw Fiber Core (parallel lines)
        core_top = Line(self.grid['D1'], self.grid['D6'], color=WHITE)
        core_bottom = Line(self.grid['E1'], self.grid['E6'], color=WHITE)
        fiber_core = VGroup(core_top, core_bottom)
        
        # Fiber Ray - Bouncing via Total Internal Reflection
        # Calculate points for a zigzag path
        p1 = self.grid['D1'] + DOWN*0.5  # Entry point (midpoint D-E)
        p2 = self.grid['D2']             # Hit top
        p3 = self.grid['E3']             # Hit bottom
        p4 = self.grid['D4']             # Hit top
        p5 = self.grid['E5']             # Hit bottom
        p6 = self.grid['D6'] + DOWN*0.5  # Exit point
        
        fiber_ray_path = VMobject(color=ray_color)
        fiber_ray_path.set_points_as_corners([p1, p2, p3, p4, p5, p6])
        
        self.play(Create(fiber_core))
        self.play(Create(fiber_ray_path), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Clear fiber and highlight third line
        self.play(
            FadeOut(fiber_core), FadeOut(fiber_ray_path),
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        closing_text = Text("Snell's Law: The Logic of Light", font_size=32, color=WHITE)
        # Applied positioning fix for Issue 47
        self.place_in_area(closing_text, 'C1', 'D6', scale_factor=1.0)
        
        self.play(Write(closing_text))
        self.wait(3)
