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
        self.setup_layout(
            "Defining the Sphere: From 1D to 3D",
            [
                "A sphere is all points at distance R from center.",
                "In 1D, a sphere is just two points on line.",
                "In 2D, a sphere is the boundary of circle.",
                "In 3D, it is the surface of solid ball.",
                "The boundary dimension is always one lower than space."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        center_dot = Dot(color=WHITE)
        radius = 1.5
        radius_line = Line(ORIGIN, [radius, 0, 0], color="#00FFFF")
        
        anim1_group = VGroup(center_dot, radius_line)
        self.place_in_area(anim1_group, "B2", "E5")
        
        rot_tracker = ValueTracker(0)
        radius_line.add_updater(lambda m: m.set_angle(rot_tracker.get_value()))
        
        trace = TracedPath(radius_line.get_end, stroke_color="#00FFFF", stroke_width=2)
        
        self.add(center_dot, radius_line, trace)
        self.play(rot_tracker.animate.set_value(2 * PI), run_time=3)
        radius_line.clear_updaters()
        
        self.play(FadeOut(anim1_group), FadeOut(trace))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        axis = Line(LEFT * 2, RIGHT * 2, color=WHITE)
        dot_minus_r = Dot(color="#00FFFF").move_to(LEFT * 1.5)
        dot_plus_r = Dot(color="#00FFFF").move_to(RIGHT * 1.5)
        
        dim1_group = VGroup(axis, dot_minus_r, dot_plus_r)
        # Fix: Positioning shifted to right (Issue 22)
        self.place_in_area(dim1_group, 'B3', 'B6', scale_factor=0.8)
        
        self.play(Create(axis))
        self.play(FadeIn(dot_minus_r, scale=0.5), FadeIn(dot_plus_r, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        circle_2d = Circle(radius=1.5, color="#00FF00")
        # Fix: Positioning shifted to right (Issue 23)
        self.place_in_area(circle_2d, 'C3', 'E6', scale_factor=0.8)
        
        self.play(
            ReplacementTransform(axis, circle_2d),
            FadeOut(dot_minus_r),
            FadeOut(dot_plus_r)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        
        # Integration of Asset (Issue 19)
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        sphere_asset.set_color("#FFD700")
        
        # Fix: Positioning shifted to right (Issue 23)
        self.place_in_area(sphere_asset, 'C3', 'E6', scale_factor=0.8)
        
        self.play(ReplacementTransform(circle_2d, sphere_asset))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        notation = MathTex(r"S^{n-1} \subset \mathbb{R}^n", color=WHITE)
        # Fix: Scaling and positioning adjusted (Issue 24)
        self.place_in_area(notation, 'F3', 'F6', scale_factor=1.0)
        
        self.play(Write(notation))
        self.wait(3)
