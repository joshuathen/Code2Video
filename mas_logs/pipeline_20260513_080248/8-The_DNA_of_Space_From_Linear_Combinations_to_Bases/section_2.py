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
        # Initial Setup
        lines = [
            'Combine instructions by scaling and adding them.',
            'Scale vector v and w independently.',
            'Their sum is called a linear combination.',
            'Weights determine the contribution of each vector.',
            'This arithmetic builds every path in space.'
        ]
        self.setup_layout("The Recipe: Linear Combinations", lines)
        
        origin_pt = self.grid["D3"]
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        v = Arrow(origin_pt, origin_pt + RIGHT * 1.0, buff=0, color="#00FFFF")
        w = Arrow(origin_pt, origin_pt + UP * 1.0, buff=0, color="#FFFF00")
        
        v_label = Text("v", font_size=20, color="#00FFFF").next_to(v.get_end(), DOWN, buff=0.1)
        w_label = Text("w", font_size=20, color="#FFFF00").next_to(w.get_end(), LEFT, buff=0.1)
        
        self.play(GrowArrow(v), FadeIn(v_label))
        self.play(GrowArrow(w), FadeIn(w_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#00FFFF"))
        
        # Scale v to 1.5 and w to 2.0
        v_scaled_target = Arrow(origin_pt, origin_pt + RIGHT * 1.5, buff=0, color="#00FFFF")
        w_scaled_target = Arrow(origin_pt, origin_pt + UP * 2.0, buff=0, color="#FFFF00")
        
        v_weight_label = Text("1.5v", font_size=20, color="#00FFFF").next_to(v_scaled_target.get_end(), DOWN, buff=0.1)
        w_weight_label = Text("2w", font_size=20, color="#FFFF00").next_to(w_scaled_target.get_end(), LEFT, buff=0.1)
        
        self.play(
            Transform(v, v_scaled_target),
            Transform(w, w_scaled_target),
            Transform(v_label, v_weight_label),
            Transform(w_label, w_weight_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color(WHITE))
        
        # Move w to tip of v
        w_shifted_target = Arrow(v.get_end(), v.get_end() + UP * 2.0, buff=0, color="#FFFF00")
        self.play(
            w.animate.move_to(v.get_end() + UP * 1.0), # Vector center is at mid-point
            w_label.animate.next_to(v.get_end() + UP * 2.0, RIGHT, buff=0.1)
        )
        
        # Draw resultant z
        z = Arrow(origin_pt, v.get_end() + UP * 2.0, buff=0, color="#FFFFFF")
        z_label = Text("z", font_size=24, color="#FFFFFF").next_to(z.get_center(), UL, buff=0.05)
        
        self.play(GrowArrow(z), FadeIn(z_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(GRAY), self.lecture[3].animate.set_color("#FFA500"))
        
        # Formula: z = c1v + c2w
        z_f = Text("z", color="#FFFFFF")
        eq_f = Text(" = ", color="#FFFFFF")
        c1_f = Text("c1", color="#FFA500")
        v_f = Text("v", color="#00FFFF")
        plus_f = Text(" + ", color="#FFFFFF")
        c2_f = Text("c2", color="#FFA500")
        w_f = Text("w", color="#FFFF00")
        formula = VGroup(z_f, eq_f, c1_f, v_f, plus_f, c2_f, w_f).arrange(RIGHT, buff=0.1)
        self.place_at_grid(formula, "A4", scale_factor=0.8)
        
        self.play(FadeIn(formula))
        self.play(Indicate(c1_f, color="#FFA500", scale_factor=1.3), Indicate(c2_f, color="#FFA500", scale_factor=1.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(GRAY), self.lecture[4].animate.set_color(WHITE))
        
        # Trail and variety
        dot = Dot(z.get_end(), radius=0.05, color="#FFFFFF")
        trail = TracedPath(dot.get_center, stroke_color="#FFFFFF", stroke_width=2)
        self.add(trail, dot)
        
        # Points to move to: (c1, c2)
        # Point 1: (0.5, 1.0) -> pos: origin + 0.5*RIGHT*1.0 + 1.0*UP*1.0
        # Point 2: (-0.5, 0.5) -> pos: origin - 0.5*RIGHT*1.0 + 0.5*UP*1.0
        # Point 3: (1.2, -0.5) -> pos: origin + 1.2*RIGHT*1.0 - 0.5*UP*1.0
        
        pts = [
            origin_pt + RIGHT * 0.5 + UP * 1.0,
            origin_pt - RIGHT * 0.5 + UP * 0.5,
            origin_pt + RIGHT * 1.2 - UP * 0.5,
            origin_pt + RIGHT * 1.5 + UP * 2.0  # Back to original z
        ]
        
        for pt in pts:
            new_z = Arrow(origin_pt, pt, buff=0, color="#FFFFFF")
            self.play(
                Transform(z, new_z),
                dot.animate.move_to(pt),
                z_label.animate.next_to(pt, UR, buff=0.1),
                run_time=1.5
            )
            
        self.wait(2)
