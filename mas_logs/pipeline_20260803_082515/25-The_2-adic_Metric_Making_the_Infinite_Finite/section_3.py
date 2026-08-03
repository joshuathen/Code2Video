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
        # Data from shared state
        title = "The 2-adic Metric: A New Geometry"
        lecture_lines = [
            "2-adic distance depends on the highest power of 2 dividing.",
            "Distance d(x, y) equals 1 over 2 raised to n.",
            "Now, 1024 is much closer to 0 than 1.",
            "Numbers rearrange into a surprising, fractal-like geometry.",
            "Large powers of 2 practically touch the origin."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors from Animation Description
        color_0 = "#FFFFFF"
        color_1 = "#FFFF00"
        color_2 = "#00FF00"
        color_1024 = "#00FFFF"
        color_fractal = "#555555"

        # === Animation for Lecture Line 1 ===
        # 2-adic distance depends on the highest power of 2 dividing.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        formula_main = MathTex(r"d(x, 0) = \frac{1}{2^n}", color=WHITE)
        # Resolved Issue 25: use place_in_area to avoid overflow
        self.place_in_area(formula_main, 'B2', 'B3', scale_factor=1.0)
        
        v2_def = MathTex(r"n = v_2(x)", color=WHITE)
        # Resolved Issue 23: move to B4 for continuity
        self.place_at_grid(v2_def, "B4", scale_factor=1.0)
        
        self.play(Write(formula_main))
        self.play(Write(v2_def))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Distance d(x, y) equals 1 over 2 raised to n.
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        highlight_box = SurroundingRectangle(formula_main, color=WHITE, buff=0.1)
        self.play(Create(highlight_box))
        self.play(FadeOut(highlight_box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Now, 1024 is much closer to 0 than 1.
        self.play(self.lecture[2].animate.set_color(color_1024))
        
        # Conceptual number line
        num_line = Line(self.grid["E1"], self.grid["E6"], color=GREY_E)
        
        # 0 Point
        dot_0 = Dot(color=color_0)
        self.place_at_grid(dot_0, "E1")
        label_0 = Text("0", font_size=20, color=color_0).next_to(dot_0, DOWN, buff=0.1)
        
        # 1 Point
        dot_1 = Dot(color=color_1)
        self.place_at_grid(dot_1, "E6")
        label_1 = Text("1", font_size=20, color=color_1).next_to(dot_1, DOWN, buff=0.1)
        
        # 2 Point (Halfway conceptually)
        dot_2 = Dot(color=color_2)
        self.place_at_grid(dot_2, "E4")
        label_2 = Text("2", font_size=20, color=color_2).next_to(dot_2, DOWN, buff=0.1)
        
        # 1024 Point (Extremely close to 0)
        dot_1024 = Dot(color=color_1024)
        # Resolved Issue 24: move to D1 to avoid overlap with dot_0 at E1
        self.place_at_grid(dot_1024, "D1", scale_factor=0.8) 
        label_1024 = Text("1024", font_size=20, color=color_1024).next_to(dot_1024, UP, buff=0.2)
        
        self.play(Create(num_line))
        self.play(FadeIn(dot_0, label_0), FadeIn(dot_1, label_1))
        self.play(FadeIn(dot_2, label_2))
        self.play(FadeIn(dot_1024, label_1024))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Numbers rearrange into a surprising, fractal-like geometry.
        self.play(self.lecture[3].animate.set_color(color_fractal))
        
        # Binary Tree Fractal
        def build_tree(start_pos, depth, length, angle, angle_step):
            if depth == 0:
                return VGroup()
            end_pos = start_pos + np.array([np.cos(angle) * length, np.sin(angle) * length, 0])
            line = Line(start_pos, end_pos, color=color_fractal, stroke_width=2)
            left = build_tree(end_pos, depth - 1, length * 0.7, angle + angle_step, angle_step)
            right = build_tree(end_pos, depth - 1, length * 0.7, angle - angle_step, angle_step)
            return VGroup(line, left, right)

        fractal_root = self.grid["C4"]
        fractal_tree = build_tree(fractal_root, 4, 1.2, -PI/2, PI/4)
        
        self.play(
            FadeOut(num_line, dot_0, label_0, dot_1, label_1, dot_2, label_2, dot_1024, label_1024, formula_main, v2_def),
            Create(fractal_tree)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Large powers of 2 practically touch the origin.
        self.play(self.lecture[4].animate.set_color(color_2))
        
        # Origin Marker
        origin_highlight = Star(n=5, color=color_2, fill_opacity=0.5).scale(0.2)
        origin_highlight.move_to(fractal_root)
        origin_label = Text("0 (Origin)", font_size=18, color=color_2).next_to(origin_highlight, UP)
        
        # Sequential dots moving towards root
        dots_conv = VGroup()
        for i in range(1, 6):
            p_dot = Dot(radius=0.04, color=color_2)
            # Conceptually placing them closer and closer to origin
            p_dot.move_to(fractal_root + np.array([0, -0.5 * (0.8**i), 0]))
            dots_conv.add(p_dot)
            
        self.play(FadeIn(origin_highlight, origin_label))
        self.play(LaggedStart(*[FadeIn(d) for d in dots_conv], lag_ratio=0.3))
        self.wait(2)
