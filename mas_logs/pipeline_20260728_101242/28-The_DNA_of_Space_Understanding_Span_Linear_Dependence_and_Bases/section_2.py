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
        # Setup layout
        self.setup_layout("The Reach: Defining Span", [
            "- Span is the set of all reachable points.",
            "- Imagine paint spreading everywhere a vector reaches.",
            "- A single vector spans a simple straight line.",
            "- Two non-parallel vectors span a flat 2D floor.",
            "- Linear combinations fill the space they span entirely."
        ])

        # Colors
        color_v = "#FFD700"
        color_w = "#00FFFF"
        color_paint = "#FFFFFF"
        color_dots = "#ADFF2F"
        
        # Positions
        origin_pos = self.grid["D3"]
        v_end = self.grid["B4"]
        w_end = self.grid["E5"]

        # === Animation for Lecture Line 1 ===
        # "Span is the set of all reachable points."
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Imagine paint spreading everywhere a vector reaches."
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A single vector spans a simple straight line."
        self.play(self.lecture[2].animate.set_color(color_v))
        
        vector_v = Arrow(origin_pos, v_end, buff=0, color=color_v)
        v_label = MathTex("\\vec{v}", color=color_v)
        self.place_at_grid(v_label, "B5", scale_factor=0.8)
        
        # Infinite line through v
        dir_v = v_end - origin_pos
        line_v = Line(origin_pos - 4 * dir_v, origin_pos + 4 * dir_v, color=color_v, stroke_width=2)
        
        self.play(Create(vector_v), Write(v_label))
        self.play(Create(line_v))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Two non-parallel vectors span a flat 2D floor."
        self.play(self.lecture[3].animate.set_color(color_w))
        
        vector_w = Arrow(origin_pos, w_end, buff=0, color=color_w)
        w_label = MathTex("\\vec{w}", color=color_w)
        self.place_at_grid(w_label, "E6", scale_factor=0.8)
        
        # Paint spreading effect
        # Issue 21: Reduce paint area to B2-E5 to avoid cramped layout
        paint_area = Rectangle(width=3.0, height=3.0, fill_color=color_paint, fill_opacity=0.2, stroke_width=0)
        self.place_in_area(paint_area, 'B2', 'E5')
        
        # Span label
        # Issue 22: F5 with scale_factor 0.8
        span_label = MathTex("\\text{Span}(\\vec{v}, \\vec{w})", color=WHITE)
        self.place_at_grid(span_label, 'F5', scale_factor=0.8)

        self.play(Create(vector_w), Write(w_label))
        # Animate "paint spreading" by growing from the origin
        self.play(
            GrowFromPoint(paint_area, origin_pos),
            Write(span_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Linear combinations fill the space they span entirely."
        self.play(self.lecture[4].animate.set_color(color_dots))
        
        # Generate dots as linear combinations of v and w
        np.random.seed(42)
        dots = VGroup()
        vec_v_val = v_end - origin_pos
        vec_w_val = w_end - origin_pos
        for _ in range(40):
            # Coefficients
            a = np.random.uniform(-1.5, 1.5)
            b = np.random.uniform(-1.5, 1.5)
            dot_pos = origin_pos + a * vec_v_val + b * vec_w_val
            # Ensure within right side bounds (approx x: 0.5 to 5.5, y: -2.8 to 2.2)
            if 0.5 < dot_pos[0] < 5.5 and -2.8 < dot_pos[1] < 2.2:
                dot = Dot(dot_pos, radius=0.04, color=color_dots)
                dots.add(dot)
            
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.05))
        self.wait(2)
