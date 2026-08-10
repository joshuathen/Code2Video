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
            "Define A(x) as the accumulated area bucket.",
            "As x grows, the area bucket increases.",
            "The rate of change is the function's height.",
            "Area accumulation links directly to function values.",
            "The Fundamental Theorem connects these two ideas."
        ]
        self.setup_layout("The Fundamental Theorem of Calculus", lecture_lines)

        # Create Graph
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        axes.set_color(WHITE)
        func = lambda x: 0.1 * x**3 - 0.5 * x**2 + 0.8 * x + 1
        graph = axes.plot(func, x_range=[0, 3.5], color=WHITE)
        
        # Setup visuals in grid
        self.place_in_area(axes, "A1", "D4", scale_factor=0.5)
        graph.set_color(WHITE)
        
        # Area under curve
        x_tracker = ValueTracker(0.5)
        
        def get_area():
            x_val = x_tracker.get_value()
            return axes.get_area(graph, x_range=[0, x_val], color=BLUE, opacity=0.3)
        
        area = always_redraw(get_area)
        
        label_ax = MathTex("A(x)").set_color(BLUE)
        self.place_at_grid(label_ax, "C5", scale_factor=0.9)
        
        # Asset: Bucket
        bucket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bucket.svg")
        self.place_at_grid(bucket, "B5", scale_factor=0.3)
        
        self.add(axes, graph, area, bucket)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(x_tracker.animate.set_value(3), bucket.animate.move_to(self.grid["B3"]), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        slice_rect = always_redraw(lambda: axes.get_area(graph, x_range=[x_tracker.get_value(), x_tracker.get_value() + 0.2], color="#FFD700", opacity=0.8))
        self.add(slice_rect)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(RED)
        final_eq = MathTex("A'(x) = f(x)").set_color(RED)
        self.place_at_grid(final_eq, "D5", scale_factor=1.0)
        self.play(Write(final_eq))
        self.wait(2)
