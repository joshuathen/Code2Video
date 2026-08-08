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
        # Setup data from storyboard
        title = "The Core Insight: How Fast Does Area Grow?"
        lines = [
            "How fast does this accumulated area actually grow?",
            "Consider a tiny sliver added at position x.",
            "The sliver height is exactly the function's height.",
            "Area added equals height times the tiny width.",
            "The growth rate is the original function's height."
        ]
        
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Color the first line
        self.lecture[0].set_color(YELLOW)
        
        # Setup the plot
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY}
        )
        
        # Define the function f(x)
        def f_func(x):
            return 0.15 * x**2 + 0.5
            
        curve = axes.plot(f_func, x_range=[0, 4.5], color="#00FF00")
        # Accumulated area up to x = 3.0
        x_at = 3.0
        area = axes.get_area(curve, x_range=[0, x_at], color="#00FF00", opacity=0.3)
        
        # Place graph elements on the grid
        # Fix for Issue 35: Move graph_visual to D1-F6 with scale 0.9
        graph_visual = VGroup(axes, curve, area)
        self.place_in_area(graph_visual, "D1", "F6", scale_factor=0.9)
        
        self.play(Create(axes), Create(curve), run_time=1.5)
        self.play(FadeIn(area), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color the second line
        self.lecture[1].set_color(YELLOW)
        
        dx = 0.25
        # Rectangle representing the 'extra sliver'
        w = axes.c2p(dx, 0)[0] - axes.c2p(0, 0)[0]
        h = axes.c2p(0, f_func(x_at))[1] - axes.c2p(0, 0)[1]
        
        sliver = Rectangle(
            width=w,
            height=h,
            fill_color="#FFFF00",
            fill_opacity=0.6,
            stroke_width=1,
            stroke_color=WHITE
        )
        # Position sliver immediately following the area at x_at
        sliver.move_to(axes.c2p(x_at + dx/2, f_func(x_at)/2))
        
        x_label = MathTex("x", font_size=24, color=WHITE).next_to(axes.c2p(x_at, 0), DOWN, buff=0.1)
        
        self.play(Create(sliver), Write(x_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color the third line
        self.lecture[2].set_color("#00FF00")
        
        # Brace for vertical dimension
        h_brace = Brace(sliver, LEFT, buff=0.05)
        h_label = MathTex("f(x)", color="#00FF00", font_size=24).next_to(h_brace, LEFT, buff=0.05)
        
        self.play(FadeIn(h_brace), Write(h_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Color the fourth line
        self.lecture[3].set_color(YELLOW)
        
        # Brace for horizontal dimension
        w_brace = Brace(sliver, DOWN, buff=0.05)
        w_label = MathTex("dx", color="#FFFFFF", font_size=24).next_to(w_brace, DOWN, buff=0.05)
        
        # Display Area Formula
        # Fix for Issue 36: Move formula_1 to B2-B5
        formula_1 = MathTex("dA = f(x) \\cdot dx", color="#FFFF00", font_size=32)
        self.place_in_area(formula_1, "B2", "B5", scale_factor=1.0)
        
        self.play(FadeIn(w_brace), Write(w_label))
        self.play(Write(formula_1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Color the fifth line
        self.lecture[4].set_color(YELLOW)
        
        # Transform to Derivative Formula
        # Fix for Issue 36: Move formula_2 to B2-B5
        # Highlight f(x) with Green color
        formula_2 = MathTex(r"\frac{dA}{dx} =", "f(x)", color="#FFFF00", font_size=32)
        formula_2.set_color_by_tex("f(x)", "#00FF00")
        self.place_in_area(formula_2, "B2", "B5", scale_factor=1.0)
        
        self.play(Transform(formula_1, formula_2))
        
        # Visual Emphasis
        h_rect = SurroundingRectangle(formula_1, color=YELLOW, buff=0.1)
        self.play(Create(h_rect))
        
        # Summary label
        # Fix for Issue 37: Move summary to C2-C5 with scale 0.8
        summary = Text("Slope of A(x) is f(x)", font_size=22, color=YELLOW)
        self.place_in_area(summary, "C2", "C5", scale_factor=0.8)
        self.play(FadeIn(summary))
        
        self.wait(2)
