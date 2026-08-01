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
        # Configuration and Script Data
        title_text = "Defining the PDF: Height vs. Area"
        lecture_lines = [
            "The function f(x) represents probability density.",
            "Crucially, the height f(x) is not the probability.",
            "Instead, the area under the curve represents probability.",
            "First, the curve never falls below the axis.",
            "Second, the total area must always equal one."
        ]
        
        # Initialize Layout
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_CURVE = "#00FFFF"
        COLOR_SHADE = "#FF00FF"
        COLOR_TOTAL = "#00FF00"
        COLOR_TEXT = "#FFFFFF"
        COLOR_HIGHLIGHT = YELLOW

        # Define PDF function (Normal-like)
        def pdf_func(x):
            return np.exp(-x**2)

        # Create Axes and Curve
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.4],
            x_length=4.0,
            y_length=3.0,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Shifted slightly to ensure Col 2 is mostly a buffer
        self.place_in_area(axes, "B2", "E6")
        
        curve = axes.plot(pdf_func, color=COLOR_CURVE)
        curve_label = MathTex("f(x)", color=COLOR_CURVE)
        # Position label in B6 to avoid overlap with height_info (B4-B5)
        self.place_at_grid(curve_label, "B6", scale_factor=0.8)
        
        density_label = Text("Density", font_size=20, color=COLOR_TEXT)
        self.place_at_grid(density_label, "B2", scale_factor=0.8)
        density_label.shift(LEFT * 0.4 + UP * 0.2)

        # === Animation for Lecture Line 1 ===
        # L1: The function f(x) represents probability density.
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        self.play(
            Write(axes),
            Create(curve),
            FadeIn(curve_label),
            Write(density_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L2: Crucially, the height f(x) is not the probability.
        self.play(self.lecture[1].animate.set_color(COLOR_HIGHLIGHT))
        
        # Pulse thicker parts (Animation Stage 5)
        peak_part = axes.plot(pdf_func, x_range=[-0.5, 0.5], color=COLOR_HIGHLIGHT, stroke_width=6)
        
        # Visualize height at a point
        x_pt = -0.7
        height_pt = axes.c2p(x_pt, pdf_func(x_pt))
        dot = Dot(height_pt, color=COLOR_HIGHLIGHT)
        v_line = axes.get_vertical_line(height_pt, color=COLOR_HIGHLIGHT)
        
        height_info = Text("Height ≠ Prob", font_size=22, color=COLOR_HIGHLIGHT)
        # Fix 21: Position height_info to avoid overlap with curve label
        self.place_in_area(height_info, 'B4', 'B5', scale_factor=0.7)
        
        self.play(FadeIn(dot), Create(v_line))
        self.play(Write(height_info))
        self.play(Indicate(peak_part, color=COLOR_HIGHLIGHT, scale_factor=1.2))
        self.wait(1)
        self.play(FadeOut(dot), FadeOut(v_line), FadeOut(height_info), FadeOut(peak_part))

        # === Animation for Lecture Line 3 ===
        # L3: Instead, the area under the curve represents probability.
        self.play(self.lecture[2].animate.set_color(COLOR_SHADE))
        
        region_a, region_b = 0.5, 1.8
        area_ab = axes.get_area(curve, x_range=[region_a, region_b], color=COLOR_SHADE, fill_opacity=0.5)
        label_a = MathTex("a", font_size=24, color=WHITE).move_to(axes.c2p(region_a, -0.3))
        label_b = MathTex("b", font_size=24, color=WHITE).move_to(axes.c2p(region_b, -0.3))
        
        prob_label = Text("Area = Probability", font_size=20, color=COLOR_SHADE)
        # Fix 22: Position prob_label to avoid overlap with shaded area
        self.place_at_grid(prob_label, 'D6', scale_factor=0.8)
        
        self.play(FadeIn(area_ab), Write(label_a), Write(label_b), Write(prob_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # L4: First, the curve never falls below the axis.
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        pos_math = MathTex("f(x) \\ge 0", color=WHITE)
        self.place_at_grid(pos_math, "E3", scale_factor=0.8)
        
        # Highlight x-axis (Animation Stage 3)
        x_axis_highlight = Line(axes.c2p(-3, 0), axes.c2p(3, 0), color=COLOR_HIGHLIGHT, stroke_width=5)
        
        self.play(Create(x_axis_highlight))
        self.play(Write(pos_math))
        self.wait(1)
        self.play(FadeOut(x_axis_highlight))

        # === Animation for Lecture Line 5 ===
        # L5: Second, the total area must always equal one.
        self.play(self.lecture[4].animate.set_color(COLOR_TOTAL))
        
        total_area = axes.get_area(curve, x_range=[-3, 3], color=COLOR_TOTAL, fill_opacity=0.3)
        total_info = Text("Total Area = 1", font_size=22, color=COLOR_TOTAL)
        # Fix 23: Position total_info to avoid obstructing the peak
        self.place_in_area(total_info, 'A3', 'A4', scale_factor=0.7)
        
        self.play(
            FadeOut(area_ab), 
            FadeOut(prob_label), 
            FadeOut(label_a), 
            FadeOut(label_b),
            FadeOut(pos_math)
        )
        self.play(FadeIn(total_area))
        self.play(Write(total_info))
        self.wait(2)
