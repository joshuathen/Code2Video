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
        title_text = "The Visual Concept: Density vs. Probability"
        lecture_lines = [
            "Meet the Probability Density Function, or the PDF curve.",
            "The height of the curve represents density, not probability.",
            "We select a specific range on the horizontal axis.",
            "Probability is the area under the curve's shape.",
            "Shading this region finds the likelihood of an outcome."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_CURVE = "#f1c40f"  # Yellow
        COLOR_HEIGHT = "#5dade2" # Cyan
        COLOR_AREA = "#2ecc71"   # Green

        # Axes setup
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 1, 0.5],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        self.place_in_area(axes, "A1", "F6", scale_factor=0.8)

        # PDF Curve function (Bell Curve)
        def pdf_func(x):
            return np.exp(-(x - 2.5)**2 / (2 * 0.8**2)) / (0.8 * np.sqrt(2 * np.pi))

        curve = axes.plot(pdf_func, x_range=[0.5, 4.5], color=COLOR_CURVE)
        curve_label = MathTex("f(x)", color=COLOR_CURVE, font_size=24)
        self.place_at_grid(curve_label, "B5", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_CURVE)
        self.play(Write(axes))
        self.play(Create(curve), FadeIn(curve_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HEIGHT)
        
        # Point to height at x=1.5
        x_point = 1.5
        y_point = pdf_func(x_point)
        dot = Dot(axes.c2p(x_point, y_point), color=COLOR_HEIGHT)
        v_line = axes.get_vertical_line(axes.c2p(x_point, y_point), color=COLOR_HEIGHT, line_func=Line)
        
        # Issue 30 Fix: height_label positioning
        height_label = Text("Density", color=COLOR_HEIGHT, font_size=24)
        self.place_in_area(height_label, 'B2', 'C3', scale_factor=0.7)
        
        self.play(Create(v_line), FadeIn(dot), Write(height_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_AREA)
        
        # Mark interval [2, 3] on x-axis
        interval_line = Line(axes.c2p(2, 0), axes.c2p(3, 0), color=COLOR_AREA, stroke_width=6)
        label_2 = MathTex("2", font_size=20).next_to(axes.c2p(2, 0), DOWN, buff=0.1)
        label_3 = MathTex("3", font_size=20).next_to(axes.c2p(3, 0), DOWN, buff=0.1)
        
        self.play(FadeOut(v_line), FadeOut(dot), FadeOut(height_label))
        self.play(Create(interval_line), FadeIn(label_2), FadeIn(label_3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_AREA)
        
        # Shade area between 2 and 3
        area = axes.get_area(curve, x_range=[2, 3], color=COLOR_AREA, opacity=0.5)

        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_AREA)
        
        # Issue 31 Fix: area_label positioning
        area_label = Text("Probability (Area)", color=COLOR_AREA, font_size=24)
        self.place_in_area(area_label, 'C4', 'D5', scale_factor=0.8)

        self.play(Write(area_label))
        self.wait(2)

        # Reset colors for end of scene
        self.lecture[4].set_color(WHITE)
        self.wait(1)
