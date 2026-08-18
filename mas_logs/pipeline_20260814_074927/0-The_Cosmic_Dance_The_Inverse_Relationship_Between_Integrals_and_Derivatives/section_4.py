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
        title_text = "The Bridge: The Fundamental Theorem of Calculus"
        lecture_lines = [
            "Define a function representing the accumulated area.",
            "Watch as we add a tiny sliver of width dx.",
            "The area's growth rate depends on the function's height.",
            "Thus, the derivative of area is the original function.",
            "This bridge connects the derivative and the integral."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        STEEL_BLUE = "#4682B4"
        GOLD = "#FFD700"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(STEEL_BLUE)
        
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 3, 1],
            axis_config={"include_tip": True},
            x_length=4.5,
            y_length=3.5
        )
        self.place_in_area(axes, "B1", "F6", scale_factor=1.0)
        
        def func(x):
            return 0.1 * x**2 + 0.5
        
        graph = axes.plot(func, x_range=[0, 4.5], color=WHITE_COLOR)
        
        area_x_val = 3.0
        area = axes.get_area(graph, x_range=[0, area_x_val], color=STEEL_BLUE, opacity=0.5)
        area_label = MathTex("A(x)", color=STEEL_BLUE, font_size=32)
        # Position label inside the area relative to the axes
        area_label.move_to(axes.c2p(area_x_val/2, 0.4))
        
        self.play(Create(axes), Create(graph))
        self.play(FadeIn(area), Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GOLD)
        
        dx_val = 0.4
        sliver = axes.get_area(graph, x_range=[area_x_val, area_x_val + dx_val], color=GOLD, opacity=0.8)
        
        dx_brace = Brace(sliver, DOWN, buff=0.1)
        # Fix: Create MathTex manually with font_size and then position it 
        # to avoid passing font_size to next_to() via get_tex()
        dx_label = MathTex("dx", font_size=24, color=GOLD)
        dx_label.next_to(dx_brace, DOWN, buff=0.1)
        
        self.play(FadeIn(sliver), Create(dx_brace), Write(dx_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(STEEL_BLUE)
        
        # Indicator line for the height of the sliver
        height_line = Line(
            axes.c2p(area_x_val + dx_val, 0),
            axes.c2p(area_x_val + dx_val, func(area_x_val + dx_val)),
            color=STEEL_BLUE,
            stroke_width=6
        )
        height_label = MathTex("f(x)", color=STEEL_BLUE, font_size=30)
        # Place label near the height line
        height_label.next_to(height_line, RIGHT, buff=0.1)
        
        self.play(Create(height_line), Write(height_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE_COLOR)
        
        da_equation = MathTex("dA = f(x) \cdot dx", color=WHITE_COLOR, font_size=36)
        self.place_at_grid(da_equation, "A3", scale_factor=1.0)
        
        self.play(Write(da_equation))
        self.play(Flash(da_equation, color=WHITE_COLOR, line_length=0.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE_COLOR)
        
        ftc_formula = MathTex(r"\frac{d}{dx} \int_{a}^{x} f(t) dt = f(x)", color=WHITE_COLOR, font_size=36)
        # Position the formula to A3 area to show transition
        self.place_at_grid(ftc_formula, "A3", scale_factor=1.1)
        
        self.play(
            FadeOut(da_equation),
            ReplacementTransform(height_label.copy(), ftc_formula)
        )
        self.wait(2)
