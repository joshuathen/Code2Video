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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Step-by-Step Visualization: The Growing Area", 
            [
                'Let the green area represent the accumulated total.', 
                'Nudge the boundary forward by a tiny amount, dx.', 
                'This creates a thin sliver of new area.', 
                'The sliver’s height is determined by the curve’s value.', 
                'Thus, the rate of area growth is the function.'
            ]
        )

        # Helper for coordinates
        # Axes will be placed in the B2-F6 area
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, "B2", "F6", scale_factor=0.8)
        
        def func(x):
            return 0.2 * x**2 + 0.5
        
        curve = axes.plot(func, x_range=[0, 3.5], color="#58C4DD")
        
        # Initial area A(x)
        x_val = 2.0
        area = axes.get_area(curve, x_range=[0, x_val], color="#87C2A5", opacity=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87C2A5"))
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        dx = 0.4
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        line_x = axes.get_vertical_line(axes.c2p(x_val, func(x_val)), color=WHITE, line_func=DashedLine)
        line_xdx = axes.get_vertical_line(axes.c2p(x_val + dx, func(x_val + dx)), color=WHITE, line_func=DashedLine)
        
        label_x = Text("x", font_size=24, slant=ITALIC)
        self.place_at_grid(label_x, "F4", scale_factor=1.0)
        
        label_xdx = Text("x+dx", font_size=24, slant=ITALIC)
        self.place_at_grid(label_xdx, "F5", scale_factor=1.0)

        self.play(Create(line_x), Create(line_xdx), Write(label_x), Write(label_xdx))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Yellow strip (the sliver)
        sliver = axes.get_area(curve, x_range=[x_val, x_val + dx], color="#FFFF00", opacity=0.8)
        self.play(FadeIn(sliver))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Labels for the sliver
        label_f_x = Text("f(x)", font_size=24, color=WHITE, slant=ITALIC)
        # Issue 48: Position label_f_x at B5
        self.place_at_grid(label_f_x, 'B5', scale_factor=0.9)
        
        label_dx = Text("dx", font_size=24, color=WHITE, slant=ITALIC)
        self.place_at_grid(label_dx, "E5", scale_factor=1.0)

        self.play(Write(label_f_x), Write(label_dx))
        self.play(Indicate(sliver), Indicate(label_f_x))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#58C4DD"))
        
        formula1 = Text("dA = f(x) · dx", font_size=30, color=WHITE, slant=ITALIC)
        formula2 = Text("dA/dx = f(x)", font_size=30, color=WHITE, slant=ITALIC)
        
        # Issue 46: Position formula1 at A2
        self.place_at_grid(formula1, 'A2', scale_factor=0.8)
        # Issue 47: Position formula2 at A4
        self.place_at_grid(formula2, 'A4', scale_factor=0.8)
        
        # Issue 36: Asset icon next to final result
        icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/based.svg")
        self.place_at_grid(icon, "A5", scale_factor=0.4)

        self.play(Write(formula1))
        self.wait(1)
        self.play(ReplacementTransform(formula1.copy(), formula2))
        self.play(FadeIn(icon))
        self.wait(2)
