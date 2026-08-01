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
        title = "Step-by-Step Mathematical Application"
        # Removing LaTeX symbols from lecture lines as they are rendered with Text()
        lecture_lines = [
            "Find the area under 3t^2 from 0.",
            "Find a function whose derivative is 3t^2.",
            "We identify t^3 as the anti-derivative.",
            "Subtract the start value from the end.",
            "Distance and area are now one."
        ]
        self.setup_layout(title, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # The equation v(t) = 3t^2 is displayed prominently in white.
        # [Lecture Line 1 color changes to #FFFF00]
        v_eq = MathTex(r"v(t) = 3t^2", color=WHITE)
        # Resolved Issue 29: Fixed scale factor to 1.0
        self.place_at_grid(v_eq, 'A2', scale_factor=1.0)
        
        self.play(
            Write(v_eq),
            self.lecture[0].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The anti-derivative t^3 appears next to the equation and scales up.
        # [Lecture Line 2 color changes to #FFFF00]
        f_eq = MathTex(r"F(t) = t^3", color=WHITE)
        # Resolved Issue 30: Fixed scale factor to 1.0
        self.place_at_grid(f_eq, 'B2', scale_factor=1.0)
        
        self.play(
            FadeIn(f_eq),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.play(f_eq.animate.scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the definite integral calculation: [t^3] evaluated from 0 to 2.
        # [Lecture Line 3 color changes to #FFFF00]
        int_calc = MathTex(r"\int_0^2 3t^2 \, dt = \left[ t^3 \right]_0^2", color=WHITE)
        # Place calculation in column 3 to give space
        self.place_at_grid(int_calc, 'C3', scale_factor=1.0)
        
        self.play(
            Write(int_calc),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The numerical result '8' appears inside a bright white circle.
        # [Lecture Line 4 color changes to #FFFF00]
        eval_step = MathTex(r"= 2^3 - 0^3 = 8", color=WHITE)
        self.place_at_grid(eval_step, 'D3', scale_factor=1.0)
        
        result_num = MathTex("8", color=WHITE)
        result_circle = Circle(radius=0.4, color=WHITE, stroke_width=4)
        result_group = VGroup(result_circle, result_num)
        self.place_at_grid(result_group, 'E3', scale_factor=1.0)
        
        self.play(
            Write(eval_step),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        self.play(Create(result_circle), Write(result_num))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The area under the 3t^2 curve fills up to the value of 8 to match the calculation.
        # [Lecture Line 5 color changes to #FFFF00]
        
        # Setup Graph
        axes = Axes(
            x_range=[0, 2.5, 1],
            y_range=[0, 13, 4],
            x_length=3,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True}
        )
        # Positioning labels within 1 grid unit of the axes
        v_label = Text("v(t)", color=WHITE, font_size=18).next_to(axes.y_axis, UP, buff=0.1)
        t_label = Text("t", color=WHITE, font_size=18).next_to(axes.x_axis, RIGHT, buff=0.1)
        graph = axes.plot(lambda t: 3 * t**2, x_range=[0, 2.1], color=WHITE)
        
        graph_group = VGroup(axes, v_label, t_label, graph)
        # Area A4 to F6 covers the right part of the screen
        self.place_in_area(graph_group, 'A4', 'F6', scale_factor=1.0)
        
        area = axes.get_area(graph, x_range=[0, 2], color=BLUE, opacity=0.5)
        area_val_text = Text("Area = 8", color=BLUE, font_size=18)
        area_val_text.move_to(axes.c2p(1, 4))
        
        self.play(
            Create(axes),
            Write(v_label),
            Write(t_label),
            Create(graph)
        )
        
        self.play(
            DrawBorderThenFill(area),
            Write(area_val_text),
            self.lecture[4].animate.set_color("#FFFF00")
        )
        self.wait(3)
