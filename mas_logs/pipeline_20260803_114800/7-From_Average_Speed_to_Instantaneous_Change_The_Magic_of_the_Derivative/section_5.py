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
        # Section 5 Data
        title = "The Tangent Line and the Derivative"
        lecture_lines = [
            "The secant line finally becomes a tangent line.",
            "It touches the curve at exactly one point.",
            "This slope is the instantaneous rate of change.",
            "We call this new value the derivative.",
            "The formal limit definition defines this magic moment."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        ORANGE = "#FFA500"
        BLUE = "#0000FF"
        YELLOW = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"
        RED_COLOR = "#FF0000"

        # Helper for lines
        func = lambda x: 0.5 * x**2
        
        # === Animation for Lecture Line 1 ===
        # Create Graph
        # Issue 28: Scale axes to 0.7 in C2-F6
        axes = Axes(
            x_range=[-0.5, 3.5, 1],
            y_range=[-0.5, 4.5, 1],
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.7)
        
        curve = axes.plot(func, x_range=[-0.2, 3], color=WHITE_COLOR)
        
        # Points for Secant
        x_a = 1.0
        x_b = 2.5
        dot_a = Dot(axes.c2p(x_a, func(x_a)), color=RED_COLOR)
        dot_b = Dot(axes.c2p(x_b, func(x_b)), color=BLUE)
        
        # Label for Point A
        label_a = MathTex("A", font_size=24).next_to(dot_a, LEFT, buff=0.1)
        
        def get_line(x1, x2, color, length=4):
            y1, y2 = func(x1), func(x2)
            p1_coord = np.array([x1, y1, 0])
            p2_coord = np.array([x2, y2, 0])
            
            if abs(x2 - x1) > 0.0001:
                direction = p2_coord - p1_coord
            else:
                # Tangent direction: [1, f'(x1), 0]. f'(x) = x for 0.5x^2
                direction = np.array([1, x1, 0]) 
            
            direction = direction / np.linalg.norm(direction)
            mid_coord = (p1_coord + p2_coord) / 2
            
            start_coord = mid_coord - direction * (length / 2)
            end_coord = mid_coord + direction * (length / 2)
            
            return Line(axes.c2p(*start_coord), axes.c2p(*end_coord), color=color)

        secant_line = get_line(x_a, x_b, BLUE, length=3.5)

        self.add(axes, curve, dot_a, dot_b, label_a, secant_line)
        self.wait(1)

        # Transition to Tangent
        tangent_line = get_line(x_a, x_a, ORANGE, length=4.0)
        
        # Matching color for Lecture Line 1
        self.play(self.lecture[0].animate.set_color(ORANGE))
        self.play(
            Transform(secant_line, tangent_line),
            dot_b.animate.move_to(dot_a.get_center()).set_opacity(0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color for Lecture Line 2
        self.play(self.lecture[1].animate.set_color(ORANGE))
        
        # Issue 27: Scale tangent_label to 0.7 at C5
        tangent_label = Text("Tangent: Instantaneous Rate", font_size=16, color=ORANGE)
        self.place_at_grid(tangent_label, 'C5', scale_factor=0.7)
        
        self.play(Write(tangent_label))
        self.play(Indicate(dot_a, color=ORANGE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching color for Lecture Line 3 (Orange refers to slope)
        self.play(self.lecture[2].animate.set_color(ORANGE))
        
        # Formula: f'(x) = lim_{h -> 0} (f(x+h) - f(x)) / h
        # Issue 26: Scale formula to 0.8 in A3-B6
        formula = MathTex(
            "f'(x)", "=", "\\lim_{h \\to 0}", "\\frac{f(x+h) - f(x)}{h}",
            font_size=32, color=WHITE_COLOR
        )
        self.place_in_area(formula, 'A3', 'B6', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Matching color for Lecture Line 4
        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        # Flash "h -> 0" (index 2)
        self.play(formula[2].animate.set_color(YELLOW))
        self.play(Flash(formula[2], color=YELLOW, flash_radius=0.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Matching color for Lecture Line 5
        self.play(self.lecture[4].animate.set_color(WHITE_COLOR))
        
        # Arrow from tangent slope label to f'(x) in the formula
        arrow = Arrow(
            start=tangent_label.get_top(),
            end=formula[0].get_bottom(),
            color=WHITE_COLOR,
            buff=0.1
        )
        self.play(Create(arrow))
        self.play(Indicate(formula[0], color=ORANGE))
        self.wait(2)
