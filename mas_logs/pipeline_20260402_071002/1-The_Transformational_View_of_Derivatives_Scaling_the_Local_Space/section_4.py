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
        # 1. Setup Layout
        lecture_lines = [
            "Let's map intervals between two parallel number lines.",
            "A blue window slides steadily along the input line.",
            "The red output window responds by changing its width.",
            "Notice the output window stretches and shrinks dynamically.",
            "The derivative is the ratio of these window widths."
        ]
        self.setup_layout("Visualizing Parallel Number Lines", lecture_lines)
        
        # Mapping function definition
        def f(x):
            return x**2
        
        # === Animation for Lecture Line 1 ===
        # Visualize mapping between two parallel number lines.
        input_line = NumberLine(
            x_range=[-3, 3, 1], 
            length=5, 
            include_numbers=True, 
            label_constructor=Text,
            color="#FFFFFF",
            font_size=16
        )
        output_line = NumberLine(
            x_range=[0, 6, 1], 
            length=5, 
            include_numbers=True, 
            label_constructor=Text,
            color="#FFFFFF",
            font_size=16
        )
        
        input_label = Text("Input (x)", font_size=20, color="#FFFFFF")
        output_label = Text("Output (f(x)=x²)", font_size=20, color="#FFFFFF")
        
        # Positioning using grid system
        self.place_in_area(input_line, "B1", "B6")
        # Issue 33 Fix: input_label area constraint
        self.place_in_area(input_label, "A1", "A2", scale_factor=0.7)
        
        self.place_in_area(output_line, "E1", "E6")
        # Issue 32 Fix: output_label area constraint
        self.place_in_area(output_label, "D1", "D3", scale_factor=0.7)
        
        self.play(
            Create(input_line), 
            Create(output_line), 
            Write(input_label), 
            Write(output_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # A blue window slides steadily along the input line.
        # Highlight lecture line in blue
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        
        x_tracker = ValueTracker(-2.0)
        dx = 0.5
        
        # Issue 22 Fix: Integrate window asset icon
        # Load asset once for performance and robustness
        window_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/window.svg")
        
        # Calculate unit size on screen manually to avoid potential AttributeError with get_unit_size
        input_unit_len = input_line.number_to_point(1)[0] - input_line.number_to_point(0)[0]
        
        blue_window = window_svg.copy()
        blue_window.set_color("#0000FF")
        blue_window.set_opacity(0.3)
        blue_window.stretch_to_fit_width(dx * input_unit_len)
        blue_window.stretch_to_fit_height(0.4)
        
        # Update window position as x_tracker changes
        blue_window.add_updater(lambda m: m.move_to(input_line.number_to_point(x_tracker.get_value())))
        
        self.play(FadeIn(blue_window))
        self.wait(0.5)
        
        # === Animation for Lecture Line 3 ===
        # The red output window responds by changing its width.
        # Highlight lecture line in red
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Create red rectangle with dynamic boundary mapping
        red_rect = Rectangle(
            width=0.1, 
            height=0.4,
            fill_color="#FF0000",
            fill_opacity=0.3,
            stroke_width=2,
            stroke_color="#FF0000"
        )
        
        def update_red_rect(m):
            curr_x = x_tracker.get_value()
            x1, x2 = curr_x - dx/2, curr_x + dx/2
            # Boundary mapping via f(x)=x^2
            y1, y2 = f(x1), f(x2)
            p1 = output_line.number_to_point(y1)
            p2 = output_line.number_to_point(y2)
            center = (p1 + p2) / 2
            w = abs(p2[0] - p1[0])
            m.stretch_to_fit_width(max(w, 0.01))
            m.move_to(center)
            
        red_rect.add_updater(update_red_rect)
        
        self.play(FadeIn(red_rect))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # Notice the output window stretches and shrinks dynamically.
        # Highlight in yellow to signify observation
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Movement from x = -2 to x = 0
        self.play(x_tracker.animate.set_value(0.0), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # The derivative is the ratio of these window widths.
        # Highlight lecture line in green
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        ratio_prefix = Text("Ratio: ", font_size=22, color="#00FF00")
        ratio_val = DecimalNumber(
            0.0,
            num_decimal_places=2,
            color="#00FF00",
            mob_class=Text
        )
        
        # Dynamic calculation of the ratio
        ratio_val.add_updater(lambda d: d.set_value(
            abs(f(x_tracker.get_value() + dx/2) - f(x_tracker.get_value() - dx/2)) / dx
        ))
        
        ratio_group = VGroup(ratio_prefix, ratio_val).arrange(RIGHT, buff=0.15)
        # Issue 31 Fix: Position ratio_group to avoid overlap with lecture text
        self.place_in_area(ratio_group, "C4", "C6", scale_factor=0.6)
        
        self.play(Write(ratio_group))
        
        # Movement from x = 0 to x = 2 to demonstrate changing ratio
        self.play(x_tracker.animate.set_value(2.0), run_time=3, rate_func=linear)
        self.wait(2)
