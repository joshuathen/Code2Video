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
        # Data
        title_text = "Defining the PDF: Height vs. Area"
        lecture_lines = [
            "The curve f(x) is not the actual probability.",
            "Height f(x) represents the density at a point.",
            "Probability is the area under the curve.",
            "The function f(x) can never be negative.",
            "Total area under the curve must equal one."
        ]
        
        # Setup
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        COLOR_CURVE = WHITE
        COLOR_HEIGHT = "#00FFFF" # Cyan
        COLOR_AREA = "#800080" # Purple
        COLOR_RULE = "#FFFF00" # Yellow
        
        # Create Axes
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1, 0.5],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": WHITE}
        )
        labels = axes.get_axis_labels(
            x_label=Text("x", font_size=24), 
            y_label=Text("f(x)", font_size=24)
        )
        
        # Math curve for area logic (invisible)
        def pdf_func(x):
            return 0.8 * np.exp(-0.5 * ((x - 5) / 0.5)**2)
        curve_math = axes.plot(pdf_func).set_stroke(opacity=0)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg]
        bell_curve = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        bell_curve.set_color(COLOR_CURVE)
        
        # Placement logic
        plot_group = VGroup(axes, labels, curve_math)
        self.place_in_area(plot_group, "B1", "F6", scale_factor=0.9)
        
        # Align SVG to curve_math
        bell_curve.stretch_to_fit_width(curve_math.get_width())
        bell_curve.stretch_to_fit_height(curve_math.get_height())
        bell_curve.move_to(curve_math)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_CURVE)
        self.play(Create(axes), Write(labels))
        self.play(Create(bell_curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_HEIGHT)
        
        x_val = 5
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cyan.svg]
        cyan_line = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cyan.svg")
        cyan_line.set_color(COLOR_HEIGHT)
        
        # Match dimensions to a vertical line at x=5
        line_start = axes.c2p(x_val, 0)
        line_end = axes.c2p(x_val, 0.8)
        cyan_line.stretch_to_fit_height(line_end[1] - line_start[1])
        cyan_line.stretch_to_fit_width(0.05) # Thin line
        cyan_line.move_to(line_start, aligned_edge=DOWN)
        
        dot = Dot(line_end, color=COLOR_HEIGHT)
        
        # Issue 26 Fix: 'A4' scale 0.8
        height_label = Text("f(5) = 0.8", font_size=22, color=COLOR_HEIGHT)
        self.place_at_grid(height_label, "A4", scale_factor=0.8)
        
        self.play(Create(cyan_line), Create(dot))
        self.play(Write(height_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_AREA)
        
        # Use curve_math for the area
        area_small = axes.get_area(curve_math, x_range=[4.7, 5.3], color=COLOR_AREA, opacity=0.6)
        area_label = Text("Area = Prob.", font_size=24, color=COLOR_AREA)
        self.place_at_grid(area_label, "D5", scale_factor=1.0)

        self.play(FadeIn(area_small), Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_RULE)
        
        # Issue 25 Fix: 'A2' scale 0.8
        rule_label = Text("f(x) >= 0", color=COLOR_RULE, font_size=24)
        self.place_at_grid(rule_label, "A2", scale_factor=0.8)
        
        self.play(Write(rule_label))
        self.play(Indicate(bell_curve, color=COLOR_RULE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_AREA)
        
        # Shade entire visible area using curve_math
        area_full = axes.get_area(curve_math, x_range=[0, 10], color=COLOR_AREA, opacity=0.4)
        
        # Issue 27 Fix: 'F5'-'F6' scale 0.8
        total_area_label = Text("Total Area = 1", color=COLOR_AREA, font_size=24)
        self.place_in_area(total_area_label, "F5", "F6", scale_factor=0.8)

        self.play(
            FadeOut(area_small),
            FadeOut(area_label),
            FadeIn(area_full),
            Write(total_area_label)
        )
        self.wait(2)
