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

class Section1Scene(TeachingScene):
    def construct(self):
        # Configuration
        title = "The Summation Machine: A Prerequisite Look"
        lines = [
            "Meet Sigma, our infinite summation robot.",
            "Summing one over n creates a stack growing forever.",
            "Other sums converge to a finite height."
        ]
        self.setup_layout(title, lines)

        # Colors
        ROBOT_COLOR = "#00AEFF"
        DIVERGENT_COLOR = "#FF0000"
        CONVERGENT_COLOR = "#00FF00"
        LIMIT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ROBOT_COLOR)
        
        # Asset integration [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg]
        robot_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        robot_svg.set_color(ROBOT_COLOR)
        
        # Create a circle and a Sigma inside to form the 'Robot Sigma' icon
        sigma_circle = Circle(radius=0.5, color=ROBOT_COLOR)
        sigma_char = Text("Σ", color=ROBOT_COLOR).scale(1.2)
        sigma_char.move_to(sigma_circle.get_center())
        
        # Placing SVG above the sigma circle to create a unified robot icon
        robot_svg.scale(0.4).next_to(sigma_circle, UP, buff=0.1)
        robot_group = VGroup(robot_svg, sigma_circle, sigma_char)
        
        self.place_at_grid(robot_group, "B1", scale_factor=0.8)
        
        self.play(Create(robot_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(DIVERGENT_COLOR)
        
        # Grid positioning: stack base at the bottom center of the visual area
        base_point = self.grid["F3"] + DOWN * 0.5
        num_terms = 20
        harmonic_bars = VGroup()
        current_y = base_point[1]
        
        # Height scaling to ensure the stack exceeds the screen height
        # Harmonic sum H_20 is ~3.6. 3.6 * 2.5 = 9.0 units (screen height is 8.0)
        scale_h = 2.5
        
        for n in range(1, num_terms + 1):
            h = (1.0 / n) * scale_h
            bar = Rectangle(
                width=1.0, 
                height=h, 
                fill_color=DIVERGENT_COLOR, 
                fill_opacity=0.7, 
                stroke_width=1
            )
            bar.move_to([base_point[0], current_y + h/2, 0])
            harmonic_bars.add(bar)
            current_y += h

        # Label for the growing stack (Issue 32 fix)
        label_harmonic = Text("Divergent", font_size=20, color=DIVERGENT_COLOR)
        self.place_in_area(label_harmonic, 'D4', 'D6', scale_factor=0.6)
        
        self.play(
            LaggedStart(*[FadeIn(bar, shift=UP*0.1) for bar in harmonic_bars], lag_ratio=0.08),
            Write(label_harmonic),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(CONVERGENT_COLOR)
        
        # Create convergent bars for 1/n^2 series
        convergent_bars = VGroup()
        current_y_conv = base_point[1]
        
        for n in range(1, num_terms + 1):
            h = (1.0 / (n**2)) * scale_h
            bar = Rectangle(
                width=1.0, 
                height=h, 
                fill_color=CONVERGENT_COLOR, 
                fill_opacity=0.7, 
                stroke_width=1
            )
            bar.move_to([base_point[0], current_y_conv + h/2, 0])
            convergent_bars.add(bar)
            current_y_conv += h

        # Convergent label (Issue 33 fix)
        label_convergent = Text("Convergent Series", font_size=20, color=CONVERGENT_COLOR)
        self.place_in_area(label_convergent, 'D4', 'D6', scale_factor=0.6)

        # Convergent limit horizontal line and label
        # Limit sum_{1}^inf 1/n^2 = pi^2/6 approx 1.64493
        limit_height_units = 1.64493 * scale_h
        limit_y = base_point[1] + limit_height_units
        
        dashed_line = DashedLine(
            start=[base_point[0] - 1.5, limit_y, 0],
            end=[base_point[0] + 1.5, limit_y, 0],
            color=LIMIT_COLOR
        )
        
        # Limit label (Issue 34 fix)
        limit_label = Text("Convergent Limit", font_size=20, color=CONVERGENT_COLOR)
        self.place_in_area(limit_label, 'C4', 'C6', scale_factor=0.6)

        self.play(
            ReplacementTransform(harmonic_bars, convergent_bars),
            ReplacementTransform(label_harmonic, label_convergent),
            Create(dashed_line),
            Write(limit_label),
            run_time=2
        )
        self.wait(2)
