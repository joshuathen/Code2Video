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
        # Data and Setup
        self.setup_layout("The Local Zoom: Linear Approximation", [
            "Non-linear functions stretch space differently at each point.",
            "Let's zoom into a single point on a curve.",
            "Locally, every smooth transformation looks like constant scaling.",
            "A tiny input interval maps to an output interval.",
            "Zooming in makes any curve look perfectly linear."
        ])

        # === Animation for Lecture Line 1 ===
        # Non-linear functions stretch space differently at each point.
        self.lecture[0].set_color(BLUE_C)
        
        # Input line (x: 0 to 4)
        input_line = NumberLine(
            x_range=[0, 4, 1],
            length=1.25,
            include_numbers=True,
            color=TEAL,
            label_direction=UP,
            font_size=18,
            stroke_width=2
        )
        # Output line (f(x)=x^2: 0 to 16)
        output_line = NumberLine(
            x_range=[0, 16, 4],
            length=5.0,
            include_numbers=True,
            color=PINK,
            label_direction=DOWN,
            font_size=18,
            stroke_width=2
        )
        
        # Resolved Issue 32: Positioning input_line at B5 to avoid lecture area
        self.place_at_grid(input_line, 'B5', scale_factor=0.8)
        
        # Resolved Issue 30: Positioning output_line at E4-E6 area
        self.place_in_area(output_line, 'E4', 'E6', scale_factor=1.0)
        
        # Mapping arrows
        arrow_x_vals = [0.8, 1.4, 2.0, 2.6, 3.2]
        arrows = VGroup(*[
            Arrow(
                start=input_line.number_to_point(x),
                end=output_line.number_to_point(x**2),
                buff=0.05,
                color=GRAY,
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.1
            ) for x in arrow_x_vals
        ])
        
        mapping_group = VGroup(input_line, output_line, arrows)
        self.play(Create(input_line), Create(output_line))
        self.play(Create(arrows), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Let's zoom into a single point on a curve.
        self.lecture[1].set_color(YELLOW)
        
        x_center = 2.0
        dx = 0.4
        
        # Highlight dx on input line
        dx_start = input_line.number_to_point(x_center - dx/2)
        dx_end = input_line.number_to_point(x_center + dx/2)
        dx_rect = Line(dx_start, dx_end, color="#FFD700", stroke_width=6)
        dx_label = MathTex("dx", color="#FFD700", font_size=24).next_to(dx_rect, UP, buff=0.1)
        
        self.play(Create(dx_rect), Write(dx_label))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Locally, every smooth transformation looks like constant scaling.
        self.lecture[2].set_color(ORANGE)
        
        # Highlight corresponding df on output line
        # f(x) = x^2, df approx 4 * dx at x=2
        f_start = (x_center - dx/2)**2
        f_end = (x_center + dx/2)**2
        df_start = output_line.number_to_point(f_start)
        df_end = output_line.number_to_point(f_end)
        df_rect = Line(df_start, df_end, color=ORANGE, stroke_width=6)
        df_label = MathTex("df", color=ORANGE, font_size=24).next_to(df_rect, DOWN, buff=0.1)
        
        self.play(Create(df_rect), Write(df_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A tiny input interval maps to an output interval.
        self.lecture[3].set_color(WHITE)
        
        # Emphasize mapping by adding arrows for the boundaries
        boundary_arrows = VGroup(
            Arrow(dx_start, df_start, color="#FFD700", stroke_width=2, buff=0.05),
            Arrow(dx_end, df_end, color="#FFD700", stroke_width=2, buff=0.05)
        )
        self.play(Create(boundary_arrows))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Zooming in makes any curve look perfectly linear.
        self.lecture[4].set_color(WHITE)
        
        # Group everything for the zoom
        zoom_group = VGroup(
            mapping_group, dx_rect, dx_label, df_rect, df_label, boundary_arrows
        )
        
        # Point to zoom into (average center between intervals)
        zoom_pivot = (input_line.number_to_point(x_center) + output_line.number_to_point(x_center**2)) / 2
        
        # Zoom scale factor
        scale_factor = 8.0
        
        # Target position: middle of the right grid area (C1 to D6 area)
        target_center = (self.grid["C1"] + self.grid["D6"]) / 2
        
        # Hide line numbers during zoom for performance and clarity
        self.play(
            input_line.numbers.animate.set_opacity(0),
            output_line.numbers.animate.set_opacity(0),
            dx_label.animate.scale(0.5).set_opacity(0),
            df_label.animate.scale(0.5).set_opacity(0),
        )
        
        self.play(
            zoom_group.animate.scale(scale_factor, about_point=zoom_pivot).move_to(target_center),
            run_time=3
        )
        
        # Resolved Issue 31: Positioning ll_text at A5-B6 to avoid center overlap
        ll_text = Text("Local Linearity", color=WHITE, font_size=32)
        self.place_in_area(ll_text, 'A5', 'B6', scale_factor=0.7)
        
        # Region boundary
        region_box = SurroundingRectangle(VGroup(dx_rect, df_rect), color=WHITE, buff=0.4)
        
        self.play(Create(region_box), Write(ll_text))
        self.wait(2)
