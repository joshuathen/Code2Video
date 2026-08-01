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
        # Data from storyboard
        title_text = "Prerequisite: Independence and Joint Space"
        lecture_lines = [
            "Assume Nutty and Pip gather acorns independently of each other.",
            "The joint probability is simply the product of individual probabilities.",
            "We visualize this joint space as a two-dimensional grid."
        ]
        
        # Setup the layout
        self.setup_layout(title_text, lecture_lines)

        # Colors for lecture lines
        color_1 = "#DFEFFF" # Light Blue
        color_2 = "#00BFFF" # Deep Sky Blue
        color_3 = "#ADD8E6" # Light Blue Variant

        # === Animation for Lecture Line 1 ===
        # Task: Draw a 2D coordinate plane with X and Y axes.
        self.play(self.lecture[0].animate.set_color(color_1))
        
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4.0,
            y_length=4.0,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        
        # Fixing FileNotFoundError: No such file or directory: 'latex' 
        # by using Text instead of string for labels.
        x_label = axes.get_x_axis_label(Text("X", font_size=20), edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label(Text("Y", font_size=20), edge=UP, direction=UP)
        axes_group = VGroup(axes, x_label, y_label)
        
        # Position axes group in the grid area - FIXED as per Issue 24
        # Moving to 'B1'-'F6' and scaling to 0.9 prevents overlap with lecture notes.
        self.place_in_area(axes_group, "B1", "F6", scale_factor=0.9)
        
        self.play(Create(axes), run_time=1.5)
        self.play(Write(x_label), Write(y_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Task: Render a grid with color intensity #00BFFF for P(X)*P(Y).
        self.play(self.lecture[1].animate.set_color(color_2))
        
        # P(X)*P(Y) where X and Y are independent. 
        # A simple 2D Gaussian to simulate joint probability density.
        def intensity_func(x, y):
            return np.exp(-((x-2.5)**2 + (y-2.5)**2) / 2.0)
        
        joint_grid = VGroup()
        cell_count = 10
        step = 5.0 / cell_count
        
        # Use a list to store cells for potentially better performance with lag_ratio
        for i in range(cell_count):
            for j in range(cell_count):
                x_val = (i + 0.5) * step
                y_val = (j + 0.5) * step
                prob = intensity_func(x_val, y_val)
                
                cell = Square(
                    side_length=step * (axes.x_length / 5.0),
                    fill_color="#00BFFF",
                    fill_opacity=prob * 0.85,
                    stroke_width=0.1,
                    stroke_color=WHITE
                )
                cell.move_to(axes.c2p(x_val, y_val))
                joint_grid.add(cell)
        
        self.play(FadeIn(joint_grid, lag_ratio=0.002), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Task: Label point (x,y) showing multiplication of probabilities.
        self.play(self.lecture[2].animate.set_color(color_3))
        
        # Highlight a specific grid cell
        target_x, target_y = 2.5, 2.5 
        highlight = Square(
            side_length=step * (axes.x_length / 5.0),
            color=YELLOW,
            stroke_width=2
        ).move_to(axes.c2p(target_x, target_y))
        
        # Fixing FileNotFoundError: No such file or directory: 'latex'
        # by using Text instead of MathTex.
        formula = Text("P(x,y) = P(x) * P(y)", font_size=20, color=YELLOW)
        # Position formula at B5 grid position - FIXED as per Issue 25
        # Scaling to 0.9 prevents it from being too large or overlapping.
        self.place_at_grid(formula, "B5", scale_factor=0.9)
        
        # Connection from formula to highlight
        connection = Arrow(formula.get_left(), highlight.get_top(), color=YELLOW, buff=0.1, stroke_width=2)
        
        self.play(Create(highlight))
        self.play(Write(formula))
        self.play(GrowArrow(connection))
        self.wait(2)
