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

class Section3Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title Setup
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=20, color=WHITE) for line in lecture_lines]
        self.lecture_group = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture_group.to_edge(LEFT, buff=0.7)
        self.add(self.lecture_group)

        # Define animation grid for the right side of the screen
        # x ranges from roughly 1.0 to 6.0, y ranges from -3.0 to 2.0
        self.grid_map = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Calculate coordinates for the right half of the canvas
                x_coord = 1.5 + (j * 0.8)
                y_coord = 2.0 - (i * 0.8)
                self.grid_map[f"{row}{col}"] = np.array([x_coord, y_coord, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """Helper to place mobjects on the defined grid."""
        mobject.scale(scale_factor)
        if grid_pos in self.grid_map:
            mobject.move_to(self.grid_map[grid_pos])
        return mobject

    def construct(self):
        # Define content for the scene
        title_content = "The Rhythm of Change: Derivative Formulas"
        bullets = [
            "- Power Rule: d/dx [x^n] = n*x^(n-1)",
            "- Constant Rule: d/dx [c] = 0",
            "- Sum Rule: (f+g)' = f' + g'",
            "- Geometric Interpretation",
            "- Slope of the Tangent Line"
        ]

        # Initialize layout
        self.setup_layout(title_content, bullets)

        # Visual Demonstration on the Right side
        # Create a coordinate system in the grid area
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3,
            y_length=3,
            axis_config={"color": BLUE, "include_tip": False}
        )
        # Position axes in the middle of the right grid area
        axes.move_to(self.grid_map["C4"])
        
        # Plot a simple parabola segment
        parabola = axes.plot(lambda x: 0.5 * x**2, x_range=[0, 2.5], color=YELLOW)
        
        # Tangent line logic
        dot = Dot(axes.c2p(2, 2), color=RED)
        tangent = Line(
            start=axes.c2p(1, 0), 
            end=axes.c2p(3, 4), 
            color=WHITE, 
            stroke_width=2
        )

        # Display Animations
        self.play(Create(axes), run_time=1)
        self.play(Create(parabola), run_time=1.5)
        self.play(FadeIn(dot), Create(tangent))
        self.wait(1)

        # Highlighting the Power Rule Formula
        formula = MathTex(r"\frac{d}{dx} x^n = n x^{n-1}", color=GOLD).scale(1.1)
        # Move formula to the bottom right grid area
        self.place_at_grid(formula, "F3")
        
        self.play(Write(formula))
        self.play(Indicate(formula))
        
        # Move objects to grid positions to demonstrate grid functionality
        self.play(
            dot.animate.move_to(self.grid_map["A6"]),
            formula.animate.move_to(self.grid_map["B6"]).scale(0.7),
            run_time=2
        )

        self.wait(2)
