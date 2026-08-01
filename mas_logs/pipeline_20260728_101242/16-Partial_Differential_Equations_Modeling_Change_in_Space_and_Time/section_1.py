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
        lecture_lines = [
            "ODEs track changes over one variable, like time.",
            "But reality often involves multiple variables, like position.",
            "PDEs describe how systems evolve in space and time."
        ]
        self.setup_layout("From One to Many: The PDE Intuition", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw a vertical rectangle (#FF0000) representing a thermometer. Label horizontal axis 't' (#FFFFFF).
        self.lecture[0].set_color(YELLOW)
        thermometer = Rectangle(height=3, width=0.6, color="#FF0000", fill_opacity=0.8)
        self.place_in_area(thermometer, "B3", "E3")
        
        t_axis = Line(start=self.grid["F2"], end=self.grid["F4"], color=WHITE)
        t_label = MathTex("t", color="#FFFFFF")
        # Fix: Centering t_label under the thermometer column (Issue 25)
        self.place_at_grid(t_label, "F3", scale_factor=0.8)
        
        self.play(Create(thermometer), Create(t_axis), Write(t_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Transform the rectangle into a 2D grid of small squares (#FFFFFF) to represent space.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        grid_rows, grid_cols = 5, 5
        grid_squares = VGroup(*[
            Square(side_length=0.4, color="#FFFFFF", stroke_width=2)
            for _ in range(grid_rows * grid_cols)
        ]).arrange_in_grid(rows=grid_rows, cols=grid_cols, buff=0.05)
        
        self.place_in_area(grid_squares, "B2", "E5")
        
        space_label = MathTex("x, y", color="#FFFFFF")
        # Fix: Centering space_label under the grid spanning col 2-5 (Issue 26)
        self.place_in_area(space_label, "F3", "F4", scale_factor=0.8)

        self.play(
            Transform(thermometer, grid_squares),
            FadeOut(t_axis),
            Transform(t_label, space_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Color central squares red (#FF0000) and outer squares blue (#0000FF). Animate heat spreading.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Use a copy for the colored grid to handle the animation transition cleanly
        colored_grid = grid_squares.copy()
        
        def get_color_for_dist(dist, max_dist, heat_factor):
            # heat_factor goes from 0 to 1
            # intensity = 1 (red) at center, 0 (blue) at edges
            normalized_dist = dist / max_dist
            intensity = np.clip(1.0 - normalized_dist + (heat_factor * 0.8) - 0.2, 0, 1)
            return interpolate_color(BLUE, RED, intensity)

        # Initial colors
        max_dist = 3.0
        for i in range(grid_rows):
            for j in range(grid_cols):
                dist = np.sqrt((i - grid_rows//2)**2 + (j - grid_cols//2)**2)
                idx = i * grid_cols + j
                colored_grid[idx].set_fill(get_color_for_dist(dist, max_dist, 0), opacity=0.8)

        self.play(FadeIn(colored_grid), FadeOut(thermometer))
        
        heat_tracker = ValueTracker(0)
        
        def update_grid(obj):
            h = heat_tracker.get_value()
            for i in range(grid_rows):
                for j in range(grid_cols):
                    dist = np.sqrt((i - grid_rows//2)**2 + (j - grid_cols//2)**2)
                    idx = i * grid_cols + j
                    obj[idx].set_fill(get_color_for_dist(dist, max_dist, h), opacity=0.8)

        colored_grid.add_updater(update_grid)
        self.play(heat_tracker.animate.set_value(1), run_time=4)
        colored_grid.remove_updater(update_grid)
        
        self.wait(3)
