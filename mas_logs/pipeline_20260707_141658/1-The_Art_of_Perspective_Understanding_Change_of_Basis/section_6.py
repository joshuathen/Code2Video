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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        title_text = "Summary and Visual Wrap-up"
        lecture_lines = [
            "Vectors are absolute, but their coordinates are relative.",
            "The transition matrix P connects different coordinate perspectives.",
            "This concept is fundamental to modern science and technology."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Vectors are absolute, but their coordinates are relative.
        self.lecture[0].set_color(YELLOW)
        
        # Standard Grid (#444444)
        std_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#444444", "stroke_opacity": 0.6},
            axis_config={"stroke_color": "#444444"}
        )
        # Fix 39: Adjusted area to A1-E6 and scale to 0.7 to avoid obstruction
        self.place_in_area(std_grid, "A1", "E6", scale_factor=0.7)
        
        # White vector arrow (#FFFFFF)
        grid_center = std_grid.get_center()
        # Arrow from (0,0) to (2,1) in plane units. 
        # With scale_factor=0.7, 1 unit = 0.7 manim units.
        vec = Arrow(
            grid_center,
            grid_center + np.array([1.4, 0.7, 0]), 
            buff=0,
            color="#FFFFFF",
            stroke_width=6
        )

        self.play(Create(std_grid), GrowArrow(vec), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The transition matrix P connects different coordinate perspectives.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create target tilted purple grid (#AA00FF)
        tilted_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#AA00FF", "stroke_opacity": 0.6},
            axis_config={"stroke_color": "#AA00FF"}
        )
        # Fix 40: Consistency with std_grid area and scale
        self.place_in_area(tilted_grid, "A1", "E6", scale_factor=0.7)
        
        # Apply transformation to the tilted grid
        matrix = [[1.2, -0.5], [0.3, 1.0]]
        tilted_grid.apply_matrix(matrix)

        # Morph the grid while keeping the vector still
        self.play(
            Transform(std_grid, tilted_grid),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This concept is fundamental to modern science and technology.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Final formula [v]_S = P [v]_B - Using Text due to environment constraints
        formula = Text("[v]_S = P [v]_B", color=WHITE)
        # Fix 38: Position formula in E2-F5 area and scale 0.8 to avoid obscuring the vector
        self.place_in_area(formula, "E2", "F5", scale_factor=0.8)
        
        # Black background for formula for legibility over the grid
        bg_rect = SurroundingRectangle(formula, color=BLACK, fill_opacity=0.8, buff=0.2)
        formula_group = VGroup(bg_rect, formula)

        self.play(FadeIn(formula_group))
        # Scale it up for emphasis
        self.play(formula_group.animate.scale(1.2), run_time=1.5)
        self.wait(2)
        
        # Fade to black
        self.play(FadeOut(VGroup(self.title, self.lecture, std_grid, vec, formula_group)))
        self.wait(1)
