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
        # Fetching title and lines from the shared storyboard
        title_text = "The Concept of Linear Transformation (1:30-2:30)"
        lecture_lines = [
            "A transformation morphs the entire coordinate grid.",
            "In linear transformations, the origin always remains fixed.",
            "Grid lines must stay parallel and evenly spaced.",
            "Watch how the space warps while keeping structure.",
            "This visual shift represents a mathematical function."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard requirements
        COLOR_GRID = "#FFFFFF"
        COLOR_ORIGIN = "#FFFF00"
        COLOR_HIGHLIGHT = "#00FFFF"

        # Asset Paths
        ASSET_GRID = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"
        ASSET_DOT = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dot.svg"

        # Initialize Assets
        # Loading SVG assets as specified in the storyboard and issues
        grid_svg = SVGMobject(ASSET_GRID)
        grid_svg.set_color(COLOR_GRID)
        grid_svg.set_stroke(width=2)
        
        origin_dot = SVGMobject(ASSET_DOT)
        origin_dot.set_color(COLOR_ORIGIN)
        # Scale dot relative to the grid size
        origin_dot.scale(0.15)
        
        # Group them to manage as a single visual unit
        grid_group = VGroup(grid_svg, origin_dot)
        
        # Refined positioning based on Issue 34: prevent overlap with lecture text (Col 1)
        # and prevent hitting screen boundaries by using a restricted area and scale.
        self.place_in_area(grid_group, 'C2', 'E6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # L1: "A transformation morphs the entire coordinate grid."
        self.lecture[0].set_color(COLOR_GRID)
        self.wait(1.5)
        # Fade in the coordinate system
        self.play(FadeIn(grid_group))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # L2: "In linear transformations, the origin always remains fixed."
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color(COLOR_ORIGIN)
        self.wait(2.0)
        
        # Storyboard Anim 2: Morph the grid to a skewed state.
        # Matrix for a shear transformation: (x, y) -> (x + y, y)
        skew_matrix = np.array([[1.0, 1.0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        self.play(
            grid_svg.animate.apply_matrix(skew_matrix),
            run_time=2,
            rate_func=rate_functions.smooth
        )
        self.wait(1.5)
        
        # Storyboard Anim 3: Flash the origin dot to emphasize invariance
        self.play(Indicate(origin_dot, color=COLOR_ORIGIN, scale_factor=1.5))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # L3: "Grid lines must stay parallel and evenly spaced."
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        self.wait(1.5)
        
        # Storyboard Anim 4: Use SurroundingRectangles to highlight parallel nature.
        # Create temporary lines that align with the skewed horizontal grid lines for highlighting.
        grid_center = grid_group.get_center()
        
        # Highlight top and bottom horizontal lines (which are still horizontal after shear)
        # Offset slightly to align with visual grid lines
        line_top = Line(grid_center + LEFT*0.8 + UP*0.8, grid_center + RIGHT*2.4 + UP*0.8, color=COLOR_HIGHLIGHT, stroke_width=0)
        line_bottom = Line(grid_center + LEFT*2.4 + DOWN*0.8, grid_center + RIGHT*0.8 + DOWN*0.8, color=COLOR_HIGHLIGHT, stroke_width=0)
        
        rect_1 = SurroundingRectangle(line_top, color=COLOR_HIGHLIGHT, buff=0.1)
        rect_2 = SurroundingRectangle(line_bottom, color=COLOR_HIGHLIGHT, buff=0.1)
        
        self.play(Create(rect_1), Create(rect_2))
        self.wait(1.5)
        self.play(FadeOut(rect_1), FadeOut(rect_2))
        self.wait(1.0)

        # === Animation for Lecture Line 4 ===
        # L4: "Watch how the space warps while keeping structure."
        self.lecture[2].set_color("#FFFFFF")
        self.lecture[3].set_color(COLOR_GRID)
        self.wait(2.0)
        # Observation pause
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # L5: "This visual shift represents a mathematical function."
        self.lecture[3].set_color("#FFFFFF")
        self.lecture[4].set_color(COLOR_GRID)
        self.wait(2.0)
        
        # Storyboard Anim 5: Morph the grid back to its original state using the inverse transformation.
        inv_skew_matrix = np.linalg.inv(skew_matrix)
        self.play(
            grid_svg.animate.apply_matrix(inv_skew_matrix),
            run_time=2,
            rate_func=rate_functions.smooth
        )
        self.wait(2.0)
