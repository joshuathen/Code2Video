from manim import *
import numpy as np

# Base class provided in the prompt
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
        # Data from storyboard
        title_str = "Reversing the Process (The Inverse)"
        lecture_lines = [
            "To go the other way, use the inverse matrix.",
            "Multiply standard coordinates by this inverse matrix.",
            "This finds the coordinates in the robot's world."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        COLOR_FORMULA = WHITE
        COLOR_POINT = "#FFA500" # Orange
        COLOR_STD_GRID = BLUE_D
        COLOR_NEW_GRID = GREEN_E

        # === Animation for Lecture Line 1 ===
        # Step 1: Display the inverse formula using Text to avoid FileNotFoundError: 'latex'
        self.lecture[0].set_color(YELLOW)
        
        # Use Text instead of MathTex to bypass the requirement for a system LaTeX installation
        formula = Text("[x]_new = P^-1 [x]_std", color=COLOR_FORMULA, font_size=24)
        # Issue 33: Use Row A for formula to avoid bottom-heavy layout
        self.place_in_area(formula, 'A2', 'A6', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Plot a point at (3,3) in the standard grid in orange (#FFA500).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create standard grid
        std_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            background_line_style={"stroke_color": COLOR_STD_GRID, "stroke_width": 1, "stroke_opacity": 0.5},
            axis_config={"include_tip": True, "stroke_width": 2}
        )
        # Issue 34: Use B2-F6 for grid to reduce vertical crowding
        self.place_in_area(std_grid, 'B2', 'F6', scale_factor=0.6)
        
        # Point (3,3)
        point_std = Dot(std_grid.c2p(3, 3), color=COLOR_POINT)
        label_std = Text("(3, 3)_std", color=COLOR_POINT, font_size=18)
        label_std.next_to(point_std, UR, buff=0.1) 
        
        self.play(Create(std_grid))
        self.play(FadeIn(point_std), Write(label_std))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: Show the point's coordinates in the skewed grid after applying the inverse matrix P⁻¹.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Skewed grid (P = [[2, 1], [1, 2]])
        skewed_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            background_line_style={"stroke_color": COLOR_NEW_GRID, "stroke_width": 2, "stroke_opacity": 0.3},
            axis_config={"stroke_width": 0}
        )
        # Apply transformation to the Mobject to represent the robot's tilted grid
        skewed_grid.apply_matrix([[2, 1], [1, 2]])
        # Issue 35: Perfect overlay with the standard grid
        self.place_in_area(skewed_grid, 'B2', 'F6', scale_factor=0.6)
        
        # Basis vectors for visualization (Context: P columns)
        v1 = Arrow(start=std_grid.c2p(0,0), end=std_grid.c2p(2,1), buff=0, color=GREEN, stroke_width=4)
        v2 = Arrow(start=std_grid.c2p(0,0), end=std_grid.c2p(1,2), buff=0, color=YELLOW, stroke_width=4)
        
        # New coordinates label [1, 1] relative to skewed basis
        label_new = Text("(1, 1)_new", color=GREEN, font_size=18)
        label_new.next_to(point_std, DR, buff=0.1)
        
        self.play(Create(skewed_grid), Create(v1), Create(v2))
        self.play(Write(label_new), label_std.animate.set_opacity(0.5))
        self.wait(2)
