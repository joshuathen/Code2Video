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
        title = "The 'Walls' of the Math: Boundary Conditions"
        lines = [
            "Boundaries define the physical limits of our PDE system.",
            "Dirichlet conditions fix the value at the boundary's edge.",
            "Neumann conditions specify the slope or flux across boundaries."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_SEGMENT = "#FFFFFF"
        COLOR_DIRICHLET = "#FF0000"
        COLOR_NEUMANN = "#00FF00"
        COLOR_HIGHLIGHT = YELLOW
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Draw a horizontal white segment (#FFFFFF) with red circles (#FF0000) at endpoints.
        p_left = self.grid["C2"]
        p_right = self.grid["C5"]
        segment = Line(p_left, p_right, color=COLOR_SEGMENT)
        
        circle_l = Circle(radius=0.1, color=COLOR_DIRICHLET, fill_opacity=1)
        circle_r = Circle(radius=0.1, color=COLOR_DIRICHLET, fill_opacity=1)
        circle_l.move_to(p_left)
        circle_r.move_to(p_right)
        
        self.play(Create(segment))
        self.play(FadeIn(circle_l), FadeIn(circle_r))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Label the endpoints 'Dirichlet: Fixed Value' (#FF0000).
        label_dirichlet = Text("Dirichlet: Fixed Value", font_size=20, color=COLOR_DIRICHLET)
        # Fix for Issue 38: Move from B2-B5 to A2-A5, scale 0.7
        self.place_in_area(label_dirichlet, 'A2', 'A5', scale_factor=0.7)
        
        self.play(Write(label_dirichlet))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Replace circles with green tangent lines (#00FF00). Label 'Neumann: Fixed Slope'.
        slope_l = Line(start=UP*0.4 + LEFT*0.2, end=DOWN*0.4 + RIGHT*0.2, color=COLOR_NEUMANN).move_to(p_left)
        slope_r = Line(start=UP*0.4 + LEFT*0.2, end=DOWN*0.4 + RIGHT*0.2, color=COLOR_NEUMANN).move_to(p_right)
        
        label_neumann = Text("Neumann: Fixed Slope", font_size=20, color=COLOR_NEUMANN)
        # Fix for Issue 39: Move from B2-B5 to A2-A5, scale 0.7
        self.place_in_area(label_neumann, 'A2', 'A5', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(circle_l, slope_l),
            ReplacementTransform(circle_r, slope_r),
            ReplacementTransform(label_dirichlet, label_neumann)
        )
        self.wait(2)
