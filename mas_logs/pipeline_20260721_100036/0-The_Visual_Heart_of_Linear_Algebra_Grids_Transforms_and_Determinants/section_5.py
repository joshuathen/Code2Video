from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Data from shared state
        title = "Synthesis: Solving Systems Geometrically"
        lines = [
            "- We can solve systems of equations through geometric visualization.",
            "- We look for a vector that transforms to destination b.",
            "- Vector x is the original instruction leading to b."
        ]
        self.setup_layout(title, lines)
        
        # Colors from storyboard
        highlight_color = "#CC00FF"
        b_color = "#FF3399"
        x_color = "#00CCFF"
        eq_color = "#FFFFFF"

        # Matrix for transformation: a simple shear
        matrix = [[1, 1], [0, 1]]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(highlight_color))
        
        # Show equation "Ax = b" (#FFFFFF) at the top.
        equation = MathTex(r"A", r"\mathbf{x}", r"=", r"\mathbf{b}", font_size=42, color=eq_color)
        equation.set_color_by_tex(r"\mathbf{x}", x_color)
        equation.set_color_by_tex(r"\mathbf{b}", b_color)
        
        # ISSUE 26 FIX: place_in_area(equation, 'A3', 'A5', scale_factor=0.8)
        self.place_in_area(equation, "A3", "A5", scale_factor=0.8)
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(highlight_color)
        )
        
        # Show a target vector "b" (#FF3399) on a transformed grid.
        plane = NumberPlane(
            x_range=[-2, 3, 1],
            y_range=[-1, 2, 1],
            x_length=4,
            y_length=3,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        # Destination state: x=[0,1] maps to b=[1,1] under shear [[1,1],[0,1]]
        b_vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 1), buff=0, color=b_color, stroke_width=6)
        b_label = MathTex(r"\mathbf{b}", color=b_color, font_size=32).next_to(b_vec.get_end(), UR, buff=0.1)
        
        # Original state mobjects (needed later but created now for grouping)
        x_vec = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=x_color, stroke_width=6)
        x_label = MathTex(r"\mathbf{x}", color=x_color, font_size=32).next_to(x_vec.get_end(), UL, buff=0.1)
        
        transformed_plane = plane.copy().apply_matrix(matrix)
        
        # ISSUE 27 FIX: place_in_area(coord_container, 'B2', 'F6', scale_factor=0.9)
        coord_container = VGroup(plane, transformed_plane, b_vec, b_label, x_vec, x_label)
        self.place_in_area(coord_container, "B2", "F6", scale_factor=0.9)
        
        # Show line 2: transformed grid and b
        self.play(
            Create(transformed_plane),
            GrowArrow(b_vec),
            Write(b_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color)
        )
        
        # Show vector "x" (#00CCFF) on the original grid
        # Transition back to original identity grid visually
        self.play(
            FadeOut(transformed_plane),
            FadeIn(plane),
            GrowArrow(x_vec),
            Write(x_label),
            b_vec.animate.set_opacity(0.3),
            b_label.animate.set_opacity(0.3)
        )
        self.wait(1)
        
        # ...being transformed until it lands exactly on "b".
        self.play(
            plane.animate.apply_matrix(matrix),
            x_vec.animate.apply_matrix(matrix),
            # Label follows the tip and fades out as it merges
            x_label.animate.move_to(b_label.get_center()).set_opacity(0),
            run_time=3
        )
        
        # Show final match and highlight result
        self.play(
            b_vec.animate.set_opacity(1),
            b_label.animate.set_opacity(1),
            Indicate(b_vec, color=b_color, scale_factor=1.2)
        )
        self.wait(3)
