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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup Layout
        title_str = "Calculating the New Reality"
        lines = [
            "Use matrix multiplication to find a vector's new position.",
            "Scale and add the transformed basis vectors.",
            "This linear combination determines the final coordinates."
        ]
        self.setup_layout(title_str, lines)

        # Highlighting colors
        COLOR_L1 = WHITE
        COLOR_L2 = RED # Matches "red basis vector"
        COLOR_L3 = YELLOW # Matches "yellow vector"

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(COLOR_L1))
        
        # Matrix multiplication display [1, 3; -2, 0] * [2, 3] = [11, -4]
        # Since latex is avoided, use text to represent matrices
        equation_text = Text(
            "[1 3; -2 0] × [2; 3] = [11; -4]",
            color="#FFFFFF",
            font_size=24
        )
        self.place_in_area(equation_text, 'A1', 'B6', scale_factor=0.8)
        self.play(Write(equation_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_L2)
        )

        # Initialize Coordinate System
        # Accommodating result (11, -4)
        plane = NumberPlane(
            x_range=[-1, 13, 2],
            y_range=[-5, 2, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={
                "include_numbers": True, 
                "font_size": 16,
                "label_constructor": Text
            }
        )
        
        # Group plane for placement
        plane_group = VGroup(plane)
        # Issue 34 fix: set scale factor to 0.8
        self.place_in_area(plane_group, 'C1', 'F6', scale_factor=0.8)
        
        self.play(Create(plane))

        # Transformed basis vectors (Matrix cols: [1, -2] and [3, 0])
        origin = plane.coords_to_point(0, 0)
        
        i_new_tip = plane.coords_to_point(1, -2)
        j_new_tip = plane.coords_to_point(3, 0)
        
        i_new_arrow = Arrow(origin, i_new_tip, color=RED, buff=0, stroke_width=4)
        j_new_arrow = Arrow(origin, j_new_tip, color=GREEN, buff=0, stroke_width=4)
        
        i_label = Text("i_new", color=RED, font_size=18)
        j_label = Text("j_new", color=GREEN, font_size=18)
        
        i_label.move_to(plane.coords_to_point(1.5, -2.5))
        j_label.move_to(plane.coords_to_point(3.5, 0.5))

        self.play(GrowArrow(i_new_arrow), GrowArrow(j_new_arrow), FadeIn(i_label), FadeIn(j_label))
        self.wait(1)

        # Scale basis vectors (x=2, y=3)
        scaled_i_tip = plane.coords_to_point(2, -4)
        scaled_j_tip = plane.coords_to_point(9, 0)
        
        scaled_i_arrow = Arrow(origin, scaled_i_tip, color=RED, buff=0, stroke_width=6)
        scaled_j_arrow = Arrow(origin, scaled_j_tip, color=GREEN, buff=0, stroke_width=6)
        
        self.play(
            Transform(i_new_arrow, scaled_i_arrow),
            Transform(j_new_arrow, scaled_j_arrow),
            i_label.animate.move_to(plane.coords_to_point(2.5, -4.5)),
            j_label.animate.move_to(plane.coords_to_point(9.5, 0.5))
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_L3)
        )

        # Tip-to-tail addition: Move scaled Green vector to tip of scaled Red vector
        # Resultant is [2, -4] + [9, 0] = [11, -4]
        final_tip = plane.coords_to_point(11, -4)
        
        self.play(
            j_new_arrow.animate.move_to(
                (scaled_i_tip + final_tip) / 2
            ),
            FadeOut(j_label),
            FadeOut(i_label)
        )
        
        # Resultant vector in yellow (#FFFF00)
        res_arrow = Arrow(origin, final_tip, color=YELLOW, buff=0, stroke_width=6)
        
        coord_label = Text("(11, -4)", color="#FFFFFF", font_size=22)
        # Issue 33 fix: place at C2 with scale 0.8 to avoid y-axis overlap
        self.place_at_grid(coord_label, 'C2', scale_factor=0.8)

        self.play(GrowArrow(res_arrow))
        self.play(Indicate(res_arrow), FadeIn(coord_label))
        self.wait(2)
