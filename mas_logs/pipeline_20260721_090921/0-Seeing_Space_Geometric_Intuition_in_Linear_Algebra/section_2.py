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
        # Setup data
        title_str = "The DNA of Space: Basis Vectors"
        lecture_lines_str = [
            "Every 2D vector is built from two fundamental pieces.",
            "i-hat is a unit vector along the x-axis.",
            "j-hat is a unit vector along the y-axis.",
            "Combine these scaled pieces to reach any point.",
            "These basic building blocks are called basis vectors."
        ]
        
        # Colors
        COLOR_I = "#FF0000"
        COLOR_J = "#00FF00"
        COLOR_V = "#FFFFFF"
        HIGHLIGHT = YELLOW
        
        self.setup_layout(title_str, lecture_lines_str)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))
        
        # Coordinate Plane
        # Fix: Issue 32 - Occupy larger area B1-F6
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-4, 3, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'B1', 'F6', scale_factor=1.0)
        
        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_I)
        )
        
        origin = plane.coords_to_point(0, 0)
        i_tip = plane.coords_to_point(1, 0)
        i_hat = Arrow(origin, i_tip, buff=0, color=COLOR_I, stroke_width=4)
        i_hat_label = Text("î", color=COLOR_I, font_size=20)
        i_hat_label.next_to(i_hat, UP, buff=0.1)
        
        self.play(GrowArrow(i_hat), Write(i_hat_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_J)
        )
        
        j_tip = plane.coords_to_point(0, 1)
        j_hat = Arrow(origin, j_tip, buff=0, color=COLOR_J, stroke_width=4)
        j_hat_label = Text("ĵ", color=COLOR_J, font_size=20)
        j_hat_label.next_to(j_hat, LEFT, buff=0.1)
        
        self.play(GrowArrow(j_hat), Write(j_hat_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_V)
        )
        
        # Scaling i-hat by 3
        i_scaled_tip = plane.coords_to_point(3, 0)
        i_scaled = Arrow(origin, i_scaled_tip, buff=0, color=COLOR_I, stroke_width=4)
        i_scaled_label = Text("3î", color=COLOR_I, font_size=20)
        i_scaled_label.next_to(i_scaled, UP, buff=0.1)
        
        # Scaling j-hat by -2
        j_scaled_tip = plane.coords_to_point(0, -2)
        j_scaled = Arrow(origin, j_scaled_tip, buff=0, color=COLOR_J, stroke_width=4)
        j_scaled_label = Text("-2ĵ", color=COLOR_J, font_size=20)
        j_scaled_label.next_to(j_scaled, LEFT, buff=0.1)
        
        self.play(
            ReplacementTransform(i_hat, i_scaled),
            ReplacementTransform(i_hat_label, i_scaled_label),
            ReplacementTransform(j_hat, j_scaled),
            ReplacementTransform(j_hat_label, j_scaled_label)
        )
        self.wait(0.5)
        
        # Move j_scaled to tip of i_scaled
        shift_vector = i_scaled_tip - origin
        self.play(
            j_scaled.animate.shift(shift_vector),
            j_scaled_label.animate.shift(shift_vector)
        )
        
        # Resultant vector [3, -2]
        res_tip = plane.coords_to_point(3, -2)
        v_res = Arrow(origin, res_tip, buff=0, color=COLOR_V, stroke_width=6)
        v_label = Text("v = 3î - 2ĵ", color=COLOR_V, font_size=20)
        v_label.next_to(v_res.get_end(), DOWN, buff=0.2)
        
        self.play(Create(v_res), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT)
        )
        
        # Fix: Issue 31 - Positioning of basis_tag
        basis_tag = Text("Basis Vectors", font_size=22, color=HIGHLIGHT)
        self.place_in_area(basis_tag, 'A3', 'A5', scale_factor=0.8)
        
        self.play(
            Indicate(i_scaled, color=COLOR_I),
            Indicate(j_scaled, color=COLOR_J),
            Write(basis_tag)
        )
        
        self.wait(3)
