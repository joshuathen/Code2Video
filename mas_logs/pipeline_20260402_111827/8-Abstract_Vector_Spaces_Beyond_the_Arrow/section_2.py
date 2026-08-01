from manim import *
import numpy as np
import os

# Pre-create the directory to handle the race condition causing FileExistsError in Manim's Text mobject
try:
    os.makedirs(os.path.join("media", "texts"), exist_ok=True)
except Exception:
    pass

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
        # Setup layout
        lines = [
            "Reviewing Euclidean space: addition and scalar multiplication.",
            "The parallelogram law visualizes vector addition in 2D.",
            "Scalars stretch or shrink vectors along their direction."
        ]
        self.setup_layout("Prerequisite: The Standard Euclidean Space", lines)

        # Coordinate Plane for Visuals
        # Placed in the area A1 to F6
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, "A1", "F6")
        self.add(plane)

        # Coordinate conversion helper
        def p_to_s(coords):
            return plane.coords_to_point(*coords)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        u_vec = Vector([2, 1], color="#00FF00")
        u_vec.move_to(p_to_s([0, 0]), aligned_edge=DL)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        u_label = Text("u", color="#00FF00", slant=ITALIC).scale(0.8)
        u_label.next_to(u_vec.get_end(), UR, buff=0.1)

        v_vec = Vector([1, 2], color="#0000FF")
        v_vec.move_to(p_to_s([0, 0]), aligned_edge=DL)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        v_label = Text("v", color="#0000FF", slant=ITALIC).scale(0.8)
        v_label.next_to(v_vec.get_end(), UL, buff=0.1)

        # Replaced MathTex with Text to avoid LaTeX dependency error
        op_text_1 = Text("u + v", color=WHITE, slant=ITALIC)
        op_text_2 = Text("c · u", color=WHITE, slant=ITALIC)
        ops_group = VGroup(op_text_1, op_text_2).arrange(RIGHT, buff=1.0)
        self.place_at_grid(ops_group, "A3", scale_factor=0.8)

        self.play(Create(u_vec), Write(u_label), Create(v_vec), Write(v_label))
        self.play(Write(ops_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        # Parallel dashed lines
        p_line1 = DashedLine(p_to_s([2, 1]), p_to_s([3, 3]), color="#AAAAAA")
        p_line2 = DashedLine(p_to_s([1, 2]), p_to_s([3, 3]), color="#AAAAAA")
        
        # Resultant
        res_vec = Vector([3, 3], color="#FF0000")
        res_vec.move_to(p_to_s([0, 0]), aligned_edge=DL)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        res_label = Text("u + v", color="#FF0000", slant=ITALIC).scale(0.8)
        res_label.next_to(res_vec.get_end(), UR, buff=0.1)

        self.play(Create(p_line1), Create(p_line2))
        self.play(Create(res_vec), Write(res_label))
        self.wait(2)

        # Clear for scaling animation
        self.play(
            FadeOut(u_vec, u_label, v_vec, v_label, p_line1, p_line2, res_vec, res_label, ops_group)
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))

        w_coords = np.array([1, 0.5, 0])
        
        # We define a vector
        w_vec = Vector(w_coords[:2], color="#FFFF00")
        w_vec.move_to(p_to_s([0, 0]), aligned_edge=DL)
        
        # Replaced MathTex with Text to avoid LaTeX dependency error
        w_label = Text("w", color="#FFFF00", slant=ITALIC).scale(0.8)
        w_label.next_to(w_vec.get_end(), DR, buff=0.1)
        
        # Replaced MathTex with Text to avoid LaTeX dependency error
        scaling_label = Text("c · w", color="#FFFF00", slant=ITALIC).scale(0.8)
        self.place_at_grid(scaling_label, "A3")

        self.play(Create(w_vec), Write(w_label), Write(scaling_label))
        self.wait(0.5)

        # Scale to 2w
        self.play(
            w_vec.animate.scale(2, about_point=p_to_s([0, 0])),
            w_label.animate.move_to(p_to_s([2.2, 1.2])),
            run_time=1.5
        )
        self.wait(0.5)

        # Scale to 0.5w (relative to original 1.0, so scale by 0.25 from current size 2.0)
        self.play(
            w_vec.animate.scale(0.25, about_point=p_to_s([0, 0])),
            w_label.animate.move_to(p_to_s([0.7, 0.4])),
            run_time=1.5
        )
        self.wait(2)
