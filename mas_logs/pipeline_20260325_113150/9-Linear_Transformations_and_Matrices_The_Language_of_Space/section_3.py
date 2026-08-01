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
        # 1. Setup layout
        self.setup_layout("The Secret of the Basis Vectors", [
            "Every vector is a combination of our basis vectors.",
            "Simply track where i-hat and j-hat land.",
            "The entire grid warps to follow their new positions.",
            "Scale the new basis vectors by the original coordinates.",
            "This determines the vector's final position on the grid."
        ])

        # 2. Coordinate System setup
        # Issue 31 Fix: self.place_in_area(plane, 'B2', 'F6', scale_factor=0.5)
        plane = NumberPlane(
            x_range=[-2, 8, 1],
            y_range=[-4, 4, 1],
            x_length=5.0,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.5)
        
        origin = plane.get_origin()
        
        # Colors
        color_i = "#FF0000"
        color_j = "#00FF00"
        color_v = "#FFFF00"
        
        # Basis vectors
        i_hat = Arrow(origin, plane.c2p(1, 0), buff=0, color=color_i)
        j_hat = Arrow(origin, plane.c2p(0, 1), buff=0, color=color_j)
        
        # Issue 33 Fix: self.place_at_grid(i_hat_label, 'D4', scale_factor=0.8)
        i_hat_label = Text("i", slant=ITALIC, color=color_i)
        self.place_at_grid(i_hat_label, 'D4', scale_factor=0.8)
        
        j_hat_label = Text("j", slant=ITALIC, color=color_j)
        self.place_at_grid(j_hat_label, 'B3', scale_factor=0.8)

        # Vector v = 1*i + 2*j
        v_vec = Arrow(origin, plane.c2p(1, 2), buff=0, color=color_v)
        v_label = Text("v", slant=ITALIC, color=color_v)
        self.place_at_grid(v_label, 'B4', scale_factor=0.8)
        
        # Component visualizers (1*i and 2*j)
        comp_i = Arrow(origin, plane.c2p(1, 0), buff=0, color=color_i, stroke_width=2)
        comp_j = Arrow(plane.c2p(1, 0), plane.c2p(1, 2), buff=0, color=color_j, stroke_width=2)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(plane))
        self.play(Create(i_hat), Create(j_hat), Write(i_hat_label), Write(j_hat_label))
        self.wait(0.5)
        self.play(Create(comp_i), Create(comp_j))
        self.play(Create(v_vec), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        target_i_tip = plane.c2p(3, -2)
        target_j_tip = plane.c2p(2, 1)
        
        self.play(
            i_hat.animate.put_start_and_end_on(origin, target_i_tip),
            j_hat.animate.put_start_and_end_on(origin, target_j_tip),
            i_hat_label.animate.move_to(self.grid['E4']),
            j_hat_label.animate.move_to(self.grid['C4']),
            FadeOut(v_vec), FadeOut(v_label), FadeOut(comp_i), FadeOut(comp_j)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Matrix transformation
        matrix = [[3, 2], [-2, 1]]
        self.play(
            plane.animate.apply_matrix(matrix),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        v_final_tip = target_i_tip + 2 * (target_j_tip - origin)
        
        ghost_comp_i = Arrow(origin, target_i_tip, buff=0, color=color_i, stroke_opacity=0.6)
        ghost_comp_j = Arrow(target_i_tip, v_final_tip, buff=0, color=color_j, stroke_opacity=0.6)
        
        self.play(Create(ghost_comp_i))
        self.play(Create(ghost_comp_j))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        v_transformed = Arrow(origin, v_final_tip, buff=0, color=color_v)
        v_new_label = Text("v_new", slant=ITALIC, color=color_v)
        
        # Issue 32 Fix: self.place_at_grid(v_new_label, 'E6', scale_factor=0.8)
        self.place_at_grid(v_new_label, 'E6', scale_factor=0.8)
        
        self.play(Create(v_transformed), Write(v_new_label))
        self.wait(2)
