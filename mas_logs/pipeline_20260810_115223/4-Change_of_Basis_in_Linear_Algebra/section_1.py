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
        self.setup_layout("Prerequisites & Intuition", [
            "A basis is like a coordinate system.",
            "Basis vectors define our grid.",
            "Any vector is a linear combination.",
            "Change perspectives, keep the vector.",
            "How do we describe this change?"
        ])
        
        # Assets
        grid_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEFA")
        dot = Dot(color="#87CEFA")
        self.place_at_grid(dot, 'B2', scale_factor=0.5)
        self.play(FadeIn(dot))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        basis_group = VGroup(grid_img)
        i_vec = Vector(RIGHT, color=WHITE).next_to(ORIGIN, RIGHT)
        j_vec = Vector(UP, color=WHITE).next_to(ORIGIN, UP)
        basis_group.add(i_vec, j_vec, Text("i", font_size=20), Text("j", font_size=20))
        self.place_in_area(basis_group, 'B2', 'C4', scale_factor=0.8)
        self.play(FadeIn(basis_group))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#90EE90")
        vec_v = Vector([1, 1], color=YELLOW)
        self.place_at_grid(vec_v, 'C4', scale_factor=0.6)
        self.play(FadeIn(vec_v))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        new_basis = VGroup(
            Vector([1, 1], color="#90EE90"),
            Vector([-1, 1], color="#90EE90")
        )
        self.place_at_grid(new_basis, 'D4', scale_factor=0.6)
        self.play(FadeIn(new_basis))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF6347")
        self.play(FadeOut(self.lecture), FadeOut(self.title), FadeOut(dot), FadeOut(basis_group), FadeOut(vec_v), FadeOut(new_basis))
