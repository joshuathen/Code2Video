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
        self.setup_layout("The Transition Matrix (The Bridge)", [
            "We need a translator between bases.",
            "The change of basis matrix bridges systems.",
            "Matrix multiplication maps coordinates across perspectives."
        ])
        
        # Elements
        basis_a = VGroup(
            Text("Basis A", color="#3498DB").scale(0.6),
            Square(color="#3498DB", side_length=1.5)
        ).arrange(DOWN)
        
        basis_b = VGroup(
            Text("Basis B", color="#E74C3C").scale(0.6),
            Square(color="#E74C3C", side_length=1.5)
        ).arrange(DOWN)
        
        basis_group = VGroup(basis_a, basis_b).arrange(RIGHT, buff=0.5)
        
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE)
        p_label = MathTex("P", color=YELLOW)
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        self.place_at_grid(bridge_icon, "C3", scale_factor=0.5)
        self.play(FadeIn(bridge_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        self.place_in_area(basis_group, "B2", "B6", scale_factor=0.7)
        self.play(Create(basis_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F1C40F"))
        self.place_at_grid(arrow, "B4", scale_factor=0.8)
        self.place_at_grid(p_label, "C4", scale_factor=0.6)
        p_label.next_to(arrow, UP)
        self.play(GrowArrow(arrow), Write(p_label))
        self.wait(2)
