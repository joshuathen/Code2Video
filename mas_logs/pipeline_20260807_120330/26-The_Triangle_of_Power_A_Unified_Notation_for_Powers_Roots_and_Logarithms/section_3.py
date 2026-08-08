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
        # Teaching Content
        title = "Introducing the Triangle of Power"
        lines = [
            "Meet the Triangle of Power, a unified geometric map.",
            "Place the Base at the bottom-left vertex.",
            "The Exponent sits naturally at the very top.",
            "Finally, the Result occupies the bottom-right corner.",
            "One shape now replaces all three traditional notations."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_BASE = "#3498DB"
        COLOR_EXP = "#E74C3C"
        COLOR_RES = "#2ECC71"
        
        # Grid positions for triangle vertices for arrow anchors
        pos_base_vertex = self.grid["E2"]
        pos_res_vertex = self.grid["E5"]
        pos_exp_vertex = (self.grid["B3"] + self.grid["B4"]) / 2
        
        # === Animation for Lecture Line 1 ===
        # Meet the Triangle of Power, a unified geometric map.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Replace SVG with native Manim Triangle to avoid FileNotFoundError
        triangle = Triangle().set_color(WHITE)
        self.place_in_area(triangle, "B2", "E5", scale_factor=2.4)
        
        self.play(DrawBorderThenFill(triangle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place the Base at the bottom-left vertex.
        self.play(self.lecture[1].animate.set_color(COLOR_BASE))
        
        # Replace missing SVG with MathTex for "Base"
        base_label = MathTex("\\text{Base}", color=COLOR_BASE)
        self.place_at_grid(base_label, "F2", scale_factor=0.8)
        
        self.play(FadeIn(base_label, shift=UP * 0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The Exponent sits naturally at the very top.
        self.play(self.lecture[2].animate.set_color(COLOR_EXP))
        
        # Exponent label at the top vertex, anchor in area A3-A4
        exp_label = MathTex("\\text{Exponent}", color=COLOR_EXP)
        self.place_in_area(exp_label, "A3", "A4", scale_factor=0.8)
        
        self.play(Write(exp_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Finally, the Result occupies the bottom-right corner.
        self.play(self.lecture[3].animate.set_color(COLOR_RES))
        
        # Result label at the bottom-right vertex (E5), anchor at F5
        res_label = MathTex("\\text{Result}", color=COLOR_RES)
        self.place_at_grid(res_label, "F5", scale_factor=0.8)
        
        self.play(Write(res_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # One shape now replaces all three traditional notations.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Connecting arrows between vertices
        arrow_be = CurvedArrow(pos_base_vertex, pos_exp_vertex, angle=-PI/4, color=WHITE, tip_length=0.2)
        arrow_er = CurvedArrow(pos_exp_vertex, pos_res_vertex, angle=-PI/4, color=WHITE, tip_length=0.2)
        arrow_br = Arrow(pos_base_vertex, pos_res_vertex, color=WHITE, buff=0.2, tip_length=0.2)
        
        self.play(
            Create(arrow_be),
            Create(arrow_er),
            Create(arrow_br),
            run_time=2
        )
        self.wait(3)
