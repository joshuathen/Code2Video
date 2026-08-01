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
        # 1. Setup Layout
        title_text = "Mathematical Symmetry: Swapping V and F"
        lecture_lines = [
            "A cube has eight vertices and six faces.",
            "Its dual swaps these counts: six vertices, eight faces.",
            "Both graphs share exactly twelve edges."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define Colors
        color_v = YELLOW
        color_f = GREEN
        color_e = BLUE

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(color_v))
        
        # Cube Graph Asset
        cube_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cube.svg", color=WHITE)
        self.place_in_area(cube_icon, "B1", "E3", scale_factor=1.8)
        
        # Stats labels for primal graph - Using Text instead of MathTex to avoid LaTeX dependency error
        v_text = Text("V = 8", color=color_v, font_size=24)
        e_text = Text("E = 12", color=WHITE, font_size=24)
        f_text = Text("F = 6", color=color_f, font_size=24)
        
        self.place_at_grid(v_text, "B5", scale_factor=1.0)
        self.place_at_grid(e_text, "C5", scale_factor=1.0)
        self.place_at_grid(f_text, "D5", scale_factor=1.0)
        
        self.play(DrawBorderThenFill(cube_icon))
        self.play(Write(v_text), Write(e_text), Write(f_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_f)
        )
        
        # Dual stats labels (Target positions) - Using Text to avoid LaTeX dependency error
        v_star = Text("V* = 6", color=color_v, font_size=24)
        f_star = Text("F* = 8", color=color_f, font_size=24)
        
        self.place_at_grid(v_star, "B6", scale_factor=1.0)
        self.place_at_grid(f_star, "D6", scale_factor=1.0)
        
        # Swapping arrows to show the duality mapping
        arrow_v_to_fstar = CurvedArrow(v_text.get_right(), f_star.get_left(), angle=-PI/4, color=WHITE, stroke_width=2)
        arrow_f_to_vstar = CurvedArrow(f_text.get_right(), v_star.get_left(), angle=PI/4, color=WHITE, stroke_width=2)
        
        self.play(Create(arrow_v_to_fstar), Create(arrow_f_to_vstar))
        self.play(
            TransformFromCopy(v_text, f_star),
            TransformFromCopy(f_text, v_star)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_e)
        )
        
        # Show E remains same - Using Text to avoid LaTeX dependency error
        e_star = Text("E* = 12", color=color_e, font_size=24)
        self.place_at_grid(e_star, "C6", scale_factor=1.0)
        
        self.play(e_text.animate.set_color(color_e))
        self.play(TransformFromCopy(e_text, e_star))
        
        # Visual highlight of the edge count equality
        self.play(Indicate(e_text), Indicate(e_star))
        self.wait(2)
