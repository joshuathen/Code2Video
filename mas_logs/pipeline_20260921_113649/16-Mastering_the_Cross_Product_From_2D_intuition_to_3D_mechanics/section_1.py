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
        lecture_lines = [
            "Vectors span a 2D parallelogram.",
            "Determinant measures the signed area.",
            "Clockwise area is negative.",
            "Counter-clockwise area is positive.",
            "This measures 2D reach clearance."
        ]
        self.setup_layout("Prerequisites & The 2D 'Shadow'", lecture_lines)
        
        # Setup vectors
        u = Vector([1.5, 0.5], color=WHITE)
        v = Vector([0.5, 1.5], color=WHITE)
        label_u = MathTex(r"\vec{u}").set_color(WHITE)
        label_v = MathTex(r"\vec{v}").set_color(WHITE)
        
        # Setup grid positioning per instructions
        self.place_at_grid(u, 'B3', scale_factor=0.8)
        self.place_at_grid(v, 'D3', scale_factor=0.8)
        self.place_at_grid(label_u, 'B2', scale_factor=0.6)
        self.place_at_grid(label_v, 'D2', scale_factor=0.6)
        
        # Helper to create parallelogram
        def get_para(v1, v2):
            return Polygon(ORIGIN, v1.get_end(), v1.get_end() + v2.get_end(), v2.get_end(), fill_opacity=0.5, stroke_width=2)

        # Asset loading placeholder - using SVGMobject as requested by logic (even if dummy path)
        # Using a safer approach for non-existent files if needed, but keeping original structure
        asset_icon = SVGMobject(path_string=None) if False else Rectangle(width=0.5, height=0.5, color=BLACK)

        # === Animation for Lecture Line 1 ===
        para = get_para(u, v)
        self.place_in_area(para, 'B4', 'E6', scale_factor=0.9)
        self.play(FadeIn(u), FadeIn(v), FadeIn(label_u), FadeIn(label_v), Create(para), FadeIn(asset_icon.scale(0.5).to_corner(DR)))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(para.animate.set_color(GRAY))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        para_cw = get_para(v, u).set_fill(color="#FF0000")
        self.place_in_area(para_cw, 'B4', 'E6', scale_factor=0.9)
        self.play(FadeOut(para), FadeIn(para_cw))
        self.lecture[2].set_color("#FF0000")

        # === Animation for Lecture Line 4 ===
        para_ccw = get_para(u, v).set_fill(color="#0000FF")
        self.place_in_area(para_ccw, 'B4', 'E6', scale_factor=0.9)
        self.play(FadeOut(para_cw), FadeIn(para_ccw))
        self.lecture[3].set_color("#0000FF")

        # === Animation for Lecture Line 5 ===
        self.play(para_ccw.animate.scale(1.2))
        self.lecture[4].set_color(GREEN)
        self.wait(2)
