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
        self.setup_layout("Prerequisite: The Area Interpretation of Determinants", 
                          ["The determinant measures the signed area of a parallelogram.", 
                           "Unit squares transform into areas scaled by the determinant.", 
                           "This scaling factor is our fundamental geometric tool."])
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: SVGAsset is generic icon here, used as an overlay indicator.
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")

        # Vectors
        v_i = Arrow(ORIGIN, RIGHT * 1.5, color="#FF6347")
        v_j = Arrow(ORIGIN, UP * 1.5, color="#4682B4")
        
        # Parallelogram
        parallelogram = Polygon(ORIGIN, v_i.get_end(), v_i.get_end() + v_j.get_end(), v_j.get_end(), 
                                fill_opacity=0.3, fill_color="#FFFACD", stroke_color=WHITE)
        parallelogram_group = VGroup(v_i, v_j, parallelogram, icon)
        
        # Labels
        det_label = MathTex(r"\\det(A)", color=WHITE)
        formula_box = VGroup(det_label, icon)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(parallelogram_group, 'C3', 'E5', scale_factor=0.8)
        self.play(FadeIn(v_i), FadeIn(v_j), FadeIn(parallelogram))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(det_label, 'A6', scale_factor=0.9)
        self.play(FadeIn(det_label))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        # Transform animation
        new_v_i = Arrow(ORIGIN, RIGHT * 2 + UP * 0.5, color="#FF6347")
        new_v_j = Arrow(ORIGIN, LEFT * 0.5 + UP * 2, color="#4682B4")
        new_para = Polygon(ORIGIN, new_v_i.get_end(), new_v_i.get_end() + new_v_j.get_end(), new_v_j.get_end(), 
                           fill_opacity=0.3, fill_color="#FFFACD", stroke_color=WHITE)
        
        self.place_in_area(formula_box, 'F3', 'F5', scale_factor=0.7)
        self.play(ReplacementTransform(v_i, new_v_i), 
                  ReplacementTransform(v_j, new_v_j), 
                  ReplacementTransform(parallelogram, new_para))
        self.lecture[2].set_color("#FFD700")
        self.wait(1)
