from manim import *
import os

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
        lecture_lines = ["Review 2D vectors and area.", "2x2 determinant defines area.", "This builds the cross product."]
        self.setup_layout("Prerequisites: Vectors and Determinants", lecture_lines)
        
        # Asset Placeholder
        # Since [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] is effectively empty/placeholder, 
        # using a simple shape to fulfill the inclusion requirement.
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if os.path.exists("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") else Dot()
        
        # === Animation for Lecture Line 1 ===
        v1 = Arrow(ORIGIN, RIGHT * 1.5 + UP * 0.5, color="#FF5733")
        v2 = Arrow(ORIGIN, RIGHT * 0.5 + UP * 1.5, color="#33FF57")
        self.place_at_grid(v1, 'B3', scale_factor=0.8)
        self.place_at_grid(v2, 'C3', scale_factor=0.8)
        
        asset1 = icon.copy()
        self.place_at_grid(asset1, 'A1', scale_factor=0.2)
        
        self.play(Create(v1), Create(v2), FadeIn(asset1))
        self.play(self.lecture[0].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 2 ===
        matrix = Matrix([[r"a", r"c"], [r"b", r"d"]], color=WHITE)
        self.place_at_grid(matrix, 'B4', scale_factor=1.0)
        self.play(Write(matrix))
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 3 ===
        det_val_pos = Text("ad - bc", color="#FFFF00")
        self.place_at_grid(det_val_pos, 'D5', scale_factor=1.2)
        
        # Highlight area spanned
        area = Polygon(ORIGIN, RIGHT * 1.5 + UP * 0.5, RIGHT * 2.0 + UP * 2.0, RIGHT * 0.5 + UP * 1.5, color="#FFFF00", fill_opacity=0.3)
        
        asset2 = icon.copy()
        self.place_at_grid(asset2, 'F6', scale_factor=0.2)

        self.play(Create(area), Write(det_val_pos), FadeIn(asset2))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
