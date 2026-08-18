from manim import *

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
        self.setup_layout("Prerequisite Review: Basis as Building Blocks", 
                          ["A basis is our building block.", 
                           "Vectors must be linearly independent.", 
                           "Standard basis uses cardinal directions."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # Standard basis vectors
        i_vec = Arrow(ORIGIN, RIGHT * 1.5, color=WHITE, buff=0)
        j_vec = Arrow(ORIGIN, UP * 1.5, color=WHITE, buff=0)
        
        i_label = MathTex(r"\\mathbf{i}", color=WHITE).scale(0.7)
        j_label = MathTex(r"\\mathbf{j}", color=WHITE).scale(0.7)
        
        # Applying requested grid fixes
        self.place_at_grid(i_vec, 'E5', scale_factor=0.9)
        self.place_at_grid(j_vec, 'B3', scale_factor=0.9)
        i_label.next_to(i_vec.get_end(), RIGHT, buff=0.1)
        j_label.next_to(j_vec.get_end(), UP, buff=0.1)
        
        self.play(Create(i_vec), Create(j_vec), Write(i_label), Write(j_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg
        grid_visual = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_visual, 'A3', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(grid_visual), run_time=1.5)
        self.bring_to_front(i_vec, j_vec, i_label, j_label)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        b1_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        b2_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        # Requested fixes for basis vectors
        self.place_at_grid(b1_asset, 'D5', scale_factor=0.6)
        self.place_at_grid(b2_asset, 'D2', scale_factor=0.6)
        
        b1_label = MathTex(r"\\mathbf{b}_1", color=GREEN).scale(0.7)
        b2_label = MathTex(r"\\mathbf{b}_2", color=GREEN).scale(0.7)
        b1_label.next_to(b1_asset, RIGHT, buff=0.1)
        b2_label.next_to(b2_asset, UP, buff=0.1)
        
        self.play(FadeIn(b1_asset), FadeIn(b2_asset), Write(b1_label), Write(b2_label))
        self.wait(2)
