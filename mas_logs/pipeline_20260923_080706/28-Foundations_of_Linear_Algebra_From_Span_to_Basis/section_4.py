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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["A basis is a minimal spanning set.", "Basis vectors are always linearly independent.", "Basis provides an efficient coordinate system."]
        self.setup_layout("Defining the Basis", lecture_lines)
        
        # Define mobjects
        i_vec = Vector(RIGHT, color="#00FFFF")
        j_vec = Vector(UP, color="#00FFFF")
        label_i = MathTex(r"\\vec{i}", color="#00FFFF")
        label_j = MathTex(r"\\vec{j}", color="#00FFFF")
        box = Square(side_length=1.0, color="#FFFFFF", fill_opacity=0.2)
        
        # Place according to critique
        basis = VGroup(i_vec, j_vec)
        self.place_at_grid(basis, 'C4', scale_factor=0.9)
        self.place_at_grid(label_i, 'C5', scale_factor=0.7)
        self.place_at_grid(label_j, 'B4', scale_factor=0.7)
        self.place_in_area(box, 'C4', 'D5', scale_factor=0.8)

        # Asset loading (dummy icons for [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg])
        # Since the assets are '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg', we use placeholders
        asset_icon_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.3)
        asset_icon_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.3)
        self.place_at_grid(asset_icon_1, 'A6', scale_factor=1.0)
        self.place_at_grid(asset_icon_2, 'F6', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(i_vec), Create(j_vec), Write(label_i), Write(label_j), FadeIn(asset_icon_1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.play(Create(box))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(box.animate.set_color("#FFFF00").scale(1.2), FadeIn(asset_icon_2))
        self.wait(2)
