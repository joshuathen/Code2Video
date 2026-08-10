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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Cramer's Rule uses a ratio of areas.",
            "Replace one column with target vector b.",
            "The new area is scaled by x.",
            "x equals the ratio of these areas.",
            "This logic defines Cramer's Rule geometrically."
        ]
        self.setup_layout("The Substitution Intuition", lecture_lines)
        
        # Elements
        original_para = Polygon([0, 0, 0], [2, 0, 0], [2.5, 1.5, 0], [0.5, 1.5, 0], color=BLUE)
        b_vec = Vector([0.5, 1.5], color=YELLOW)
        new_para = Polygon([0, 0, 0], [3, 0, 0], [3.5, 2.5, 0], [0.5, 2.5, 0], color=RED)
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(original_para, 'B2', 'C4', scale_factor=0.5)
        self.play(Create(original_para))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(b_vec, 'C6', scale_factor=0.6)
        self.play(GrowArrow(b_vec))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.place_in_area(new_para, 'B2', 'C4', scale_factor=0.5)
        self.place_at_grid(asset_icon, 'F6', scale_factor=0.5)
        self.play(ReplacementTransform(original_para.copy(), new_para), FadeIn(asset_icon))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GREEN))
        ratio_eq = MathTex("x = \\frac{Area(New)}{Area(Original)}", font_size=32)
        self.place_at_grid(ratio_eq, 'D3', scale_factor=0.8)
        self.play(Write(ratio_eq))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        vg = VGroup(original_para, new_para, b_vec, ratio_eq, asset_icon)
        self.play(vg.animate.set_opacity(0.5))
