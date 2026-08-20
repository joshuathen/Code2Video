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
        lecture_lines = [
            "Energy and momentum remain constant throughout collisions.",
            "These laws dictate post-collision velocity changes.",
            "Collisions act as transformations of velocity vectors."
        ]
        self.setup_layout("Prerequisite Physics: Conservation Laws", lecture_lines)
        
        # Elements
        energy_circle = Circle(radius=0.8, color="#FFFF00").set_fill(opacity=0.3)
        self.place_at_grid(energy_circle, "B5", scale_factor=0.8)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        momentum_vec = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#FF8000")
        momentum_group = VGroup(block_icon, momentum_vec).arrange(RIGHT)
        self.place_at_grid(momentum_group, "D4", scale_factor=0.9)
        
        cons_eq = MathTex(r"m_1 v_1 + m_2 v_2 = \text{const}", color=WHITE)
        self.place_in_area(cons_eq, "E3", "E6", scale_factor=0.85)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(energy_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8000"))
        self.play(FadeIn(momentum_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(Write(cons_eq))
        self.wait(1)
