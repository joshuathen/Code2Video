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
        self.setup_layout("Prerequisite Knowledge: Conservation Laws", 
                          ["Collisions obey momentum and energy conservation.", 
                           "These laws govern how velocity changes.", 
                           "Blocks eventually move away from the wall."])
        
        # === Animation for Lecture Line 1 ===
        # Show momentum equation p=mv on screen in #00FF00
        p_eq = MathTex(r"p = mv", color="#00FF00")
        self.place_in_area(p_eq, 'C2', 'C3', scale_factor=0.9)
        self.play(Write(p_eq))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 2 ===
        # Show energy equation KE=0.5mv^2 in #FF00FF
        ke_eq = MathTex(r"KE = \frac{1}{2}mv^2", color="#FF00FF")
        self.place_in_area(ke_eq, 'C5', 'C6', scale_factor=0.9)
        self.play(Write(ke_eq))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        # Assets: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        # Assets: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        self.place_at_grid(wall, 'D2', scale_factor=0.5)
        self.place_at_grid(blocks, 'D3', scale_factor=0.8)
        
        self.play(FadeIn(wall), FadeIn(blocks))
        self.play(
            blocks.animate.shift(RIGHT * 1.0)
        )
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
