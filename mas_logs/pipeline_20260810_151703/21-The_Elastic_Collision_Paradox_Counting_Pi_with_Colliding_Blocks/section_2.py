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
        self.setup_layout("Prerequisite Knowledge: Conservation Laws", [
            "Elastic collisions preserve total energy.",
            "Both kinetic energy and momentum are conserved.",
            "Blocks trade velocities based on mass ratios."
        ])
        
        # Load assets
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        # === Animation for Lecture Line 1 ===
        energy_formula = MathTex(r"E = \frac{1}{2}mv^2", color="#ADD8E6")
        energy_group = VGroup(block_icon.copy(), energy_formula).arrange(RIGHT, buff=0.2)
        self.place_in_area(energy_group, 'A3', 'B5', scale_factor=1.0)
        self.play(FadeIn(energy_group))
        self.lecture[0].set_color("#ADD8E6")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        momentum_formula = MathTex(r"p = mv", color="#90EE90")
        self.place_in_area(momentum_formula, 'D3', 'E5', scale_factor=1.2)
        self.play(FadeIn(momentum_formula))
        self.lecture[1].set_color("#90EE90")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        conservation_group = VGroup(energy_group, momentum_formula)
        self.play(
            Indicate(conservation_group, color="#FFFF00"),
            run_time=2
        )
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
