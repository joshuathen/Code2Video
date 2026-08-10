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
        lecture_lines = ["Mass ratios $100^N$ link to $\\pi$.", "Total collisions equal $\\pi$ digits.", "$N=0$ yields three collisions.", "$N=1$ yields thirty-one collisions.", "$N=2$ yields three hundred fourteen."]
        self.setup_layout("Unveiling Pi", lecture_lines)
        
        # Load Assets
        particles = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particles.svg")
        weights = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weights.svg")
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        pi_val = MathTex(r"\\pi = 3.14159...", font_size=48, color=WHITE)
        mass_ratio = MathTex(r"100^N = m_A/m_B", font_size=36, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Display Pi at B4-B6, Mass Ratio near it with weights
        self.place_in_area(pi_val, 'B4', 'B6', scale_factor=0.6)
        self.place_at_grid(particles, 'B3', scale_factor=0.5)
        self.place_at_grid(weights, 'C5', scale_factor=0.5)
        self.place_at_grid(mass_ratio, 'C6', scale_factor=0.6)
        
        self.play(FadeIn(pi_val), FadeIn(particles), FadeIn(weights), FadeIn(mass_ratio))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        three = Text("3", color="#FFD700", font_size=48)
        self.place_at_grid(three, 'E3', scale_factor=0.9)
        self.place_at_grid(blocks, 'E2', scale_factor=0.5)
        self.play(FadeIn(three), FadeIn(blocks))
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        three_one = Text("31", color="#FFD700", font_size=48)
        self.place_at_grid(three_one, 'E3', scale_factor=0.9)
        self.play(ReplacementTransform(three, three_one))
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        three_one_four = Text("314", color="#FFD700", font_size=48)
        self.place_at_grid(three_one_four, 'E3', scale_factor=0.9)
        self.play(ReplacementTransform(three_one, three_one_four))
        self.lecture[4].set_color("#FFD700")
        self.wait(2)
