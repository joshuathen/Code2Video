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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Two blocks slide on a frictionless surface.",
            "Massive block A approaches stationary block B.",
            "Block B sits before a fixed wall.",
            "A tiny hamster analogy illustrates the masses.",
            "The system is a simple physical paradox."
        ]
        self.setup_layout("The Paradoxical Setup", lecture_lines)
        
        # Assets
        block_a_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        block_b_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        wall_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        hamster_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hamster.svg"
        
        # Mobjects
        block_a = SVGMobject(block_a_asset, color=WHITE).scale(0.5)
        block_b = SVGMobject(block_b_asset, color=WHITE).scale(0.3)
        wall = SVGMobject(wall_asset, color=RED).scale(0.8)
        hamster = SVGMobject(hamster_asset, color=GREEN).scale(0.5)
        
        # Labels
        label_a = Text("m_A", font_size=20)
        label_b = Text("m_B", font_size=20)

        # === Animation for Lecture Line 1 ===
        # Two blocks slide on a frictionless surface.
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(block_a, 'C2', 'D3', 1.0)
        self.place_in_area(block_b, 'C4', 'D4', 1.0)
        self.play(FadeIn(block_a), FadeIn(block_b))

        # === Animation for Lecture Line 2 ===
        # Massive block A approaches stationary block B.
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(block_a.animate.shift(RIGHT * 1.0))

        # === Animation for Lecture Line 3 ===
        # Block B sits before a fixed wall.
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.place_at_grid(wall, 'D5', 1.0)
        self.play(Create(wall))

        # === Animation for Lecture Line 4 ===
        # A tiny hamster analogy illustrates the masses.
        self.play(self.lecture[3].animate.set_color(BLUE))
        self.place_in_area(hamster, 'B3', 'D4', 0.9)
        self.play(FadeIn(hamster))

        # === Animation for Lecture Line 5 ===
        # The system is a simple physical paradox.
        self.play(self.lecture[4].animate.set_color(BLUE))
        self.play(FadeOut(block_a), FadeOut(block_b), FadeOut(wall), FadeOut(hamster))
