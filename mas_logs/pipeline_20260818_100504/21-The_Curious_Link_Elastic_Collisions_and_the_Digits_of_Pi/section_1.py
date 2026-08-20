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
        self.setup_layout("The Setup: The Idealized Collision System", [
            "Consider two blocks on a frictionless surface.",
            "Small mass m, large mass M.",
            "Perfectly elastic collisions occur.",
            "Momentum and kinetic energy are conserved.",
            "This sets our physical system."
        ])
        
        # Assets
        block_m = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#FF5733")
        block_M = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#33FF57")
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(block_m, 'E2', scale_factor=0.6)
        self.place_at_grid(block_M, 'E5', scale_factor=1.5)
        self.play(FadeIn(block_m), FadeIn(block_M))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        label_m = Text("m", font_size=24, color="#FF5733").next_to(block_m, UP)
        label_M = Text("M", font_size=24, color="#33FF57").next_to(block_M, UP)
        self.play(Write(label_m), Write(label_M))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        velocity = Arrow(start=LEFT, end=RIGHT, color=YELLOW, buff=0.1).next_to(block_m, LEFT)
        self.play(Create(velocity))
        self.play(block_m.animate.shift(RIGHT * 1.5))
        self.play(FadeOut(velocity))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        pulse = Circle(radius=0.8, color="#FFFF00", stroke_width=4).move_to(block_M.get_center())
        self.play(Create(pulse))
        self.play(FadeOut(pulse))
        self.lecture[3].set_color("#FFFF00")

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(wall, 'E1', scale_factor=1.0)
        self.play(FadeIn(wall))
        self.play(block_m.animate.shift(LEFT * 1.5))
        self.lecture[4].set_color(GRAY)
