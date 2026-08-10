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
        lecture_lines = ["Two blocks on a frictionless surface.", "Small block sits between wall and large block.", "Elastic collisions conserve energy and momentum.", "Collisions begin as the large block approaches.", "This creates a rapid ping-pong effect."]
        self.setup_layout("The Setup: A Curious Mechanical System", lecture_lines)
        
        # Assets
        block_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        wall_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        
        block1 = SVGMobject(block_svg)
        block2 = SVGMobject(block_svg)
        wall = SVGMobject(wall_svg)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(block1, "D2", scale_factor=0.6)
        self.place_at_grid(block2, "D4", scale_factor=0.6)
        self.play(FadeIn(block1), FadeIn(block2))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(wall, "D1", scale_factor=1.0)
        self.add(wall)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        energy_label = MathTex("E_{total}", "=", "const").scale(0.8)
        self.place_in_area(energy_label, "A3", "A5", scale_factor=0.8)
        self.play(Write(energy_label))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(block2.animate.move_to(block1.get_right() + RIGHT * 0.2))
        collision_pt = Dot(point=block1.get_right(), color=RED)
        self.play(Flash(collision_pt))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(FadeOut(block1), FadeOut(block2), FadeOut(energy_label), FadeOut(wall))
