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
        lecture_lines = ["Roots help us find the base.", "Think of roots as undoing exponents.", "If two cubed is eight, cube-root eight is two."]
        self.setup_layout("The First Twist: Roots as Inverse Powers", lecture_lines)
        
        # Assets
        cube_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        
        # === Animation for Lecture Line 1 ===
        # Roots help us find the base.
        eq1 = MathTex("y^{1/n} = x", color=WHITE)
        self.place_at_grid(eq1, 'B2', scale_factor=0.7)
        self.place_at_grid(cube_asset, 'B3', scale_factor=0.5)
        
        self.play(Write(eq1), FadeIn(cube_asset))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Think of roots as undoing exponents.
        eq2 = MathTex("x^n = y", color=YELLOW)
        self.place_at_grid(eq2, 'C2', scale_factor=0.8)
        
        # Morph eq1 (now transformed or replaced) to eq2
        self.play(Transform(eq1, eq2))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # If two cubed is eight, cube-root eight is two.
        eq3 = MathTex("2^3 = 8", "\\rightarrow", "8^{1/3} = 2", color=GREEN)
        self.place_in_area(eq3, 'D2', 'D5', scale_factor=0.9)
        
        # Swap animation representation
        self.play(ReplacementTransform(eq1, eq3))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
