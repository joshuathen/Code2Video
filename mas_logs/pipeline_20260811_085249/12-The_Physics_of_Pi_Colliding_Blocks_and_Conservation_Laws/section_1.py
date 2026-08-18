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
            "Two blocks collide on a frictionless surface.",
            "Tiny block m approaches massive block M.",
            "M is one hundred to the n times m.",
            "They bounce off a wall to calculate pi."
        ]
        self.setup_layout("The Counter-Intuitive Hook", lecture_lines)
        
        # Elements using assets
        block_m = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=RED)
        block_M = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=RED)
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=WHITE)
        pi_symbol = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=YELLOW)
        
        # Applying layout constraints (Fixes 22, 23, 37, 38)
        self.place_at_grid(block_m, 'C4', scale_factor=0.3)
        self.place_at_grid(block_M, 'C5', scale_factor=0.6)
        self.place_at_grid(wall, 'C6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(block_m), FadeIn(block_M), FadeIn(wall))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        # Fixes 21, 36: Momentum label moved to B2
        label_momentum = Text("M = 10^n * m", font_size=20, color="#00FF00")
        self.place_at_grid(label_momentum, 'B2', scale_factor=0.6)
        self.play(Write(label_momentum))
        self.play(block_m.animate.shift(RIGHT * 0.5))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        # Logic for line 3
        self.play(Indicate(block_M))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.place_at_grid(pi_symbol, 'E4', scale_factor=0.4)
        self.play(FadeIn(pi_symbol))
        self.play(block_m.animate.shift(LEFT * 0.5), run_time=0.5)
