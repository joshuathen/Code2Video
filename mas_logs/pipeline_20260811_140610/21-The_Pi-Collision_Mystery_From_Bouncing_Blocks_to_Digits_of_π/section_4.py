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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Revelation: Calculating Pi", [
            "Mass ratios reveal digits of pi.",
            "Increasing mass reveals 3, 1, 4.",
            "Physical systems compute irrational numbers.",
            "Each digit emerges from the collisions.",
            "Mechanical motion calculates pi digits."
        ])
        
        # Load assets
        block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        block_1 = block.copy()
        self.place_at_grid(block_1, 'B2', scale_factor=0.5)
        self.play(FadeIn(block_1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        digit_3 = Text("3", font_size=36, color=YELLOW)
        digit_1 = Text("1", font_size=36, color=YELLOW)
        digit_4 = Text("4", font_size=36, color=YELLOW)
        
        self.place_at_grid(digit_3, 'B4', scale_factor=1.0)
        self.place_at_grid(digit_1, 'C4', scale_factor=1.0)
        self.place_at_grid(digit_4, 'D4', scale_factor=1.0)
        self.play(Write(digit_3), Write(digit_1), Write(digit_4))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3357FF")
        formula = MathTex(r"\text{Collisions} \approx \pi", font_size=28, color=WHITE)
        self.place_in_area(formula, 'E1', 'E6')
        self.play(FadeIn(formula))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF33A8")
        wall_icon = wall.copy()
        self.place_at_grid(wall_icon, 'C2', scale_factor=0.4)
        self.play(FadeIn(wall_icon))
        self.play(Indicate(wall_icon))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F3FF33")
        block_move = block.copy()
        self.place_at_grid(block_move, 'F2', scale_factor=0.4)
        self.play(block_move.animate.shift(RIGHT * 2))
        self.play(Circumscribe(block_move))
