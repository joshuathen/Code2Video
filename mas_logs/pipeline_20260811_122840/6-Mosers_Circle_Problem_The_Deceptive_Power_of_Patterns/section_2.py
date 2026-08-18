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
        lecture_lines = ["List results for 1, 2, 3, 4 points.", "The sequence is 1, 2, 4, 8.", "It seems like powers of two."]
        self.setup_layout("Pattern Recognition (The Trap)", lecture_lines)
        
        # Assets
        pizza = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg")
        cake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cake.svg")

        # === Animation for Lecture Line 1 ===
        # List results for 1, 2, 3, 4 points.
        seq_text = MathTex(r"n=1: 1, \quad n=2: 2, \quad n=3: 4, \quad n=4: 8", font_size=36)
        self.place_in_area(seq_text, "A1", "B6", scale_factor=0.7)
        self.place_at_grid(pizza, "C3", scale_factor=0.5)
        self.play(Write(seq_text), FadeIn(pizza))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # The sequence is 1, 2, 4, 8.
        seq_nums = MathTex(r"1, \quad 2, \quad 4, \quad 8", font_size=48)
        self.place_at_grid(seq_nums, "D3", scale_factor=0.8)
        self.place_at_grid(cake, "D5", scale_factor=0.5)
        self.play(FadeIn(seq_nums), FadeIn(cake))
        self.lecture[1].set_color(TEAL)

        # === Animation for Lecture Line 3 ===
        # It seems like powers of two.
        pattern_text = MathTex(r"2^{n-1}", color=ORANGE, font_size=60)
        self.place_at_grid(pattern_text, "E4", scale_factor=0.9)
        
        # Box to emphasize the "Trap"
        box = SurroundingRectangle(pattern_text, color=ORANGE, buff=0.2)
        
        self.play(Create(box), Write(pattern_text))
        self.lecture[2].set_color(ORANGE)
        self.wait(2)
