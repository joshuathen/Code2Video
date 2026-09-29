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
        lecture_lines = [
            "Mean equals n times p.",
            "Variance equals n times p times q.",
            "Mean represents the distribution's center.",
            "Variance represents the distribution's spread.",
            "Parameters shift the distribution's mass."
        ]
        self.setup_layout("Mean and Variance: The Intuition", lecture_lines)
        
        # Mobjects
        mean_formula = MathTex("E[X] = np").scale(1.5)
        var_formula = MathTex("V[X] = np(1-p)").scale(1.5)
        
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        die = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(mean_formula, 'B3', 'B6', scale_factor=0.9)
        self.place_at_grid(coin, 'A5', scale_factor=0.5)
        self.play(FadeIn(mean_formula), FadeIn(coin))
        self.play(self.lecture[0].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 2 ===
        self.place_in_area(var_formula, 'D3', 'D6', scale_factor=0.9)
        self.play(FadeIn(var_formula))
        self.play(self.lecture[1].animate.set_color("#00BFFF"))

        # === Animation for Lecture Line 3 ===
        arrow1 = Arrow(start=self.grid['B3'], end=self.grid['B2'], color=WHITE)
        self.play(Create(arrow1))
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(FadeOut(arrow1))

        # === Animation for Lecture Line 4 ===
        arrow2 = Arrow(start=self.grid['D3'], end=self.grid['D2'], color=WHITE)
        self.play(Create(arrow2))
        self.play(self.lecture[3].animate.set_color("#00BFFF"))
        self.play(FadeOut(arrow2))

        # === Animation for Lecture Line 5 ===
        highlight_box = SurroundingRectangle(mean_formula, color=WHITE, buff=0.2)
        self.place_in_area(highlight_box, 'B3', 'B6', scale_factor=1.0)
        self.play(Create(highlight_box))
        self.play(highlight_box.animate.set_color("#FFD700"))
        self.place_at_grid(die, 'F5', scale_factor=0.5)
        self.play(FadeIn(die))
        self.play(FadeOut(highlight_box))
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
