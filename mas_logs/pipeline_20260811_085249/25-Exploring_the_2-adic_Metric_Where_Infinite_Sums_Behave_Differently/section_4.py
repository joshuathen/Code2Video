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
            "Summing powers of two: 1+2+4+8...",
            "This geometric series equals negative one.",
            "Binary representation fills with ones indefinitely.",
            "Two's complement arithmetic confirms this result.",
            "A surprising convergence in 2-adic space."
        ]
        self.setup_layout("The Grand Finale: Summing to Infinity", lecture_lines)
        
        # Define Mobjects
        abacus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg", color=WHITE)
        series = MathTex(r"1 + 2 + 4 + 8 + \dots", color=WHITE)
        sum_result = MathTex(r"= -1", color=WHITE)
        binary_ones = Text("...11111111", color=WHITE, font="monospace")
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(abacus, "B1")
        self.place_in_area(series, "B2", "B3", scale_factor=0.9)
        self.play(FadeIn(abacus), Write(series))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(sum_result, "C5")
        self.play(Write(sum_result))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(binary_ones, "C4")
        self.play(FadeIn(binary_ones))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # === Animation for Lecture Line 5 ===
        self.place_at_grid(computer, "D5")
        self.play(FadeIn(computer))
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        
        self.wait(2)
