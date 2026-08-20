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
        self.setup_layout("The Paradox: Infinite Sums", [
            "Geometric series 1+2+4+8 sums to negative one.",
            "Euclidean view explodes towards infinity.",
            "2-adic lens locks the sum at negative one."
        ])
        
        # Assets
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        calculator_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        calculator_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")

        # === Animation for Lecture Line 1 ===
        series = MathTex("1 + 2 + 4 + 8 + \\dots = S", color=WHITE)
        self.place_in_area(series, 'A3', 'B5', scale_factor=1.0)
        self.place_at_grid(magnifying_glass, 'A2', scale_factor=0.3)
        self.play(Write(series), FadeIn(magnifying_glass))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        euclidean_text = Text("Euclidean: S \\to \\infty", color="#00FFFF")
        self.place_at_grid(euclidean_text, 'C2', scale_factor=0.8)
        self.place_at_grid(calculator_1, 'C5', scale_factor=0.3)
        self.play(FadeIn(euclidean_text), FadeIn(calculator_1))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        formula = MathTex("S = \\frac{1}{1-2} = -1", color="#FF00FF")
        self.place_in_area(formula, 'D3', 'F5', scale_factor=1.1)
        self.place_at_grid(calculator_2, 'E6', scale_factor=0.3)
        
        box = SurroundingRectangle(formula, color="#FF00FF", buff=0.1)
        self.play(FadeIn(formula), Create(box), FadeIn(calculator_2))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
