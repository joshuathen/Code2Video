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
        lecture_lines = [
            "The Euler product relates sum to primes.",
            "Primes act as filters for integers.",
            "This reveals hidden prime number structure.",
            "It connects analysis to prime theory.",
            "A beautiful bridge between different fields."
        ]
        self.setup_layout("The Bridge: The Euler Product Formula", lecture_lines)
        
        # Define colors for lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Define elements
        zeta_sum = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}")
        euler_prod = MathTex(r"= \prod_{p} \frac{1}{1-p^{-s}}")
        formula = VGroup(zeta_sum, euler_prod).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.place_in_area(formula, 'A2', 'C5', scale_factor=0.9)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        primes = VGroup(*[Text(str(p), font_size=24, color="#00FF00") for p in [2, 3, 5, 7, 11]])
        primes.arrange(RIGHT, buff=0.4)
        self.place_at_grid(primes, 'D2', scale_factor=0.9)
        
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg")
        self.place_at_grid(filter_icon, 'D5', scale_factor=0.5)
        self.play(FadeIn(primes, shift=UP), FadeIn(filter_icon))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        expansion = MathTex(r"= (1 + 2^{-s} + 2^{-2s} + \dots)(1 + 3^{-s} + \dots) \dots")
        self.place_at_grid(expansion, 'E2', scale_factor=0.5)
        self.play(Write(expansion))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_at_grid(bridge_icon, 'F3', scale_factor=0.5)
        self.play(FadeIn(bridge_icon))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.wait(1)
