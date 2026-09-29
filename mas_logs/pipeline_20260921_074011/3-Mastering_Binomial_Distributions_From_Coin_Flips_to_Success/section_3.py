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
            "Formula calculates exact success probability.",
            "C(n,k) counts total success arrangements.",
            "p^k defines success probability.",
            "(1-p)^(n-k) defines failure.",
            "Graph shows success likelihood distribution."
        ]
        self.setup_layout("The Binomial Formula: The Logic of Combinations", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        formula = MathTex(r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}")
        self.place_in_area(formula, 'B1', 'C4', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/balls.svg]
        balls = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balls.svg")
        self.place_at_grid(balls, "D2", scale_factor=0.5)
        
        comb = MathTex(r"\binom{n}{k} = \frac{n!}{k!(n-k)!}")
        self.place_in_area(comb, 'D1', 'D6', scale_factor=0.8)
        self.play(FadeIn(balls), ReplacementTransform(formula.copy(), comb))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        pk = MathTex(r"p^k")
        pk.set_color(BLUE)
        self.place_at_grid(pk, 'E2', scale_factor=0.9)
        self.play(FadeIn(pk))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.play(self.lecture[3].animate.set_color(YELLOW))
        qnk = MathTex(r"(1-p)^{n-k}")
        qnk.set_color(RED)
        self.place_at_grid(qnk, 'E5', scale_factor=0.9)
        self.play(FadeIn(qnk))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bins.svg]
        bins = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bins.svg")
        
        axes = Axes(x_range=[0, 10, 1], y_range=[0, 0.4, 0.1], axis_config={"include_tip": False})
        bar_chart = BarChart(values=[0.01, 0.05, 0.12, 0.2, 0.25, 0.2, 0.12, 0.05, 0.01], bar_names=[str(i) for i in range(9)])
        graph = VGroup(axes, bar_chart).scale(0.3)
        self.place_in_area(graph, 'F2', 'F5', scale_factor=0.5)
        
        self.play(FadeOut(formula), FadeOut(comb), FadeOut(pk), FadeOut(qnk), FadeOut(balls))
        self.play(Create(graph))
        self.play(bins.animate.set_color(YELLOW)) # Flashing effect representation
        self.place_at_grid(bins, "C3", scale_factor=0.5)
        self.play(FadeIn(bins))
        self.wait(2)
