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
            "Filter coefficients by their indices modulo n.",
            "Apply the roots of unity to extract sums.",
            "The formula sums filtered polynomial values precisely.",
            "Visual window isolates target indices efficiently.",
            "This isolates modulo sum configurations perfectly."
        ]
        self.setup_layout("The Roots of Unity Filter", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        indices = VGroup(*[Text(str(i), font_size=24) for i in range(6)])
        indices.arrange(RIGHT, buff=0.3)
        self.place_at_grid(indices, 'B2', scale_factor=0.8)
        self.play(FadeIn(indices))
        self.play(indices.animate.set_color("#1ABC9C"), self.lecture[0].animate.set_color("#1ABC9C"))

        # === Animation for Lecture Line 2 ===
        root = MathTex(r"\omega", color="#F39C12")
        self.place_at_grid(root, 'C3', scale_factor=1.5)
        self.play(Write(root))
        self.play(self.lecture[1].animate.set_color("#F39C12"))

        # === Animation for Lecture Line 3 ===
        formula = MathTex(r"\frac{1}{n} \sum_{k=0}^{n-1} \omega^{-rk} P(\omega^k)", color=WHITE)
        self.place_at_grid(formula, 'D3', scale_factor=0.9)
        self.play(Write(formula))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 4 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg]
        window = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg")
        window.set_color("#95A5A6")
        self.place_at_grid(window, 'B2', scale_factor=0.5)
        self.play(FadeIn(window))
        self.play(window.animate.shift(RIGHT * 1.5), self.lecture[3].animate.set_color("#95A5A6"))

        # === Animation for Lecture Line 5 ===
        final_sum = MathTex(r"\text{Sum}", color="#FF00FF")
        self.place_at_grid(final_sum, 'E4', scale_factor=1.2)
        self.play(Flash(final_sum), self.lecture[4].animate.set_color("#FF00FF"))
        self.wait(2)
