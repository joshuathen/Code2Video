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
        self.setup_layout("Defining the Riemann Zeta Function", [
            "We define the Riemann Zeta function as this sum.",
            "For s greater than 1, the series works perfectly.",
            "Analytic continuation extends this to the complex plane."
        ])
        
        pen = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pen.svg")
        paper = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paper.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        zeta_formula = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}", color=WHITE)
        self.place_in_area(zeta_formula, 'A2', 'B5', scale_factor=1.2)
        self.place_at_grid(pen, 'A1', scale_factor=0.2)
        self.play(FadeIn(pen), Write(zeta_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        
        self.place_at_grid(paper, 'D3', scale_factor=0.5)
        self.play(FadeIn(paper))
        
        s_text = MathTex(r"s \in \mathbb{C}, \text{Re}(s) > 1", color=BLUE)
        self.place_at_grid(s_text, 'B4', scale_factor=0.8)
        self.play(FadeIn(s_text))
        
        sum_expansion = MathTex(r"1 + \frac{1}{2^s} + \frac{1}{3^s} + \dots", color=GREEN)
        self.place_in_area(sum_expansion, 'C2', 'C5', scale_factor=0.9)
        self.play(Write(sum_expansion))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        
        plane = ComplexPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": False})
        self.place_in_area(plane, 'D2', 'F5', scale_factor=0.8)
        
        self.play(
            FadeOut(zeta_formula), 
            FadeOut(s_text), 
            FadeOut(sum_expansion), 
            FadeOut(pen), 
            FadeOut(paper)
        )
        self.play(Create(plane))
        self.wait(2)
