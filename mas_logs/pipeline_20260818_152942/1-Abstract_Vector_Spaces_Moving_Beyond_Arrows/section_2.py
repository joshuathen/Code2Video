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
        lecture_lines = ["Vectors need not be arrows.", "Functions, matrices, and polynomials can be vectors.", "They must satisfy eight specific algebraic axioms."]
        self.setup_layout("The Leap to Abstraction", lecture_lines)
        
        # Objects
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg]
        arrow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color=WHITE)
        func = FunctionGraph(lambda x: np.sin(x*3), x_range=[-1, 1], color=GREEN)
        matrix = MathTex(r"\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}", color=BLUE)
        poly = MathTex(r"ax^2 + bx + c", color=YELLOW)
        axiom_box = Rectangle(width=2, height=1.5, color=PURPLE)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(arrow, 'F2', scale_factor=0.6)
        self.play(FadeIn(arrow))
        self.wait(1)
        self.lecture[0].set_color("#FFFFFF")
        self.play(Transform(arrow, func))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        group = VGroup(func, matrix, poly).arrange(DOWN, buff=0.5)
        self.place_in_area(group, 'E3', 'F5', scale_factor=0.7)
        self.play(FadeIn(matrix), FadeIn(poly))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.place_in_area(axiom_box, 'C4', 'D6', scale_factor=0.9)
        self.play(Create(axiom_box))
        self.wait(2)
