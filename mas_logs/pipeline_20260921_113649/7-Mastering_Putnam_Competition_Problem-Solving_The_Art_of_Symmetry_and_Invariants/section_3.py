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
            "Identify symmetry in polynomials and functional equations.",
            "Use substitution to create mirror-image systems.",
            "Solve using f(x) + f(1-x) = 1 symmetry.",
            "Visualize balance beams pivoting at center points."
        ]
        self.setup_layout("Core Strategy: Exploiting Symmetry", lecture_lines)
        
        # Define elements
        sym_fig = Circle(color="#00CED1", fill_opacity=0.5)
        # Apply positioning fix as requested (Issue 41/28)
        self.place_in_area(sym_fig, 'D4', 'E6', scale_factor=0.45)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(sym_fig), self.lecture[0].animate.set_color("#00CED1"))
        
        # === Animation for Lecture Line 2 ===
        self.play(Rotate(sym_fig, angle=PI/2), self.lecture[1].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 3 ===
        rect1 = Rectangle(color="#ADFF2F", height=0.5, width=1.5).move_to(self.grid["D2"])
        rect2 = Rectangle(color="#ADFF2F", height=0.5, width=1.5).move_to(self.grid["D5"])
        eq_text = MathTex("f(x) + f(1-x) = 1", color="#808080").scale(0.8)
        self.place_at_grid(eq_text, "B3", scale_factor=0.8)
        self.play(Create(rect1), Create(rect2), Write(eq_text), self.lecture[2].animate.set_color("#ADFF2F"))
        
        # === Animation for Lecture Line 4 ===
        # Using SVGMobject for balance beam as requested
        balance_beam = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balance.svg", color="#FFD700")
        self.place_at_grid(balance_beam, "E4", scale_factor=0.6)
        
        self.play(FadeOut(sym_fig), FadeOut(rect1), FadeOut(rect2), FadeOut(eq_text),
                  FadeIn(balance_beam), self.lecture[3].animate.set_color("#FFD700"))
        
        self.wait(1)
