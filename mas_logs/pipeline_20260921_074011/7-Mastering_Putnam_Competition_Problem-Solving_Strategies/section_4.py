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
        self.setup_layout("Strategy 3: Generating Functions and Recurrence", 
                          ["Translate recurrences into power series.", 
                           "Generating functions simplify term extraction.", 
                           "Algebraic manipulation solves recursive sequences."])
        
        # === Animation for Lecture Line 1 ===
        title_gen = Text("Generating Functions", color="#00FFCC")
        self.place_in_area(title_gen, 'A1', 'A3', scale_factor=0.6)
        self.play(FadeIn(title_gen))
        self.play(self.lecture[0].animate.set_color("#00FFCC"))

        # === Animation for Lecture Line 2 ===
        # Use Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg
        poly_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        poly = MathTex("f(x) = \\sum_{n=0}^{\\infty} a_n x^n", color=WHITE)
        
        # Arrange icon and formula
        group_poly = VGroup(poly_icon, poly).arrange(RIGHT, buff=0.2)
        self.place_in_area(group_poly, 'B2', 'C4', scale_factor=0.6)
        
        self.play(FadeIn(group_poly))
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        # Highlight coefficients
        a_n = poly[0][6:8]
        rect = SurroundingRectangle(a_n, color=YELLOW)
        self.play(Create(rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg
        abacus_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")
        new_term = MathTex("a_{n+1}", color="#FFFF00")
        
        group_term = VGroup(abacus_icon, new_term).arrange(RIGHT, buff=0.2)
        self.place_at_grid(group_term, 'D4', scale_factor=0.6)
        
        self.play(ReplacementTransform(a_n.copy(), new_term))
        self.play(FadeIn(abacus_icon))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)
