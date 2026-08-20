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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Bridge: From Counting to Polynomials", [
            "Generating functions encode sequences as polynomial coefficients.",
            "Mapping (a0, a1, a2) to a0 + a1x + a2x^2.",
            "How can we extract every nth term?"
        ])
        
        # Assets
        abacus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg")
        
        # Elements
        poly = MathTex("P(x) = a_0 + a_1x + a_2x^2", color=WHITE)
        self.place_at_grid(poly, 'C2', scale_factor=1.0)
        self.place_at_grid(abacus, 'B2', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(poly), FadeIn(abacus))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        # Highlight coefficients and terms
        coeffs = VGroup(poly[0][5], poly[0][9], poly[0][13])
        terms = VGroup(poly[0][7], poly[0][11], poly[0][15])
        self.play(
            coeffs.animate.set_color("#FFD700"),
            terms.animate.set_color("#87CEEB")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(magnifier, 'E2', scale_factor=0.8)
        self.play(FadeIn(magnifier), self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(
            poly.animate.set_color("#FF4500"),
            run_time=1.5
        )
        self.play(poly.animate.set_color(WHITE))
        self.wait(2)
