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
        self.setup_layout("Deriving the Truth: Euler's Formula", [
            "Use Euler's formula to count regions.",
            "Points and intersections form a planar graph.",
            "Sum binomial coefficients for the true total."
        ])
        
        # Load asset
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        
        # === Animation for Lecture Line 1 ===
        eq = MathTex("V", "-", "E", "+", "F", "=", "2")
        eq.set_color(GREEN)
        
        cube_for_eq = cube.copy()
        self.place_at_grid(cube_for_eq, "B2", scale_factor=0.5)
        self.place_in_area(eq, 'B3', 'B5', scale_factor=1.2)
        
        self.play(FadeIn(cube_for_eq), Write(eq))
        self.lecture[0].set_color(GREEN)

        # === Animation for Lecture Line 2 ===
        # Morphing representation
        V = eq[0]
        E = eq[2]
        F = eq[4]
        
        self.play(
            V.animate.set_color(GOLD),
            E.animate.set_color(GOLD),
            F.animate.set_color(GOLD)
        )
        self.lecture[1].set_color(GOLD)

        # === Animation for Lecture Line 3 ===
        final_eq = MathTex("6", "-", "12", "+", "8", "=", "2")
        final_eq.set_color(WHITE)
        self.place_in_area(final_eq, 'D3', 'D5', scale_factor=1.2)
        
        formula_group = VGroup(eq, final_eq)
        self.place_in_area(formula_group, 'B3', 'E5', scale_factor=1.0)
        
        self.play(FadeIn(final_eq))
        self.play(Flash(final_eq, color=WHITE, line_length=0.2))
        self.lecture[2].set_color(WHITE)
        self.wait(2)
