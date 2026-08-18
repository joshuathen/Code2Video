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
        self.setup_layout("Prerequisite Concept: Conservation of Energy", [
            "Gravity converts potential energy to kinetic energy.", 
            "Steeper drops gain velocity much faster.", 
            "Higher velocity 'buys' time for later travel."
        ])
        
        # Load asset
        rollercoaster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg")
        
        # Formulas
        ke_formula = MathTex(r"KE = \frac{1}{2}mv^2", color="#FF00FF")
        pe_formula = MathTex(r"PE = mgh", color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        self.place_in_area(rollercoaster, 'A4', 'B6', scale_factor=0.6)
        self.play(FadeIn(rollercoaster))
        
        # Applying requested formula placement
        self.place_in_area(pe_formula, 'B2', 'B4', scale_factor=0.6)
        self.place_in_area(ke_formula, 'E2', 'E4', scale_factor=0.6)
        self.play(Write(pe_formula), Write(ke_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFFF00"))
        # Using simple lines to represent slopes
        steep = Line(start=self.grid['C4'], end=self.grid['E6'], color=RED)
        gentle = Line(start=self.grid['C4'], end=self.grid['D6'], color=BLUE)
        self.play(Create(steep), Create(gentle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
