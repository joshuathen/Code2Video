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
        self.setup_layout("Defining Pi (π)", [
            "This ratio is defined as Pi.",
            "It is an infinite, non-repeating number.",
            "Unrolling a circle shows its length."
        ])
        
        # Assets
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        diameter_line = Line(circle.get_left(), circle.get_right(), color="#00FFFF")
        d_label = MathTex("d", color="#00FFFF")
        c_label = MathTex("C", color="#FF00FF")
        
        # Fixing positions per crit
        self.place_in_area(circle, 'B3', 'C4', scale_factor=0.9)
        self.place_in_area(diameter_line, 'B3', 'C4', scale_factor=0.9)
        self.place_at_grid(d_label, 'C2', scale_factor=0.8)
        
        # Group formula for crit
        pi_def = MathTex(r"\pi = \frac{C}{d}", font_size=48)
        formula_group = VGroup(pi_def)
        self.place_in_area(formula_group, 'D3', 'F5', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(circle))
        self.play(Create(diameter_line), Write(d_label))
        self.play(Write(formula_group))
        self.play(FadeIn(c_label.next_to(circle, UP)))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        irrational_text = Text("3.14159...", font_size=24, color=RED).next_to(formula_group, DOWN)
        self.play(Write(irrational_text))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Simple unrolling visual: line length = circumference
        unrolled_line = Line(start=self.grid['F1'], end=self.grid['F6'], color=WHITE)
        self.play(Create(unrolled_line))
        self.play(FadeIn(Text("Length = C", font_size=20).next_to(unrolled_line, UP)))
        self.wait(2)
