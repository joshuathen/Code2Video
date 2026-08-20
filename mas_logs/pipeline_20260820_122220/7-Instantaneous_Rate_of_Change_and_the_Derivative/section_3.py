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
            "Derivative is the limit of change.",
            "Formalize as a difference quotient.",
            "Limit as h approaches zero.",
            "Connect to tangent line slope.",
            "Calculates instantaneous rate of change."
        ]
        self.setup_layout("Defining the Derivative", lecture_lines)
        
        # Formula for animation
        formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", font_size=40)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0].set_opacity(1)))
        # Fix issue 27 and 38
        self.place_in_area(formula, 'B3', 'E5', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1].set_opacity(1)))
        self.play(formula.animate.set_color(PURPLE)) # Highlight difference quotient
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2].set_opacity(1)))
        # Load asset and fix issue 18/28/38
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        self.place_at_grid(ruler, 'E3', scale_factor=0.5)
        self.play(FadeIn(ruler))
        self.play(ruler.animate.scale(0.5).fade(0.5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3].set_opacity(1)))
        tangent_line = Line(start=self.grid['D4'], end=self.grid['D6'], color=YELLOW)
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4].set_opacity(1)))
        self.play(Flash(formula, color=WHITE))
        self.wait(2)
