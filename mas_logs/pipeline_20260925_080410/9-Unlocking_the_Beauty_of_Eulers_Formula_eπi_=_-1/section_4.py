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
        self.setup_layout("The Euler Identity Derivation", [
            "- Series expansions of e^x, cos, and sin link.",
            "- Replacing x with ix merges these series.",
            "- This combination traces a circle.",
            "- Euler's formula connects exponentials to rotation.",
            "- This creates cos(x) plus i times sin(x)."
        ])
        
        # === Animation for Lecture Line 1 ===
        s_ex = MathTex(r"e^x = 1 + x + \frac{x^2}{2!} + \dots", font_size=20)
        s_cos = MathTex(r"\cos x = 1 - \frac{x^2}{2!} + \dots", font_size=20)
        s_sin = MathTex(r"\sin x = x - \frac{x^3}{3!} + \dots", font_size=20)
        
        series = VGroup(s_ex, s_cos, s_sin).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(series, 'B2', scale_factor=0.9)
        self.play(Write(series))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        s_eix = MathTex(r"e^{ix} = 1 + ix - \frac{x^2}{2!} - i\frac{x^3}{3!} + \dots", font_size=20, color="#FF0000")
        self.place_at_grid(s_eix, 'C3', scale_factor=0.8)
        self.play(ReplacementTransform(series, s_eix))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color="#FFFF00")
        self.place_at_grid(circle, 'E3', scale_factor=0.6)
        dot = Dot(color="#FFFF00").move_to(circle.get_right())
        self.play(Create(circle), FadeIn(dot))
        self.play(Rotate(dot, angle=2*PI, about_point=circle.get_center()), run_time=2)
        self.lecture[2].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 4 & 5 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg] as reference
        identity = MathTex(r"e^{ix} = \cos x + i \sin x", color="#FFFFFF")
        self.place_at_grid(identity, 'D3', scale_factor=1.0)
        identity.next_to(circle, UP, buff=0.5)
        self.play(Write(identity))
        self.lecture[3].set_color("#FFFFFF")
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
