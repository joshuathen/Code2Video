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
        lecture_lines = ["Newton's method finds complex roots iteratively.", "Each step improves our guess.", "It behaves like a smart leap toward zero."]
        self.setup_layout("Numerical Algorithms: The Newton-Raphson Method", lecture_lines)
        
        # Elements
        plane = ComplexPlane(x_range=[-3, 3], y_range=[-3, 3]).scale(0.7)
        self.place_in_area(plane, 'C3', 'E5', scale_factor=0.9)
        
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        z0 = Dot(plane.c2p(2, 2), color="#FF0000")
        z_root = Dot(plane.c2p(0, 0), color="#00FF00")
        z_label = MathTex("z_0", color="#FF0000")
        self.place_at_grid(z_label, 'D5', scale_factor=0.7)
        root_label = MathTex("z_{root}", color="#00FF00").next_to(z_root, DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), Create(plane), run_time=1)
        self.place_at_grid(compass, 'B2', scale_factor=0.5)
        self.play(FadeIn(z0), FadeIn(z_label), FadeIn(z_root), FadeIn(root_label), FadeIn(compass))

        # === Animation for Lecture Line 2 ===
        iter_formula = MathTex("z_{n+1} = z_n - \\frac{f(z_n)}{f'(z_n)}", font_size=24)
        self.place_at_grid(iter_formula, 'A5', scale_factor=0.6)
        self.place_at_grid(calculator, 'A2', scale_factor=0.5)
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFFF00"), Write(iter_formula), FadeIn(calculator))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"), z0.animate.move_to(plane.c2p(0.1, 0.1)))
        convergence_circle = Circle(radius=0.2, color="#00FF00").move_to(z_root)
        self.play(Create(convergence_circle), Indicate(z_root))
        self.wait(1)
