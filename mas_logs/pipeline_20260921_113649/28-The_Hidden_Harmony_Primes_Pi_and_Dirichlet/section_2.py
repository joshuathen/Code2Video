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
        lecture_lines = [
            "Coprime probability links integers to circles.",
            "The Basel problem bridges discrete and continuous.",
            "Infinite products reveal the constant pi.",
            "Zeta functions illuminate prime behavior.",
            "Geometry emerges from pure number theory."
        ]
        self.setup_layout("Prime Patterns & Pi: The Basel Connection", lecture_lines)
        
        # Assets
        circle_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg"

        # === Animation for Lecture Line 1 ===
        # Display coprime integers randomly scattered
        dots = VGroup(*[Dot(radius=0.05, color=WHITE) for _ in range(50)])
        for dot in dots:
            dot.move_to(np.array([np.random.uniform(0, 5), np.random.uniform(-2.5, 2.5), 0]))
        self.play(FadeIn(dots))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Show transition from integer grid to circle asset
        circle1 = SVGMobject(circle_svg)
        self.place_at_grid(circle1, 'C3', scale_factor=1.5)
        self.play(FadeOut(dots), FadeIn(circle1))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        # Animate infinite product terms converging to pi
        formula = MathTex(r"\prod_{p} \left(1 - \frac{1}{p^2}\right)^{-1} = \frac{\pi^2}{6}", font_size=32)
        self.place_at_grid(formula, 'B3', scale_factor=1.2)
        self.play(Write(formula))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 4 ===
        # Visualize zeta function ripple effect across primes
        ripples = VGroup(*[Circle(radius=0.1, color="#FFFF00") for _ in range(5)])
        for i, ripple in enumerate(ripples):
            self.place_at_grid(ripple, f"D{i+1}", scale_factor=0.5)
        self.play(Create(ripples), run_time=2)
        self.play(self.lecture[3].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 5 ===
        # Display emerging perfect circular symmetry
        circle2 = SVGMobject(circle_svg)
        self.place_at_grid(circle2, 'E4', scale_factor=2.0)
        self.play(FadeOut(formula), FadeOut(ripples), FadeIn(circle2))
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.wait(2)
