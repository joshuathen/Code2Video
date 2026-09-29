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
        self.setup_layout("Mathematical Formulation (The Bound)", [
            "Heisenberg-Gabor sets a fundamental bound.",
            "Uncertainty is not a measurement error.",
            "Wave packets obey this physical law.",
            "Gaussian shapes minimize this uncertainty product.",
            "Squeezing time forces frequency to widen."
        ])
        
        # === Animation for Lecture Line 1 ===
        inequality = MathTex(r"\Delta t \cdot \Delta f \geq \frac{1}{4\pi}", color=WHITE)
        self.place_in_area(inequality, 'B2', 'B5', scale_factor=1.2)
        self.play(FadeIn(inequality))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using SVG asset for ruler
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color="#FF3333")
        self.place_at_grid(ruler, "D2", scale_factor=0.5)
        self.play(FadeIn(ruler), ruler.animate.scale(1.2).scale(1/1.2))
        self.lecture[1].set_color("#FF3333")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using SVG asset for clock
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg", color="#33FF33")
        self.place_at_grid(clock, "D5", scale_factor=0.5)
        self.play(FadeIn(clock), clock.animate.scale(1.2).scale(1/1.2))
        self.lecture[2].set_color("#33FF33")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(inequality.animate.set_color("#FFFF33"))
        self.lecture[3].set_color("#FFFF33")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        bound_rect = Rectangle(width=4, height=2, color="#FF5733", stroke_width=4)
        self.place_in_area(bound_rect, 'D3', 'F5', scale_factor=0.9)
        self.play(Create(bound_rect))
        self.lecture[4].set_color("#FF5733")
        self.wait(2)
