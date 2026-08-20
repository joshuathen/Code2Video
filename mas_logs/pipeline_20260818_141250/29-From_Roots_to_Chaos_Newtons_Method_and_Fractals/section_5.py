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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Philosophical Takeaway", [
            "Deterministic rules generate extreme complexity.",
            "Fractals emerge from simple iteration.",
            "Beauty exists in infinite depth."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in simple iteration equation in #FFFFFF.
        eq = MathTex(r"z_{n+1} = z_n - \frac{f(z_n)}{f'(z_n)}", color=WHITE)
        self.place_at_grid(eq, 'B2', scale_factor=1.2)
        self.play(FadeIn(eq))
        self.lecture[0].set_color("#88AAFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Morph equation into a complex fractal shape.
        # Representing a simplified fractal look as a complex geometric path
        fractal = VGroup(*[Circle(radius=0.1, color=BLUE).shift(i*RIGHT + j*UP) for i in range(-2, 3) for j in range(-2, 3)])
        self.place_at_grid(fractal, 'D3', scale_factor=0.5)
        self.play(ReplacementTransform(eq, fractal))
        self.lecture[1].set_color("#88FF88")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Slowly zoom into infinite depth, highlighting repeating patterns.
        self.play(fractal.animate.scale(3), run_time=3)
        self.lecture[2].set_color("#FF8888")
        self.wait(2)
