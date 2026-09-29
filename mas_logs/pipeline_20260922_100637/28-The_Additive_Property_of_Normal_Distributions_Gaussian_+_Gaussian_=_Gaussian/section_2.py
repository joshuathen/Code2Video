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
        self.setup_layout("Prerequisites: Independence and Parameters", [
            "A Gaussian distribution is defined by mean and variance.", 
            "Independence means one outcome doesn't affect another.", 
            "These are the essential building blocks for our sum."
        ])
        
        # Objects
        mu_text = MathTex(r"\mu", font_size=40, color=WHITE)
        sigma_sq_text = MathTex(r"\sigma^2", font_size=40, color=WHITE)
        params = VGroup(mu_text, sigma_sq_text).arrange(RIGHT, buff=1.0)
        self.place_at_grid(params, 'B2')
        
        circle1 = Circle(radius=0.5, color=GREY).shift(LEFT * 0.8)
        circle2 = Circle(radius=0.5, color=GREY).shift(RIGHT * 0.8)
        circles = VGroup(circle1, circle2)
        self.place_at_grid(circles, 'D3')
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(mu_text), FadeIn(sigma_sq_text))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(circles))
        self.play(self.lecture[1].animate.set_color("#FFCC00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            mu_text.animate.set_color("#00FF00"),
            sigma_sq_text.animate.set_color("#00FF00")
        )
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1)
