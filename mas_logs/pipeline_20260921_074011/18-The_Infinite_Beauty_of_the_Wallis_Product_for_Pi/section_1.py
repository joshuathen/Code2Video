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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Pi is usually defined as circumference over diameter.",
            "But Pi can also be an infinite product.",
            "John Wallis discovered a surprising pattern."
        ]
        self.setup_layout("Introduction: Pi Beyond Geometry", lecture_lines)
        
        # Elements
        circle = Circle(radius=1.5, color=BLUE)
        label_pi = MathTex(r"\pi = \frac{C}{d}").set_color(YELLOW)
        
        fraction_product = MathTex(r"\pi = 2 \prod_{n=1}^{\infty} \frac{4n^2}{4n^2-1}").set_color(GREEN)
        
        # Load asset (SVG files must be loaded using SVGMobject)
        wallis_portrait = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/portrait.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(circle, 'B3', scale_factor=0.6)
        self.place_at_grid(label_pi, 'B4', scale_factor=0.8)
        self.play(FadeIn(circle), Write(label_pi))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(circle), FadeOut(label_pi))
        self.place_in_area(fraction_product, 'D2', 'E5', scale_factor=0.8)
        self.play(Write(fraction_product))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(fraction_product))
        self.place_in_area(wallis_portrait, 'D3', 'F5', scale_factor=0.9)
        self.play(FadeIn(wallis_portrait))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
