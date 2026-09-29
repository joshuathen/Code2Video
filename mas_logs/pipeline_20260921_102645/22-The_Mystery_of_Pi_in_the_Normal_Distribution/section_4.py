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
        lecture_lines = ["Integrating over dθ from zero to 2π.", "The 2π factor appears directly.", "It stems from circular symmetry."]
        self.setup_layout("The Emergence of π", lecture_lines)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")

        # Math components
        math_1 = MathTex(r"\int_{0}^{2\pi} d\theta = 2\pi").set_color(WHITE)
        math_2 = MathTex(r"I = \int_{0}^{\infty} r e^{-r^2} \cdot (2\pi) dr").set_color("#FFD700")
        math_3 = MathTex(r"= \pi \int_{0}^{\infty} 2r e^{-r^2} dr = \pi(1) = \pi").set_color("#00FF00")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(math_1, 'B3', 'B6', scale_factor=0.8)
        self.place_at_grid(compass, 'A4', scale_factor=0.5)
        self.play(Write(math_1), FadeIn(compass))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(math_2, 'C3', 'C6', scale_factor=0.8)
        self.play(Write(math_2))
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(math_3, 'D3', 'D6', scale_factor=0.8)
        self.place_at_grid(globe, 'E4', scale_factor=0.5)
        self.play(Write(math_3), FadeIn(globe))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
