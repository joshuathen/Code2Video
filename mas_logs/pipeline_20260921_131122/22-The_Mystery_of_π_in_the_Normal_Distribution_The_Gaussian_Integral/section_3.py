from manim import *
import os

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
        self.setup_layout("The Mathematical Trick: Squaring the Integral", 
                          ["To solve I, we square it.", 
                           "Multiplying two integrals creates volume.", 
                           "This yields a 2D Gaussian surface."])
        
        # Define equations
        eq1 = MathTex(r"I = \int_{-\infty}^{\infty} e^{-x^2} dx")
        eq2 = MathTex(r"I^2 = \left( \int_{-\infty}^{\infty} e^{-x^2} dx \right) \left( \int_{-\infty}^{\infty} e^{-y^2} dy \right)")
        eq3 = MathTex(r"I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2+y^2)} dx dy")

        # Load Asset
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg"
        surface_icon = SVGMobject(asset_path) if os.path.exists(asset_path) else Dot(color=BLUE)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(eq1, "B3", scale_factor=0.7)
        self.play(Write(eq1))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(eq2, "C2", "C4", scale_factor=0.8)
        self.play(FadeTransform(eq1.copy(), eq2))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(eq3, "D2", "D4", scale_factor=0.8)
        self.play(FadeTransform(eq2, eq3))
        
        self.place_at_grid(surface_icon, "E3", scale_factor=1.5)
        self.play(FadeIn(surface_icon))
        
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
