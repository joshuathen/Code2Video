from manim import *
import numpy as np

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
            "Logs help us find the hidden exponent.",
            "It's like a password for a digital lock.",
            "If 2 to the x is 8, the exponent is 3.",
            "Log base 2 of 8 is 3.",
            "The lock opens with the exponent."
        ]
        self.setup_layout("Logarithms: Finding the Hidden Exponent", lecture_lines)
        
        # Assets
        password_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/password.svg")
        lock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg")

        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        eq = MathTex(r"\\log_b(x) = n", font_size=48)
        self.place_in_area(eq, 'B2', 'B5', scale_factor=0.9)
        self.play(Write(eq))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(password_icon, 'A4', scale_factor=0.5)
        self.play(FadeIn(password_icon))
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        eq2 = MathTex(r"2^x = 8", font_size=48)
        self.place_in_area(eq2, 'C2', 'C5', scale_factor=0.9)
        self.play(Write(eq2))
        
        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]))
        log_eq = MathTex(r"\\log_2(8) = 3", font_size=48)
        self.place_in_area(log_eq, 'D2', 'D5', scale_factor=0.9)
        self.play(Write(log_eq))
        
        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.place_at_grid(lock_icon, 'E4', scale_factor=0.5)
        self.play(FadeIn(lock_icon))
        
        final_eq = MathTex(r"\\log_{10}(100) = 2", font_size=48)
        self.place_in_area(final_eq, 'F2', 'F5', scale_factor=0.7)
        self.play(Write(final_eq))
        self.play(final_eq.animate.set_color(GREEN))
        self.wait(2)
