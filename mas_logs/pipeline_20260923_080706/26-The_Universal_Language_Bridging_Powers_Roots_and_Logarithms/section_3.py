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
        self.setup_layout("Logarithms: The Time-Seeker", ["Logarithms are the true time-seeker.", "They calculate how much time passed.", "We solve for the unknown exponent."])
        
        # Define elements
        eqn1 = MathTex("b^x = y")
        eqn2 = MathTex("x = \\log_b(y)")
        log_word = eqn2[0][2:5]
        
        def safe_load_svg(path):
            if os.path.exists(path):
                return SVGMobject(path)
            return Dot(color=RED)

        clock = safe_load_svg("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        hourglass = safe_load_svg("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hourglass.svg")
        
        # Position elements
        self.place_at_grid(eqn1, 'B4', scale_factor=1.2)
        self.place_at_grid(eqn2, 'D4', scale_factor=1.2)
        self.place_at_grid(clock, 'B2', scale_factor=0.6)
        self.place_at_grid(hourglass, 'D2', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), Write(eqn1), FadeIn(clock))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"), ReplacementTransform(eqn1.copy(), eqn2), FadeOut(clock))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFA500"), log_word.animate.set_color("#FFA500"), FadeIn(hourglass))
        
        self.wait(2)
