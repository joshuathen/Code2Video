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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introduction: The Paradox of Pi", [
            "Two blocks collide on a frictionless track.",
            "Impacts count digits of the constant pi.",
            "Heavy block mass creates the sequence.",
            "Mass ratios dictate collision counts.",
            "Experience this paradox of pi."
        ])
        
        # === Animation for Lecture Line 1 ===
        track = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/track.svg", color=WHITE)
        self.place_in_area(track, "C1", "D6", scale_factor=1.5)
        m1 = Rectangle(width=0.5, height=0.5, color=RED).next_to(track, UP, buff=0)
        m2 = Rectangle(width=1.0, height=1.0, color=BLUE).next_to(track, UP, buff=0)
        m1_label = Text("m1", font_size=16).next_to(m1, UP, buff=0.1)
        m2_label = Text("m2", font_size=16).next_to(m2, UP, buff=0.1)
        self.play(FadeIn(track), FadeIn(m1), FadeIn(m2), Write(m1_label), Write(m2_label))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        count_text = Text("Collisions: 0", font_size=24, color=YELLOW)
        self.place_at_grid(count_text, "B3")
        self.play(Write(count_text))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        formula = MathTex(r"m_1=1, m_2=100^n", color=GREEN)
        self.place_at_grid(formula, "E3")
        self.play(Write(formula))
        self.lecture[2].set_color(GREEN)

        # === Animation for Lecture Line 4 ===
        # No specific visual update requested in storyboard, keeping layout
        self.lecture[3].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        blocks_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg", color=PURPLE)
        self.place_in_area(blocks_icon, "A1", "F6", scale_factor=2.0)
        self.play(FadeOut(track), FadeOut(m1), FadeOut(m2), FadeOut(m1_label), FadeOut(m2_label), FadeOut(count_text), FadeOut(formula), FadeIn(blocks_icon))
        self.lecture[4].set_color(PURPLE)
