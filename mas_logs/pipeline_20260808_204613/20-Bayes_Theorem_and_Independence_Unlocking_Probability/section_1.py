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
        lecture_lines = ["Conditional probability: P(A|B) narrows our universe.", "Consider our total sample space, S.", "Subset B acts as our new reality.", "The intersection represents occurrences of A given B.", "Example: Orange animal reduces our animal cage set."]
        self.setup_layout("Prerequisite Review: Conditional Probability & Visual Logic", lecture_lines)
        
        # Define elements
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", font_size=36)
        
        venn = VGroup(
            Circle(radius=1.5, color=WHITE, fill_opacity=0.1),
            Circle(radius=0.8, color=BLUE, fill_opacity=0.3).shift(LEFT * 0.4),
            Circle(radius=0.8, color=RED, fill_opacity=0.3).shift(RIGHT * 0.4)
        )
        intersection = Intersection(venn[1], venn[2], color="#ADD8E6", fill_opacity=0.6)
        
        # Assets
        orange_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        cage_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cage.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(formula, 'B3', scale_factor=0.9)
        self.play(Write(formula), run_time=1)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.place_in_area(venn, 'B4', 'E6', scale_factor=0.6)
        self.play(FadeIn(venn), run_time=1)
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(orange_icon, 'C4', scale_factor=0.4)
        self.play(FadeIn(intersection), FadeIn(orange_icon), run_time=1)
        self.lecture[2].set_color("#ADD8E6")

        # === Animation for Lecture Line 4 ===
        inter_label = MathTex(r"P(A \cap B)", color="#FFFFE0", font_size=24)
        self.place_at_grid(inter_label, 'D5', scale_factor=0.7)
        self.play(Flash(inter_label), FadeIn(inter_label), run_time=1)
        self.lecture[3].set_color("#FFFFE0")

        # === Animation for Lecture Line 5 ===
        note = Text("Evidence changes our belief", font_size=20, color="#F0F8FF")
        self.place_at_grid(note, 'E3', scale_factor=0.8)
        self.place_at_grid(cage_icon, 'F3', scale_factor=0.4)
        self.play(FadeIn(note), FadeIn(cage_icon), run_time=1)
        self.lecture[4].set_color("#F0F8FF")

        self.wait(2)
