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
        lecture_lines = ["Conditional probability focuses on a specific condition.", 
                         "Visualize this as zooming into subset B.", 
                         "We measure how much of A overlaps within B."]
        self.setup_layout("Prerequisite Warm-up: The Concept of Conditional Probability", lecture_lines)
        
        # Assets
        dot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        mag_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg")
        
        # Elements
        event_A = Circle(radius=1.0, color=BLUE, fill_opacity=0.3)
        event_B = Circle(radius=0.8, color=RED, fill_opacity=0.3)
        event_B.shift(RIGHT * 0.5)
        
        venn_group = VGroup(dot_icon, event_A, event_B)
        
        intersection = Intersection(event_A, event_B, color="#FFFF00", fill_opacity=0.6)
        
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(dot_icon, 'A3', 'A3', scale_factor=0.5)
        self.play(FadeIn(dot_icon), FadeIn(event_A), FadeIn(event_B))
        self.place_in_area(venn_group, 'C3', 'E5', scale_factor=0.8)
        self.lecture[0].set_color("#00FFFF")
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(intersection))
        self.lecture[1].set_color("#FF00FF")
        
        # === Animation for Lecture Line 3 ===
        self.place_in_area(mag_icon, 'F1', 'F1', scale_factor=0.5)
        self.play(Write(formula), FadeIn(mag_icon))
        self.place_at_grid(formula, 'F3', scale_factor=0.9)
        self.lecture[2].set_color("#00FF00")
        self.play(formula.animate.set_color("#00FF00"))
        
        self.wait(2)
