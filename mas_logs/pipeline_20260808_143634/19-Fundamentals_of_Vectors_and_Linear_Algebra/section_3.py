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
        lecture_lines = ["Scalars change a vector's length.", "Negative scalars reverse a vector's direction.", "Multiplying stretches or compresses the vector."]
        self.setup_layout("Scalar Multiplication: Stretching and Flipping", lecture_lines)
        
        # Asset paths
        spring_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg"
        mirror_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg"
        
        # Mobjects
        v = Arrow(ORIGIN, RIGHT * 2, color=WHITE, buff=0)
        v_group = VGroup(v, MathTex(r"\vec{v}").next_to(v.get_end(), UP))
        self.place_in_area(v_group, 'D3', 'F5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(v_group))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg]
        spring = SVGMobject(spring_path) if os.path.exists(spring_path) else Dot(color=BLUE)
        self.place_at_grid(spring, 'B3', scale_factor=0.5)
        
        v2 = Arrow(ORIGIN, RIGHT * 4, color="#00FF00", buff=0)
        label_2v = MathTex(r"2\vec{v}", color="#00FF00")
        self.place_at_grid(label_2v, 'F6', scale_factor=0.7)
        
        self.play(FadeIn(spring), ReplacementTransform(v.copy(), v2), Write(label_2v))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg]
        mirror = SVGMobject(mirror_path) if os.path.exists(mirror_path) else Dot(color=RED)
        self.place_at_grid(mirror, 'B4', scale_factor=0.5)
        
        v_neg = Arrow(ORIGIN, LEFT * 2, color="#FF0000", buff=0)
        label_neg_v = MathTex(r"-\vec{v}", color="#FF0000")
        self.place_at_grid(label_neg_v, 'D1', scale_factor=0.7)
        
        self.play(FadeIn(mirror), ReplacementTransform(v.copy(), v_neg), Write(label_neg_v))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
