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
            "Use Euler's Formula: V - E + F = 2.",
            "Count vertices and edges formed.",
            "Each intersection adds a region.",
            "Wait, n=6 gives 31 regions!",
            "The pattern of powers fails."
        ]
        self.setup_layout("The Geometric Insight", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex("V - E + F = 2", color=WHITE)
        self.place_in_area(formula, "A1", "A6")
        self.play(Write(formula))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Represent V (Vertices) and E (Edges) abstractly
        v_label = Text("V (Vertices)", color="#00FFFF")
        e_label = Text("E (Edges)", color="#00FFFF")
        VGroup(v_label, e_label).arrange(RIGHT, buff=1.0)
        self.place_at_grid(v_label, "B2")
        self.place_at_grid(e_label, "B5")
        self.play(FadeIn(v_label), FadeIn(e_label))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show an intersection region
        circle = Circle(radius=1.0, color=WHITE)
        chord1 = Line(np.array([-0.8, -0.6, 0]), np.array([0.8, 0.6, 0]), color=WHITE)
        chord2 = Line(np.array([-0.8, 0.6, 0]), np.array([0.8, -0.6, 0]), color=WHITE)
        intersection_poly = Polygon(
            np.array([0, 0, 0]), np.array([0.4, 0.2, 0]), np.array([0, 0.4, 0]), np.array([-0.4, 0.2, 0]),
            color="#FF00FF", fill_opacity=0.5
        )
        geo_group = VGroup(circle, chord1, chord2, intersection_poly)
        self.place_in_area(geo_group, "C2", "D5", scale_factor=0.8)
        self.play(Create(circle), Create(chord1), Create(chord2), FadeIn(intersection_poly))
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        count_text = Text("n=6 -> 31 regions", color="#FFFF00")
        self.place_at_grid(count_text, "E3")
        self.play(Write(count_text))
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        pattern_text = Text("Pattern: 2^(n-1)", color="#FF0000")
        cross = Line(pattern_text.get_corner(DL), pattern_text.get_corner(UR), color="#FF0000", stroke_width=4)
        pattern_group = VGroup(pattern_text, cross)
        self.place_at_grid(pattern_group, "F3")
        self.play(Write(pattern_text), Create(cross))
        self.lecture[4].set_color("#FF0000")
        self.wait(2)
