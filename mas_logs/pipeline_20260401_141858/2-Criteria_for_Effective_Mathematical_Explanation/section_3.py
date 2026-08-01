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
        # Setup layout
        title = "Criterion 1: Visual-Symbolic Mapping"
        lines = [
            "Never leave equations floating in a vacuum.",
            "Map every symbol to a visual representation.",
            "See (a+b)^2 as a divided geometric square.",
            "One large square, one small, and two rectangles.",
            "The visual proves the algebra instantly."
        ]
        self.setup_layout(title, lines)

        # Colors for the sections
        color_a2 = "#FFD700"  # Gold
        color_b2 = "#ADFF2F"  # GreenYellow
        color_ab = "#1E90FF"  # DodgerBlue

        # === Animation for Lecture Line 1-2 ===
        for i in range(2):
            self.play(self.lecture[i].animate.set_color(YELLOW), run_time=0.5)
            self.wait(0.5)

        # Replacing MathTex with VGroup of Text to avoid FileNotFoundError: 'latex'
        equation = VGroup(
            Text("(a+b)^2", font_size=32), 
            Text("=", font_size=32), 
            Text("a^2", font_size=32), 
            Text("+", font_size=32), 
            Text("2ab", font_size=32), 
            Text("+", font_size=32), 
            Text("b^2", font_size=32)
        ).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(equation, "F1", "F6")

        # === Geometric Construction ===
        a_side = 2.0
        b_side = 1.0
        
        sq_a = Square(side_length=a_side, fill_opacity=0.8, color=color_a2).set_stroke(WHITE, 2)
        sq_b = Square(side_length=b_side, fill_opacity=0.8, color=color_b2).set_stroke(WHITE, 2)
        rect_ab1 = Rectangle(width=a_side, height=b_side, fill_opacity=0.8, color=color_ab).set_stroke(WHITE, 2)
        rect_ab2 = Rectangle(width=b_side, height=a_side, fill_opacity=0.8, color=color_ab).set_stroke(WHITE, 2)

        # Labels for the geometry
        label_a2 = Text("a^2", font_size=20).move_to(sq_a.get_center())
        label_b2 = Text("b^2", font_size=20).move_to(sq_b.get_center())
        label_ab1 = Text("ab", font_size=20).move_to(rect_ab1.get_center())
        label_ab2 = Text("ab", font_size=20).move_to(rect_ab2.get_center())

        # Positioning the pieces relative to each other
        geom_group = VGroup(sq_a, rect_ab1, rect_ab2, sq_b)
        rect_ab1.next_to(sq_a, RIGHT, buff=0)
        rect_ab2.next_to(sq_a, DOWN, buff=0)
        sq_b.next_to(rect_ab2, RIGHT, buff=0)
        
        self.place_in_area(geom_group, "A2", "D5")
        
        # Update label positions after group move
        label_a2.move_to(sq_a.get_center())
        label_b2.move_to(sq_b.get_center())
        label_ab1.move_to(rect_ab1.get_center())
        label_ab2.move_to(rect_ab2.get_center())

        # Animation sequence
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Write(equation[0]), Write(equation[1]))
        self.wait(0.5)

        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        # a^2
        self.play(FadeIn(sq_a), Write(label_a2))
        self.play(Indicate(equation[2]), equation[2].animate.set_color(color_a2))
        self.wait(0.3)
        
        # 2ab
        self.play(FadeIn(rect_ab1), FadeIn(rect_ab2), Write(label_ab1), Write(label_ab2))
        self.play(Indicate(equation[4]), equation[4].animate.set_color(color_ab))
        self.wait(0.3)
        
        # b^2
        self.play(FadeIn(sq_b), Write(label_b2))
        self.play(Indicate(equation[6]), equation[6].animate.set_color(color_b2))
        self.wait(0.5)

        self.play(Write(equation[3]), Write(equation[5]))
        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        final_rect = SurroundingRectangle(geom_group, color=WHITE)
        self.play(Create(final_rect))
        self.wait(2)
