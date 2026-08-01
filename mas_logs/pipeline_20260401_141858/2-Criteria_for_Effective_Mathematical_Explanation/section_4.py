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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        title = "Criterion 2: Logical Cohesion (The Chain of 'Why')"
        lines = [
            "Every step needs a clear 'Because'.",
            "Logical bridges connect each part of the chain.",
            "If one link breaks, the explanation fails."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create boxes for steps
        box_a_rect = Rectangle(width=1.2, height=0.8, color=WHITE)
        box_a_text = Text("Step A", font_size=18, color=WHITE)
        box_a = VGroup(box_a_rect, box_a_text)
        
        box_b_rect = Rectangle(width=1.2, height=0.8, color=WHITE)
        box_b_text = Text("Step B", font_size=18, color=WHITE)
        box_b = VGroup(box_b_rect, box_b_text)
        
        box_c_rect = Rectangle(width=1.2, height=0.8, color=WHITE)
        box_c_text = Text("Step C", font_size=18, color=WHITE)
        box_c = VGroup(box_c_rect, box_c_text)
        
        # Position boxes
        self.place_in_area(box_a, "C1", "D2")
        self.place_in_area(box_b, "C3", "D4")
        self.place_in_area(box_c, "C5", "D6")
        
        self.play(FadeIn(box_a), FadeIn(box_b), FadeIn(box_c))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create Arches (Bridges)
        # Bridge 1: A to B
        start_1 = box_a.get_top() + RIGHT*0.2
        end_1 = box_b.get_top() + LEFT*0.2
        bridge_1 = ArcBetweenPoints(start_1, end_1, radius=2.0, color="#1E90FF", stroke_width=4)
        label_1 = Text("Because", font_size=14, color=WHITE)
        label_1.next_to(bridge_1, UP, buff=0.1)
        
        # Bridge 2: B to C
        start_2 = box_b.get_top() + RIGHT*0.2
        end_2 = box_c.get_top() + LEFT*0.2
        bridge_2 = ArcBetweenPoints(start_2, end_2, radius=2.0, color="#1E90FF", stroke_width=4)
        label_2 = Text("Because", font_size=14, color=WHITE)
        label_2.next_to(bridge_2, UP, buff=0.1)
        
        self.play(Create(bridge_1), Write(label_1))
        self.play(Create(bridge_2), Write(label_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Red X on the first bridge
        cross_red = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color=RED),
            Line(UP+RIGHT, DOWN+LEFT, color=RED)
        ).scale(0.3)
        self.place_at_grid(cross_red, "B2") # Positioned over bridge 1
        
        self.play(Create(cross_red))
        self.play(bridge_1.animate.set_color(RED))
        self.wait(0.5)
        
        # Step B and Step C fall off screen
        falling_group = VGroup(box_b, box_c, bridge_2, label_2, bridge_1, cross_red, label_1)
        self.play(
            falling_group.animate.shift(DOWN * 8),
            run_time=2,
            rate_func=rush_from
        )
        self.wait(2)
