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
        lecture_lines = [
            "Neural networks are interconnected nodes transforming inputs.",
            "The loss function measures the network's prediction error.",
            "A high loss means the prediction is wrong."
        ]
        self.setup_layout("The Architecture: Guessing and Checking", lecture_lines)
        
        # Define Mobjects
        nodes = VGroup(*[Circle(radius=0.3, color=BLUE, fill_opacity=0.5) for _ in range(3)])
        self.place_at_grid(nodes[0], 'B2', scale_factor=0.8)
        self.place_at_grid(nodes[1], 'C3', scale_factor=0.8)
        self.place_at_grid(nodes[2], 'D2', scale_factor=0.8)
        
        connections = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), color=GRAY),
            Line(nodes[1].get_center(), nodes[2].get_center(), color=GRAY)
        )
        network = VGroup(nodes, connections)

        loss_label = Text("Loss = |Guess - Truth|", font_size=24, color=YELLOW)
        self.place_at_grid(loss_label, 'B5', scale_factor=0.7)
        loss_label.set_opacity(0)

        error_indicator = VGroup(
            Cross(color=RED, scale_factor=0.5),
            Text("WRONG", font_size=20, color=RED)
        ).arrange(DOWN)
        self.place_at_grid(error_indicator, 'D5', scale_factor=0.9)
        error_indicator.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(network))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(loss_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(FadeIn(error_indicator))
        self.play(error_indicator.animate.set_opacity(1))
        
        self.wait(2)
