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
            "- Networks start with random internal knobs called weights.",
            "- Data flows forward from inputs to the final output.",
            "- The network calculates a weighted sum for its prediction."
        ]
        self.setup_layout("The Forward Pass: Making a Guess", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Create a network diagram with white connections (#FFFFFF) featuring small knob icons.
        self.lecture[0].set_color(WHITE)
        
        # Nodes
        input1 = Circle(radius=0.25, color=WHITE)
        input2 = Circle(radius=0.25, color=WHITE)
        hidden1 = Circle(radius=0.25, color=WHITE)
        output1 = Circle(radius=0.25, color=WHITE)
        
        self.place_at_grid(input1, "B1")
        self.place_at_grid(input2, "D1")
        self.place_at_grid(hidden1, "C3")
        self.place_at_grid(output1, "C5")
        
        # Connections
        c1 = Line(input1.get_right(), hidden1.get_left(), color=WHITE)
        c2 = Line(input2.get_right(), hidden1.get_left(), color=WHITE)
        c3 = Line(hidden1.get_right(), output1.get_left(), color=WHITE)
        
        # Knobs (Squares representing weights)
        k1 = Square(side_length=0.15, color=WHITE, fill_opacity=1).move_to(c1.point_from_proportion(0.5))
        k2 = Square(side_length=0.15, color=WHITE, fill_opacity=1).move_to(c2.point_from_proportion(0.5))
        k3 = Square(side_length=0.15, color=WHITE, fill_opacity=1).move_to(c3.point_from_proportion(0.5))
        
        network_group = VGroup(input1, input2, hidden1, output1, c1, c2, c3, k1, k2, k3)
        self.play(Create(network_group), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate green pulses (#00FF00) traveling through the network from inputs to outputs.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(GREEN)
        
        pulse1 = Dot(color=GREEN, radius=0.1).move_to(input1.get_center())
        pulse2 = Dot(color=GREEN, radius=0.1).move_to(input2.get_center())
        
        self.play(
            pulse1.animate.move_to(hidden1.get_center()),
            pulse2.animate.move_to(hidden1.get_center()),
            run_time=1.5
        )
        self.remove(pulse1, pulse2)
        
        pulse3 = Dot(color=GREEN, radius=0.1).move_to(hidden1.get_center())
        self.play(
            pulse3.animate.move_to(output1.get_center()),
            run_time=1.5
        )
        self.remove(pulse3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight a summation symbol (#FFFF00) appearing at the final output node.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(YELLOW)
        
        sum_sym = MathTex(r"\sum", color=YELLOW).scale(0.8)
        self.place_at_grid(sum_sym, "C5")
        
        self.play(
            output1.animate.set_color(YELLOW),
            Write(sum_sym),
            Flash(output1, color=YELLOW, flash_radius=0.4),
            run_time=1.5
        )
        self.wait(2)
