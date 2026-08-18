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
            "Backpropagation applies the calculus chain rule.",
            "We trace error backwards from the output.",
            "Blame is distributed to each contributing neuron.",
            "Each step identifies where the error grew.",
            "It clarifies exactly how to improve performance."
        ]
        self.setup_layout("The Core Concept: Backpropagation Mechanics", lecture_lines)
        
        # Use SVG Asset
        neuron_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg"
        nodes = VGroup(*[SVGMobject(neuron_path, color="#FF00FF") for _ in range(4)])
        nodes.arrange(RIGHT, buff=0.5)
        
        # Positioned per VideoCritic constraints (D4 / D3-E5)
        self.place_in_area(nodes, 'D3', 'E5', scale_factor=0.8)
        
        connections = VGroup(*[Line(nodes[i].get_right(), nodes[i+1].get_left(), color=WHITE) for i in range(3)])
        self.add(nodes, connections)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        pulse = Dot(color="#00FFFF").move_to(nodes[3].get_center())
        self.add(pulse)
        # Pulse travels backwards
        self.play(pulse.animate.move_to(nodes[0].get_center()), run_time=2)
        self.remove(pulse)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Highlight weights (connections) to #FFFF00
        self.play(connections.animate.set_color("#FFFF00"))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.wait(0.5)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        self.wait(1)
