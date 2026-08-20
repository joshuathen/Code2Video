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
        lecture_lines = ["Backpropagation is the game of assigning blame.", 
                         "We trace error backward, layer by layer.", 
                         "Each neuron receives feedback on its contribution."]
        self.setup_layout("The Core Concept: Backpropagation (The Blame Game)", lecture_lines)
        
        # Load assets
        neuron_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg"
        neurons = VGroup(*[SVGMobject(neuron_asset) for _ in range(6)])
        
        # Apply layout fixes (per issues 32, 33, 34)
        self.place_at_grid(neurons[0], 'C2', scale_factor=0.9)
        self.place_at_grid(neurons[1], 'C4', scale_factor=0.9)
        self.place_at_grid(neurons[2], 'D2', scale_factor=0.9)
        self.place_at_grid(neurons[3], 'D4', scale_factor=0.9)
        self.place_at_grid(neurons[4], 'E2', scale_factor=0.8)
        self.place_at_grid(neurons[5], 'E4', scale_factor=0.8)
        
        self.add(neurons)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        error_label = Text("Error", color=RED, font_size=24).next_to(neurons[5], RIGHT)
        self.play(Write(error_label))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        connections = VGroup(
            Line(neurons[5].get_center(), neurons[3].get_center(), color=GOLD),
            Line(neurons[3].get_center(), neurons[1].get_center(), color=GOLD),
            Line(neurons[5].get_center(), neurons[4].get_center(), color=GOLD),
            Line(neurons[4].get_center(), neurons[2].get_center(), color=GOLD)
        )
        arrows = VGroup(*[Arrow(line.get_start(), line.get_end(), color=GREEN, buff=0.1) for line in connections])
        self.play(Create(connections), Create(arrows))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(
            neurons[5].animate.set_color(YELLOW),
            neurons[3].animate.set_color(YELLOW),
            neurons[1].animate.set_color(YELLOW),
            run_time=2
        )
