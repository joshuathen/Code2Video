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
        self.setup_layout("The 'Locality' of Knowledge", [
            "Knowledge is localized to neuron clusters.", 
            "Different domains occupy different weight paths.", 
            "Fine-tuning may disrupt existing localized facts."
        ])
        
        # Load asset
        neuron_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        
        # Create matrix visualization
        matrix = VGroup()
        for i in range(36):
            # Create a cell with an icon
            cell = neuron_icon.copy().scale(0.02)
            matrix.add(cell)
        
        matrix.arrange_in_grid(6, 6, buff=0.1)
        # Fix: Move to right side to avoid overlap
        self.place_in_area(matrix, "A2", "F6", scale_factor=0.8)
        self.add(matrix)

        # === Animation for Lecture Line 1 ===
        # Knowledge localized: highlight a cluster in BLUE
        self.play(self.lecture[0].animate.set_color(BLUE))
        cluster1 = VGroup(matrix[7], matrix[8], matrix[13], matrix[14])
        self.play(*[m.animate.set_color(BLUE) for m in cluster1])
        
        # === Animation for Lecture Line 2 ===
        # Different domains occupy paths: highlight another cluster in GREEN
        self.play(self.lecture[1].animate.set_color(GREEN))
        cluster2 = VGroup(matrix[20], matrix[21], matrix[26], matrix[27])
        self.play(*[m.animate.set_color(GREEN) for m in cluster2])
        
        # === Animation for Lecture Line 3 ===
        # Fine-tuning: disrupt with highlight
        self.play(self.lecture[2].animate.set_color(RED))
        overlap = VGroup(matrix[14])
        # Box around cluster as requested in storyboard
        rect = SurroundingRectangle(cluster1, color=PURPLE, buff=0.1)
        self.play(Create(rect), overlap.animate.set_color(YELLOW))
        
        self.wait(2)
