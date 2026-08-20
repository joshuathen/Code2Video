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
        self.setup_layout("The Forward Pass: Data Propagation", [
            "Inputs travel through layers of neurons.",
            "Each weight influences the final output.",
            "Data propagation creates a network prediction."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display a single neuron node
        node = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        node.set_color("#FFFFFF")
        self.place_at_grid(node, 'C3', scale_factor=0.8) # Adjusted per issue 27/42
        self.play(FadeIn(node))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw arrows connecting multiple layers
        layer1 = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg").set_color("#FFFF00") for _ in range(3)])
        layer2 = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg").set_color("#FFFF00") for _ in range(3)])
        
        self.place_at_grid(layer1, 'B4', scale_factor=0.7) # Adjusted per issue 28/43
        self.place_at_grid(layer2, 'D4', scale_factor=0.7) # Adjusted per issue 28/43
        
        arrows = VGroup()
        for d1 in layer1:
            for d2 in layer2:
                arrows.add(Line(d1.get_center(), d2.get_center(), color="#FFFF00", stroke_width=2))
        
        self.play(FadeIn(layer1), FadeIn(layer2), Create(arrows))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show data flowing, activation, and final output
        flow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg").set_color("#00FF00")
        self.place_at_grid(flow, 'B4', scale_factor=0.5)
        
        # Activation function representation
        act_func = Text("σ", color="#FF00FF").scale(0.7)
        self.place_at_grid(act_func, 'C3', scale_factor=0.7)
        
        # Final output
        output = Text("Output", color="#00FFFF").scale(0.7)
        self.place_at_grid(output, 'C5', scale_factor=0.9) # Adjusted per issue 29/44
        
        self.play(
            flow.animate.move_to(self.grid['D4']),
            FadeIn(act_func),
            FadeIn(output)
        )
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
