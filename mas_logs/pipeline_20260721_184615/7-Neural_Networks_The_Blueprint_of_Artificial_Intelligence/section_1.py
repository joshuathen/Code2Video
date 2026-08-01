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
        self.setup_layout(
            "The Big Picture: Bio-inspiration to Computation", 
            [
                "Neural networks are models inspired by the human brain.",
                "Computers process data as patterns of numerical values.",
                "We map these input patterns to specific output predictions."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color matching: Brain (#88CCEE)
        self.play(self.lecture[0].animate.set_color("#88CCEE"))
        self.wait(1.5)

        # Load Brain Asset (Issue 19, 39)
        brain_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        brain_asset.set_color("#88CCEE")
        # Place in area B3-E5 (Issue 23, 39)
        self.place_in_area(brain_asset, 'B3', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(brain_asset))
        self.wait(1.0)

        # Create a network of nodes (#FFFFFF)
        nodes = VGroup(*[Dot(color="#FFFFFF", radius=0.1) for _ in range(6)])
        # Arrange nodes in 2-3-1 architecture
        layer1 = VGroup(nodes[0], nodes[1]).arrange(DOWN, buff=0.8)
        layer2 = VGroup(nodes[2], nodes[3], nodes[4]).arrange(DOWN, buff=0.8)
        layer3 = VGroup(nodes[5]).arrange(DOWN, buff=0.8)
        network_layers = VGroup(layer1, layer2, layer3).arrange(RIGHT, buff=1.5)
        
        # Add lines between layers
        lines = VGroup()
        for n1 in layer1:
            for n2 in layer2:
                lines.add(Line(n1.get_center(), n2.get_center(), color="#FFFFFF", stroke_width=1, stroke_opacity=0.5))
        for n2 in layer2:
            for n3 in layer3:
                lines.add(Line(n2.get_center(), n3.get_center(), color="#FFFFFF", stroke_width=1, stroke_opacity=0.5))
        
        network = VGroup(lines, nodes)
        # Position network in same area as brain for morphing (Issue 23, 39)
        self.place_in_area(network, 'B3', 'E5', scale_factor=0.8)

        # Morph Brain into Network
        self.play(
            ReplacementTransform(brain_asset, network),
            self.lecture[0].animate.set_color("#FFFFFF")
        )
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Color matching: Vector (#FFEE55)
        self.play(self.lecture[1].animate.set_color("#FFEE55"))
        self.wait(1.5)

        # Create numerical vector [0.1, 0.9, 0.4]
        # Use simple Text for the vector fallback (L022) to avoid LaTeX issues
        vector_txt = r"[0.1, 0.9, 0.4]"
        self.vector = Text(vector_txt, color="#FFEE55", font_size=24)
        # Move vector to C2 (Issue 22, 39)
        self.place_at_grid(self.vector, 'C2', scale_factor=0.7)
        
        self.play(FadeIn(self.vector))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Color matching: Prediction (#00FF00)
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1.5)

        # Prediction label 'Cat' using PNG Asset (Issue 19, 39)
        self.prediction_label = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        # Place the prediction_label at C6 (Issue 24, 39)
        self.place_at_grid(self.prediction_label, 'C6', scale_factor=0.7)
        
        cat_text = Text("Cat", color="#00FF00", font_size=24)
        cat_text.next_to(self.prediction_label, DOWN, buff=0.1)

        # Arrow from vector to prediction
        # Arrow starts near the vector and ends near the prediction label
        arrow = Arrow(
            start=self.vector.get_right() + RIGHT * 0.1, 
            end=self.prediction_label.get_left() + LEFT * 0.1, 
            color="#00FF00", 
            max_tip_length_to_length_ratio=0.2 # Corrected argument from L020
        )

        self.play(
            Create(arrow),
            FadeIn(self.prediction_label),
            FadeIn(cat_text),
            Indicate(self.prediction_label, color="#00FF00") # Use Indicate (L004)
        )
        self.wait(2.0)
