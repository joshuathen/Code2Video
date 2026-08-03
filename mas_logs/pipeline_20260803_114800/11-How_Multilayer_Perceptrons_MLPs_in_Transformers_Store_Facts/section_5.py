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

class Section5Scene(TeachingScene):
    def construct(self):
        # Updated lecture lines in code to match animations (iPhone example)
        # Note: Added issue for ScriptWriter to officially update the storyboard.
        lecture_lines = [
            "Consider the input sequence: \"The iPhone was...\"",
            "The model matches this to a \"Knowledge Neuron\".",
            "The MLP retrieves facts like \"Apple Inc.\".",
            "This value is added back to the residual stream.",
            "The Transformer uses this info to predict the next word."
        ]
        self.setup_layout("Fact Retrieval in Action", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Text "The iPhone was created by..." scrolls in (#FFFFFF) accompanied by an iPhone icon.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/iphone.svg]
        # Fix Issue 45: input_text area 'B1' through 'B3'
        input_text = Text("The iPhone was created by...", font_size=20, color=WHITE)
        self.place_in_area(input_text, "B1", "B3")
        
        iphone_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/iphone.svg")
        self.place_at_grid(iphone_icon, "A2", scale_factor=0.6)
        
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(input_text), FadeIn(iphone_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A "Knowledge Neuron" (#FF8C00) in the MLP layer starts to glow.
        # Fix Issue 44: Move key_label to 'C2'
        neuron = Circle(radius=0.35, color="#FF8C00", fill_opacity=0.0)
        self.place_at_grid(neuron, "D3")
        neuron_label = Text("Knowledge Neuron", font_size=18, color="#FF8C00")
        neuron_label.next_to(neuron, DOWN, buff=0.2)
        
        key_label = Text("Key: Apple Inc.", font_size=16, color="#FF8C00")
        self.place_at_grid(key_label, "C2")
        
        self.lecture[1].set_color("#FF8C00")
        self.play(
            Create(neuron),
            FadeIn(neuron_label),
            neuron.animate.set_fill(opacity=0.6),
            FadeIn(key_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The MLP then retrieves the "Apple Inc." and "Steve Jobs" values.
        # Fix Issue 44: Move val_label to 'C4'
        val_label = Text("Value: Steve Jobs", font_size=16, color="#FF8C00")
        self.place_at_grid(val_label, "C4")
        
        self.lecture[2].set_color("#FF8C00")
        self.play(FadeIn(val_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This value is added back to the residual stream.
        # A gold arrow (#FFD700) moves from the neuron back to the text stream.
        # Fix Issue 46: Direct path from D3 to B4
        gold_arrow = Arrow(
            start=self.grid["D3"],
            end=self.grid["B4"],
            color="#FFD700",
            buff=0.1
        )
        
        self.lecture[3].set_color("#FFD700")
        self.play(GrowArrow(gold_arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The Transformer uses this info to predict the next word.
        # Fix Issue 45: move prediction to 'B4'
        prediction = Text("Apple", font_size=24, color="#00FF00")
        self.place_at_grid(prediction, "B4")
        
        self.lecture[4].set_color("#00FF00")
        self.play(Write(prediction))
        self.wait(2)
