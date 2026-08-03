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
        # Data from Storyboard
        title = "The Key-Value Memory Analogy"
        lecture_lines = [
            "Each hidden neuron acts as a specific memory key.",
            "W1 weights store these keys to detect concepts.",
            "If a key matches, the corresponding value is retrieved.",
            "W2 weights store the information values for each key.",
            "This mechanism creates a searchable key-value memory system."
        ]
        
        # Setup the layout
        self.setup_layout(title, lecture_lines)
        
        # Color Constants
        CYAN = "#00FFFF"
        WHITE = "#FFFFFF"
        GREEN = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Each hidden neuron acts as a specific memory key.
        # Position adjusted per Issue 36: C3 -> C4
        neuron = Dot(radius=0.15, color=CYAN)
        self.place_at_grid(neuron, "C4")
        
        self.play(
            FadeIn(neuron),
            self.lecture[0].animate.set_color(CYAN),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # W1 weights store these keys to detect concepts.
        # Asset Integration per Issue 26
        # Position adjusted per Issue 36: B3 -> B4
        key_text = Text("Key (W1)", font_size=20, color=CYAN)
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg").scale(0.3).set_color(CYAN)
        key_label = VGroup(key_icon, key_text).arrange(RIGHT, buff=0.2)
        self.place_at_grid(key_label, "B4")
        
        self.play(
            FadeIn(key_label),
            self.lecture[1].animate.set_color(CYAN),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # If a key matches, the corresponding value is retrieved.
        # Asset Integration per Issue 26
        # Position adjusted per Issue 35: C1 -> C2
        concept_text = Text("Concept: Paris", font_size=18, color=WHITE)
        concept_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paris.svg").scale(0.35)
        concept_group = VGroup(concept_icon, concept_text).arrange(DOWN, buff=0.1)
        self.place_at_grid(concept_group, "C2")
        
        self.play(
            FadeIn(concept_group),
            self.lecture[2].animate.set_color(WHITE),
            run_time=1
        )
        # Move to neuron at C4
        self.play(
            concept_group.animate.move_to(self.grid["C4"]),
            run_time=1.5
        )
        self.play(Indicate(neuron, color=CYAN), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # W2 weights store the information values for each key.
        # Position adjusted per Issue 36: D3 -> D4
        value_label = Text("Value (W2)", font_size=20, color=GREEN)
        self.place_at_grid(value_label, "D4")
        
        # Label pointing from neuron (C4) to output (C6 per result positioning)
        pointing_arrow = Arrow(start=self.grid["C4"], end=self.grid["C6"], color=GREEN, buff=0.4)
        
        self.play(
            Write(value_label),
            Create(pointing_arrow),
            self.lecture[3].animate.set_color(GREEN),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This mechanism creates a searchable key-value memory system.
        # Position adjusted per Issue 37: C5 -> C6, scale_factor=0.8
        result_text = Text("Result: France", font_size=22, color=GREEN)
        self.place_at_grid(result_text, "C6", scale_factor=0.8)
        
        self.play(
            Write(result_text),
            self.lecture[4].animate.set_color(GREEN),
            run_time=1
        )
        self.play(Flash(result_text, color=GREEN), run_time=1)
        self.wait(2)
