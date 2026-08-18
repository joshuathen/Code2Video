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
        # Titles and lecture lines
        self.setup_layout("The Key-Value Memory Mechanism", [
            "- Each hidden neuron acts as a key-value pair.",
            "- The first weight matrix stores the \"Keys\" or patterns.",
            "- A \"Key\" might represent \"The capital of France.\"",
            "- The second weight matrix stores the \"Values\" or facts.",
            "- The \"Value\" vector injects the information \"Paris\" back."
        ])

        # Colors as per instructions
        key_color = "#FFA500"  # Orange
        value_color = "#EE82EE" # Violet

        # === Animation for Lecture Line 1 ===
        # "Each hidden neuron acts as a key-value pair."
        self.lecture[0].set_color(WHITE)
        
        # Key vector asset and container
        key_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg").set_color(key_color)
        key_box = Rectangle(width=0.8, height=2, color=key_color, fill_opacity=0.2)
        
        # Value vector container (asset used in line 4)
        value_box = Rectangle(width=0.8, height=2, color=value_color, fill_opacity=0.2)
        
        self.place_at_grid(key_box, "C2")
        self.place_at_grid(key_asset, "C2", scale_factor=0.6)
        self.place_at_grid(value_box, "C5") # Positioned at C5 per Issue 39
        
        key_label = Text("Key Vector", font_size=20, color=key_color)
        value_label = Text("Value Vector", font_size=20, color=value_color)
        
        self.place_at_grid(key_label, "B2", scale_factor=0.7) # Scaled and positioned per Issue 37
        self.place_at_grid(value_label, "B5", scale_factor=0.7) # Scaled and positioned per Issue 37
        
        self.play(
            FadeIn(key_box), FadeIn(key_asset), FadeIn(value_box),
            Write(key_label), Write(value_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The first weight matrix stores the \"Keys\" or patterns."
        self.play(self.lecture[1].animate.set_color(key_color))
        
        w1_note = Text("Part of W1 Matrix", font_size=16, color=WHITE)
        self.place_at_grid(w1_note, "D2")
        
        pattern_text = Text("Pattern:\n\"The capital of\n[Country]\"", font_size=16, color=key_color)
        self.place_at_grid(pattern_text, "C2", scale_factor=0.45) # Scaled per Issue 38
        
        self.play(Write(w1_note), Write(pattern_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A \"Key\" might represent \"The capital of France.\""
        self.play(self.lecture[2].animate.set_color(key_color))
        
        input_arrow = Arrow(start=LEFT*0.5, end=RIGHT*0.5, color=WHITE, buff=0)
        self.place_at_grid(input_arrow, "C1")
        
        self.play(input_arrow.animate.move_to(self.grid["C2"]), run_time=1)
        
        # Glow effect
        glow = key_box.copy().set_stroke(key_color, width=10).set_fill(key_color, opacity=0.4)
        self.play(FadeIn(glow), run_time=0.4)
        self.play(FadeOut(glow), run_time=0.4)
        self.remove(input_arrow)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The second weight matrix stores the \"Values\" or facts."
        self.play(self.lecture[3].animate.set_color(value_color))
        
        w2_note = Text("Part of W2 Matrix", font_size=16, color=WHITE)
        self.place_at_grid(w2_note, "D5") # Positioned at D5 per Issue 39
        
        # Label Value: 'Info: [City Name] [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/paris.svg]'
        info_desc = Text("Info: [City Name]", font_size=16, color=value_color)
        paris_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paris.svg").set_color(value_color)
        
        info_group = VGroup(info_desc, paris_asset).arrange(DOWN, buff=0.1)
        self.place_at_grid(info_group, "C5", scale_factor=0.6) # Scaled and positioned at C5 per Issue 38/39
        
        self.play(Write(w2_note), FadeIn(info_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The \"Value\" vector injects the information \"Paris\" back."
        self.play(self.lecture[4].animate.set_color(value_color))
        
        # Show the Value vector being released and moving towards the output.
        val_vec = Arrow(start=LEFT*0.4, end=RIGHT*0.4, color=value_color, buff=0)
        self.place_at_grid(val_vec, "C5") # Positioned at C5 per Issue 39
        
        self.play(val_vec.animate.move_to(self.grid["C6"]), run_time=1.5)
        
        self.wait(2)
