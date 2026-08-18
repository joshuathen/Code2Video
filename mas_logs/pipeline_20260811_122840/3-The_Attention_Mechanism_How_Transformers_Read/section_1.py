from manim import *

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
        lecture_lines = ["RNNs struggle with long sentences.", "Attention focuses on relevant information.", "Like finding clues in a mystery."]
        self.setup_layout("The Problem: Information Overload", lecture_lines)
        
        # Hide lecture lines for sequential reveal
        for line in self.lecture:
            line.set_opacity(0)
            
        # Assets
        magnifying_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg"
            
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        rnn_text = Text("RNN", color=BLUE).scale(1.2)
        sequence_text = Text("Sequence", color=WHITE).scale(1.2)
        dots = Text("...", color=WHITE)
        
        group = VGroup(rnn_text, dots, sequence_text).arrange(RIGHT, buff=0.5)
        # Applying fix for Issue 36/21: place at D4
        self.place_at_grid(group, "D4", scale_factor=0.7)
        
        self.play(Write(rnn_text), Write(sequence_text), FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_opacity(1)
        self.play(dots.animate.set_color(RED))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_opacity(1)
        
        # Applying fix for Issue 35/20: Use SVGMobject and position correctly
        magnifying_glass = SVGMobject(magnifying_icon_path, color=WHITE)
        label = Text("Attention", color=WHITE).scale(0.7)
        label.next_to(magnifying_glass, DOWN)
        magnifying_group = VGroup(magnifying_glass, label)
        
        # Position using area fix (Issue 37/22)
        self.place_in_area(magnifying_group, 'D2', 'F5', scale_factor=0.8)
        
        self.play(
            FadeOut(rnn_text),
            FadeOut(sequence_text),
            ReplacementTransform(dots, magnifying_group)
        )
        self.wait(2)
