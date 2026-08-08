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
        self.setup_layout("The Problem: The Memory Fog", [
            "Older models like RNNs process words one by one.",
            "They often forget the beginning of long sentences.",
            "This \"memory fog\" makes understanding complex context difficult."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Older models like RNNs process words one by one.
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # RNN Box using Asset (Issue 23 & 26)
        # Load the provided box asset and position it to leave margin from lecture text.
        rnn_box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        rnn_box.set_color(BLUE)
        self.place_in_area(rnn_box, "B3", "D6", scale_factor=1.8)
        
        # RNN label centered over the box (Issue 28)
        rnn_label = Text("RNN", font_size=24, color=BLUE)
        self.place_at_grid(rnn_label, "A4") 
        
        self.play(Create(rnn_box), Write(rnn_label))
        
        words_list = ["The", "quick", "brown", "fox", "jumps"]
        word_mobjects = []
        # Target grids: sequence starts inside the RNN box from Column 3
        target_grids = ["C3", "C4", "C5", "C6", "D5"]
        
        for i, w in enumerate(words_list):
            word_obj = Text(w, font_size=24)
            # Start position for animation (off-screen right relative to grid)
            word_obj.move_to(self.grid["C6"] + RIGHT * 3)
            word_mobjects.append(word_obj)
            self.play(word_obj.animate.move_to(self.grid[target_grids[i]]), run_time=0.5)
            self.wait(0.1)

        # === Animation for Lecture Line 2 ===
        # They often forget the beginning of long sentences.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Early words in the sequence (at C3 and C4) turn grey and dim to simulate forgetting.
        self.play(
            word_mobjects[0].animate.set_color(GRAY).set_opacity(0.3),
            word_mobjects[1].animate.set_color(GRAY).set_opacity(0.3),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This "memory fog" makes understanding complex context difficult.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Memory Fog graphic (#808080) covering the oldest words (Issue 27)
        fog_rect = Rectangle(
            height=2.5, 
            width=1.8, 
            fill_color="#808080", 
            fill_opacity=0.6, 
            stroke_width=0
        )
        self.place_in_area(fog_rect, "B3", "D4")
        
        fog_label = Text("Memory Fog", font_size=22, color=WHITE, weight=BOLD)
        self.place_in_area(fog_label, "B3", "D4")
        
        fog_group = VGroup(fog_rect, fog_label)
        self.play(FadeIn(fog_group, shift=RIGHT * 0.3))
        self.wait(2)
