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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Anatomy of a Neuron", [
            "Neurons receive inputs and apply adjustable weights.",
            "Weights decide which input features matter most.",
            "Biases help fine-tune the final decision."
        ])
        
        # Colors
        COLOR_INPUT = "#FFFFFF"
        COLOR_WEIGHT = "#00FFFF"
        COLOR_BIAS = "#AAAAAA"
        COLOR_NEURON = "#FFD700" 

        # === Animation for Lecture Line 1 ===
        # "Neurons receive inputs and apply adjustable weights."
        self.lecture[0].set_color(YELLOW)
        
        # Inputs
        input1_circle = Circle(radius=0.4, color=COLOR_INPUT)
        label1 = Text("Color", font_size=20, color=COLOR_INPUT)
        input1_group = VGroup(input1_circle, label1).arrange(DOWN, buff=0.1)
        self.place_at_grid(input1_group, 'B3', scale_factor=0.8) # Issue 39 Fix 2
        
        input2_circle = Circle(radius=0.4, color=COLOR_INPUT)
        label2 = Text("Shape", font_size=20, color=COLOR_INPUT)
        input2_group = VGroup(input2_circle, label2).arrange(DOWN, buff=0.1)
        self.place_at_grid(input2_group, 'D3', scale_factor=0.8) # Issue 39 Fix 2
        
        # Neuron
        neuron_circle = Circle(radius=0.6, color=COLOR_NEURON)
        neuron_label = Text("Neuron", font_size=20, color=COLOR_NEURON)
        neuron_group = VGroup(neuron_circle, neuron_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(neuron_group, 'C5', scale_factor=0.8) # Issue 39 Fix 3
        
        # Weights
        # Connections from circles
        weight1 = Line(input1_circle.get_right(), neuron_circle.get_left(), color=COLOR_WEIGHT)
        weight2 = Line(input2_circle.get_right(), neuron_circle.get_left(), color=COLOR_WEIGHT)
        
        w1_label = MathTex("w_1", font_size=24, color=COLOR_WEIGHT)
        w2_label = MathTex("w_2", font_size=24, color=COLOR_WEIGHT)
        
        # Place labels relative to lines
        w1_label.move_to(weight1.get_center() + UP * 0.3)
        w2_label.move_to(weight2.get_center() + DOWN * 0.3)

        self.play(Create(input1_group), Create(input2_group))
        self.play(Create(neuron_group))
        self.play(Create(weight1), Create(weight2), Write(w1_label), Write(w2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Weights decide which input features matter most."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(
            weight1.animate.set_stroke_width(8),
            weight2.animate.set_stroke_width(8),
            run_time=0.5
        )
        self.play(
            weight1.animate.set_stroke_width(4),
            weight2.animate.set_stroke_width(4),
            run_time=0.5
        )
        
        self.play(
            weight1.animate.set_stroke_width(10).set_color(YELLOW),
            w1_label.animate.scale(1.2).set_color(YELLOW),
        )
        self.wait(1)
        self.play(
            weight1.animate.set_stroke_width(4).set_color(COLOR_WEIGHT),
            w1_label.animate.scale(1/1.2).set_color(COLOR_WEIGHT),
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Biases help fine-tune the final decision."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Issue 22: Integration of knob asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/kn.svg]
        bias_knob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/kn.svg")
        bias_knob.set_color(COLOR_BIAS)
        bias_label_text = Text("Bias", font_size=20, color=COLOR_BIAS)
        bias_group = VGroup(bias_knob, bias_label_text).arrange(DOWN, buff=0.1)
        
        # Issue 39 Fix 1: Place at E6
        self.place_at_grid(bias_group, 'E6', scale_factor=0.8)
        
        # Connection to neuron circle
        bias_conn = Line(bias_knob.get_top(), neuron_circle.get_bottom(), color=COLOR_BIAS)
        b_val = MathTex("+b", font_size=24, color=COLOR_BIAS)
        b_val.next_to(bias_conn, RIGHT, buff=0.1)
        
        self.play(FadeIn(bias_group))
        self.play(Create(bias_conn), Write(b_val))
        
        # Simple rotation animation for the knob
        self.play(Rotate(bias_knob, angle=PI/2))
        self.wait(2)
