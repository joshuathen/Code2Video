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
        title_text = "The Artificial Neuron: The Mathematical Heart"
        lecture_lines = [
            "A neuron gathers inputs and calculates a weighted sum.",
            "Think of this sum as a funnel collecting information.",
            "The result then passes through an activation function gatekeeper.",
            "Activation functions, like ReLU, determine if the neuron fires.",
            "The neuron output depends on if math exceeds the threshold."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors corresponding to lecture lines and animations
        COLOR_1 = "#FFD700"  # Gold for Neuron
        COLOR_2 = "#D3D3D3"  # Light Gray for Inputs/Weights
        COLOR_3 = "#FFFFFF"  # White for Formula
        COLOR_4 = "#0000FF"  # Blue for Activation Graph
        COLOR_5 = "#FFFF00"  # Yellow for Output

        # === Animation for Lecture Line 1 ===
        # Line: "A neuron gathers inputs and calculates a weighted sum."
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Neuron circle at C3
        neuron = Circle(radius=0.7, color=COLOR_1, fill_opacity=0.3)
        self.place_at_grid(neuron, "C3")
        
        # Incoming arrows from B2 and D2
        arrow1 = Arrow(start=self.grid["B2"], end=self.grid["C3"], buff=0.3, color=COLOR_1)
        arrow2 = Arrow(start=self.grid["D2"], end=self.grid["C3"], buff=0.3, color=COLOR_1)
        
        self.play(
            Create(neuron), 
            GrowArrow(arrow1), 
            GrowArrow(arrow2)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Think of this sum as a funnel collecting information."
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        
        # Input labels at B2 and D2
        x1_label = MathTex("x_1", color=COLOR_2)
        x2_label = MathTex("x_2", color=COLOR_2)
        self.place_at_grid(x1_label, "B2", scale_factor=0.8)
        self.place_at_grid(x2_label, "D2", scale_factor=0.8)
        
        # Weight labels along arrows
        w1_label = MathTex("w_1", color=COLOR_2).scale(0.6)
        w2_label = MathTex("w_2", color=COLOR_2).scale(0.6)
        
        # Position them relative to the grid/arrows to avoid overlap
        w1_label.move_to(self.grid["B2"] + RIGHT * 0.5 + DOWN * 0.2)
        w2_label.move_to(self.grid["D2"] + RIGHT * 0.5 + UP * 0.2)
        
        self.play(
            FadeIn(x1_label), FadeIn(x2_label),
            FadeIn(w1_label), FadeIn(w2_label),
            arrow1.animate.set_color(COLOR_2),
            arrow2.animate.set_color(COLOR_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "The result then passes through an activation function gatekeeper."
        self.play(self.lecture[2].animate.set_color(COLOR_3))
        
        # Weighted sum formula
        # [ISSUE 27 FIX]: Move formula to B3 to avoid overlap with axes at C3
        formula = MathTex(r"\sum w_i x_i + b", color=COLOR_3, font_size=28)
        self.place_at_grid(formula, "B3", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line: "Activation functions, like ReLU, determine if the neuron fires."
        self.play(self.lecture[3].animate.set_color(COLOR_4))
        
        # Transform summation/neuron into a blue (#0000FF) Sigmoid graph as per storyboard
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1, 0.5],
            x_length=2.2,
            y_length=1.4,
            axis_config={"include_tip": False, "color": COLOR_4, "stroke_width": 2},
        )
        self.place_at_grid(axes, "C3")
        
        # Using a pronounced sigmoid to show the S-curve clearly
        sigmoid_graph = axes.plot(lambda x: 1 / (1 + np.exp(-2.5*x)), color=COLOR_4)
        
        # [ISSUE 28 FIX]: Move sigmoid_label to B4 to avoid overlap with graph at C3
        sigmoid_label = Text("Sigmoid", font_size=20, color=COLOR_4)
        self.place_at_grid(sigmoid_label, "B4", scale_factor=0.7)
        
        self.play(
            FadeOut(formula),
            ReplacementTransform(neuron, axes),
            Create(sigmoid_graph),
            Write(sigmoid_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line: "The neuron output depends on if math exceeds the threshold."
        self.play(self.lecture[4].animate.set_color(COLOR_5))
        
        # [ISSUE 29 FIX]: Move y_label to C4 (closer to the center C3)
        # Output arrow from C3 to C4
        output_arrow = Arrow(start=self.grid["C3"], end=self.grid["C4"], buff=0.2, color=COLOR_5)
        y_label = MathTex("y", color=COLOR_5)
        self.place_at_grid(y_label, "C4", scale_factor=0.8)
        # Offset label to be clearly visible at the end of the arrow
        y_label.shift(RIGHT * 0.3)
        
        self.play(GrowArrow(output_arrow), Write(y_label))
        
        # Glowing effect using Indicate as per [L004]
        self.play(
            Indicate(y_label, color=COLOR_5, scale_factor=1.3),
            Indicate(output_arrow, color=COLOR_5),
            run_time=2
        )
        self.wait(2)
