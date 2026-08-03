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
        lecture_lines = [
            "Backpropagation traces error backwards through the network.",
            "We assign blame to every weight and bias.",
            "The chain rule calculates each parameter's error contribution.",
            "Gradients show the direction of maximum error increase.",
            "This tells us how to fix our mistakes."
        ]
        self.setup_layout("Backpropagation: The Chain of Blame", lecture_lines)

        # Setup Network Components (Refined Layout for Issue 40)
        # Weights (Knobs) - Using SVGMobject [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg] (Issue 24)
        knob_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg"
        
        w1_knob = SVGMobject(knob_path).set_color(WHITE)
        w2_knob = SVGMobject(knob_path).set_color(WHITE)
        w3_knob = SVGMobject(knob_path).set_color(WHITE)
        
        # Positioned in Col 4 to avoid overlapping lecture text
        self.place_at_grid(w1_knob, "B4", scale_factor=0.3)
        self.place_at_grid(w2_knob, "C4", scale_factor=0.3)
        self.place_at_grid(w3_knob, "D4", scale_factor=0.3)

        w1_label = MathTex("w_1", font_size=18).next_to(w1_knob, UP, buff=0.1)
        w2_label = MathTex("w_2", font_size=18).next_to(w2_knob, UP, buff=0.1)
        w3_label = MathTex("w_3", font_size=18).next_to(w3_knob, UP, buff=0.1)

        # Summation Node - Positioned at C5
        sum_node = Circle(radius=0.4, color=WHITE)
        sum_label = MathTex(r"\Sigma", font_size=24).move_to(sum_node.get_center())
        sum_group = VGroup(sum_node, sum_label)
        self.place_at_grid(sum_group, "C5")

        # Sigmoid Gate - Positioned at C6
        sigmoid_gate = Square(side_length=0.8, color=WHITE)
        sigmoid_label = MathTex(r"\sigma", font_size=24).move_to(sigmoid_gate.get_center())
        sigmoid_group = VGroup(sigmoid_gate, sigmoid_label)
        self.place_at_grid(sigmoid_group, "C6")

        # Output Y - Positioned at D6 to avoid overlap with Sigmoid
        output_y = Circle(radius=0.4, color=WHITE)
        output_label = Text("Y", font_size=20).move_to(output_y.get_center())
        output_group = VGroup(output_y, output_label)
        self.place_at_grid(output_group, "D6")

        # Connections
        line_sum_sig = Line(sum_group.get_right(), sigmoid_group.get_left())
        line_sig_out = Line(sigmoid_group.get_bottom(), output_group.get_top())
        
        line_w1_sum = Line(w1_knob.get_right(), sum_group.get_left())
        line_w2_sum = Line(w2_knob.get_right(), sum_group.get_left())
        line_w3_sum = Line(w3_knob.get_right(), sum_group.get_left())

        network = VGroup(
            output_group, sigmoid_group, sum_group, 
            w1_knob, w2_knob, w3_knob,
            w1_label, w2_label, w3_label,
            line_sum_sig, line_sig_out,
            line_w1_sum, line_w2_sum, line_w3_sum
        )

        self.add(network)

        # === Animation for Lecture Line 1 ===
        # Backpropagation traces error backwards through the network.
        # Red glow #FF0000 starts at the prediction output.
        self.lecture[0].set_color(YELLOW)
        
        error_glow = output_y.copy().set_stroke(RED, 8).set_fill(RED, opacity=0.3)
        self.play(FadeIn(error_glow))
        self.play(error_glow.animate.scale(1.2), run_time=0.5)
        self.play(error_glow.animate.scale(0.83), run_time=0.5)
        
        # === Animation for Lecture Line 2 ===
        # We assign blame to every weight and bias.
        # The glow moves backward through the Sigmoid gate.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Move glow backward
        error_dot = Dot(color=RED).move_to(output_group.get_center())
        self.add(error_dot)
        self.play(
            error_dot.animate.move_to(sigmoid_group.get_center()),
            error_glow.animate.move_to(sigmoid_group.get_center()),
            run_time=1
        )
        self.play(
            error_dot.animate.move_to(sum_group.get_center()),
            error_glow.animate.move_to(sum_group.get_center()),
            run_time=1
        )
        
        # === Animation for Lecture Line 3 ===
        # The chain rule calculates each parameter's error contribution.
        # The "Error" signal splits to trace back through weights.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        dot1 = error_dot.copy()
        dot2 = error_dot.copy()
        dot3 = error_dot.copy()
        
        self.play(
            dot1.animate.move_to(w1_knob.get_center()),
            dot2.animate.move_to(w2_knob.get_center()),
            dot3.animate.move_to(w3_knob.get_center()),
            error_glow.animate.scale(2).set_opacity(0.1),
            run_time=1.5
        )
        
        # === Animation for Lecture Line 4 ===
        # Gradients show the direction of maximum error increase.
        # Each weight knob #FFFF00 flashes indicating its "Blame" level.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        blame_color = "#FFFF00"
        self.play(
            w1_knob.animate.set_color(blame_color).scale(1.2),
            w2_knob.animate.set_color(blame_color).scale(1.2),
            w3_knob.animate.set_color(blame_color).scale(1.2),
            FadeOut(dot1, dot2, dot3),
            run_time=0.5
        )
        self.play(
            w1_knob.animate.set_color(WHITE).scale(0.83),
            w2_knob.animate.set_color(WHITE).scale(0.83),
            w3_knob.animate.set_color(WHITE).scale(0.83),
            run_time=0.5
        )

        # === Animation for Lecture Line 5 ===
        # This tells us how to fix our mistakes.
        # Calculated gradients appear next to each adjustable weight knob.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Derivatives are placed to the left of weights (Col 3)
        grad1 = MathTex(r"\frac{\partial E}{\partial w_1}", font_size=16, color=RED).next_to(w1_knob, LEFT, buff=0.1)
        grad2 = MathTex(r"\frac{\partial E}{\partial w_2}", font_size=16, color=RED).next_to(w2_knob, LEFT, buff=0.1)
        grad3 = MathTex(r"\frac{\partial E}{\partial w_3}", font_size=16, color=RED).next_to(w3_knob, LEFT, buff=0.1)
        
        self.play(
            Write(grad1),
            Write(grad2),
            Write(grad3),
            run_time=1.5
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
