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
        title = "The Anatomy of a Neuron"
        lecture_lines = [
            "- An artificial neuron receives multiple input signals.",
            "- Each input is multiplied by its unique weight.",
            "- The weighted inputs are summed with a bias value.",
            "- An activation function decides the final output strength.",
            "- This process determines if a signal passes forward."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        YELLOW_C = "#FFFF00"
        GREEN_C = "#00FF00"
        WHITE_C = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Neuron circle
        neuron = Circle(radius=1.0, color=WHITE_C, stroke_width=4)
        self.place_in_area(neuron, 'B4', 'E5')
        
        # Inputs
        x1 = MathTex("x_1", color=WHITE_C)
        x2 = MathTex("x_2", color=WHITE_C)
        x3 = MathTex("x_3", color=WHITE_C)
        
        self.place_at_grid(x1, 'B2')
        self.place_at_grid(x2, 'C2')
        self.place_at_grid(x3, 'D2')
        
        # Arrows - pointing to the neuron circle boundary
        arrow1 = Arrow(x1.get_right(), neuron.get_left(), buff=0.1, color=WHITE_C)
        arrow2 = Arrow(x2.get_right(), neuron.get_left(), buff=0.1, color=WHITE_C)
        arrow3 = Arrow(x3.get_right(), neuron.get_left(), buff=0.1, color=WHITE_C)

        self.play(self.lecture[0].animate.set_color(WHITE_C))
        self.play(Create(neuron), Write(x1), Write(x2), Write(x3))
        self.play(Create(arrow1), Create(arrow2), Create(arrow3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Weights
        w1x1 = MathTex("w_1 x_1", color=YELLOW_C, font_size=30)
        w2x2 = MathTex("w_2 x_2", color=YELLOW_C, font_size=30)
        w3x3 = MathTex("w_3 x_3", color=YELLOW_C, font_size=30)
        
        # Position labels next to arrows
        w1x1.next_to(arrow1, UP, buff=0.05)
        w2x2.next_to(arrow2, UP, buff=0.05)
        w3x3.next_to(arrow3, UP, buff=0.05)

        self.play(self.lecture[1].animate.set_color(YELLOW_C))
        self.play(
            FadeIn(w1x1), FadeIn(w2x2), FadeIn(w3x3),
            arrow1.animate.set_color(YELLOW_C),
            arrow2.animate.set_color(YELLOW_C),
            arrow3.animate.set_color(YELLOW_C)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Summation formula
        sum_formula = MathTex(r"\sum w_i x_i + b", color=WHITE_C, font_size=36)
        sum_formula.move_to(neuron.get_center())

        self.play(self.lecture[2].animate.set_color(WHITE_C))
        self.play(Write(sum_formula))
        self.play(
            neuron.animate.scale(1.1).set_stroke(width=6),
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # ReLU activation graph over the sum
        axes = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-0.2, 1, 0.5],
            x_length=1.5,
            y_length=1.0,
            axis_config={"include_tip": False, "font_size": 14}
        ).set_color(GREEN_C)
        
        relu_graph = axes.plot(lambda x: np.maximum(0, x), x_range=[-1, 1], color=GREEN_C)
        relu_label = MathTex(r"\text{ReLU}", color=GREEN_C, font_size=24).next_to(axes, UP, buff=0.1)
        
        relu_group = VGroup(axes, relu_graph, relu_label)
        relu_group.move_to(neuron.get_center())

        self.play(self.lecture[3].animate.set_color(GREEN_C))
        self.play(FadeOut(sum_formula), FadeIn(relu_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Final signal pulse exits
        output_arrow = Arrow(neuron.get_right(), neuron.get_right() + RIGHT*1.5, color=WHITE_C)
        output_label = MathTex("y", color=WHITE_C).next_to(output_arrow, RIGHT)

        pulse = Dot(neuron.get_center(), color=WHITE_C).scale(1.5)
        
        self.play(self.lecture[4].animate.set_color(WHITE_C))
        self.play(Create(output_arrow), Write(output_label))
        self.play(
            pulse.animate.move_to(output_label.get_center()),
            rate_func=slow_into,
            run_time=1.5
        )
        self.play(FadeOut(pulse))
        self.wait(2)
