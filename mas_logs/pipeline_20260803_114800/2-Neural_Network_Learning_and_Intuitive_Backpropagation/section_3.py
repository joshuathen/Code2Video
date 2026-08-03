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
        title = "The Forward Pass: Making a Guess"
        lecture_lines = [
            "Information flows forward through layers of neurons.",
            "Each neuron calculates a weighted sum of inputs.",
            "This sum passes through an activation function.",
            "The network produces a final prediction.",
            "We compare this prediction to the ground truth."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors for highlights
        COLOR_FLOW = "#FFFF00"
        COLOR_SUM = "#00FFFF"
        COLOR_SIGMOID = "#FFFFFF"
        COLOR_PRED = "#FF69B4"
        COLOR_TRUTH = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Information flows forward through layers of neurons.
        # Input values 0.8 and 0.2 enter the input circles.
        self.lecture[0].set_color(COLOR_FLOW)
        
        neuron_in1 = Circle(radius=0.4, color=WHITE)
        neuron_in2 = Circle(radius=0.4, color=WHITE)
        self.place_at_grid(neuron_in1, "B2")
        self.place_at_grid(neuron_in2, "D2")
        
        label_in1 = Text("Input 1", font_size=18, color=WHITE)
        label_in2 = Text("Input 2", font_size=18, color=WHITE)
        self.place_at_grid(label_in1, "A2")
        self.place_at_grid(label_in2, "E2")
        
        val_in1 = Text("0.8", font_size=24, color=COLOR_FLOW)
        val_in2 = Text("0.2", font_size=24, color=COLOR_FLOW)
        
        # Start values from Column 1
        val_in1.move_to(self.grid["B1"])
        val_in2.move_to(self.grid["D1"])
        
        self.play(
            Create(neuron_in1), Create(neuron_in2),
            Write(label_in1), Write(label_in2)
        )
        self.play(
            val_in1.animate.move_to(self.grid["B2"]),
            val_in2.animate.move_to(self.grid["D2"]),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each neuron calculates a weighted sum of inputs.
        # Multiplication signs appear on weight lines as data flows.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_FLOW) # Align with description (weighted sum)
        
        neuron_hidden = Circle(radius=0.5, color=WHITE)
        self.place_at_grid(neuron_hidden, "C4")
        
        line1 = Line(self.grid["B2"], self.grid["C4"], color=GRAY, buff=0.4)
        line2 = Line(self.grid["D2"], self.grid["C4"], color=GRAY, buff=0.4)
        
        mult1 = Text("×", font_size=30, color=YELLOW)
        mult2 = Text("×", font_size=30, color=YELLOW)
        self.place_in_area(mult1, "B2", "C4", scale_factor=0.8)
        self.place_in_area(mult2, "D2", "C4", scale_factor=0.8)
        
        self.play(Create(line1), Create(line2))
        self.play(Create(neuron_hidden))
        self.play(FadeIn(mult1), FadeIn(mult2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This sum passes through an activation function.
        # Summation symbol #00FFFF glows inside the central neuron.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_SUM)
        
        sum_sym = MathTex(r"\sum", color=COLOR_SUM).scale(1.2)
        self.place_at_grid(sum_sym, "C4")
        
        self.play(Write(sum_sym))
        self.play(Indicate(sum_sym, color=COLOR_SUM))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The network produces a final prediction.
        # An 'S' shaped Sigmoid gate #FFFFFF filters the output.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_SIGMOID)
        
        # Sigmoid gate visualization (inside neuron C4)
        sigmoid_ax = Axes(x_range=[-2, 2], y_range=[0, 1], x_length=0.6, y_length=0.4, 
                         axis_config={"include_ticks": False, "stroke_width": 1})
        sigmoid_curve = sigmoid_ax.plot(lambda x: 1 / (1 + np.exp(-x)), color=WHITE, x_range=[-2, 2])
        sigmoid_gate = VGroup(sigmoid_ax, sigmoid_curve)
        self.place_at_grid(sigmoid_gate, "C4", scale_factor=0.8) # Issue 28: placed at C4
        
        line_out = Line(self.grid["C4"], self.grid["C5"], color=GRAY, buff=0.5)
        
        pred_val = Text("0.7", font_size=30, color=COLOR_PRED)
        self.place_at_grid(pred_val, "C5") # Issue 29: placed at C5
        
        self.play(FadeOut(sum_sym))
        self.play(FadeIn(sigmoid_gate))
        self.play(Create(line_out))
        self.play(Write(pred_val))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We compare this prediction to the ground truth.
        # Final prediction value 0.7 appears at the network exit.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_TRUTH)
        
        # Ground truth comparison
        truth_val = Text("Target: 1.0", font_size=22, color=COLOR_TRUTH)
        self.place_at_grid(truth_val, "E5") # Issue 30: placed at E5
        
        # Error line between C5 (pred) and E5 (truth)
        error_line = DashedLine(self.grid["C5"], self.grid["E5"], color=RED)
        
        self.play(FadeIn(truth_val))
        self.play(Create(error_line))
        self.wait(2)
