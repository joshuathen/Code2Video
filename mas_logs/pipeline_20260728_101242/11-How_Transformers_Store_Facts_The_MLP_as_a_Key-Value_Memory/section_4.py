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
        title = "The Second Layer: The Value Retrievers"
        lecture_lines = [
            "The second layer contains the \"Value\" vectors.",
            "These values represent the actual factual content stored.",
            "Activated neurons from the first layer pull their Values.",
            "The retrieved Value is added to the hidden state.",
            "This updates the model's understanding with the retrieved fact."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Key Neuron (from previous step, now glowing)
        key_neuron = Circle(radius=0.4, color=BLUE, fill_opacity=0.3)
        self.place_at_grid(key_neuron, "C2")
        
        # Outer glow for the Key Neuron
        glow = Circle(radius=0.45, color=BLUE, stroke_width=8, stroke_opacity=0.5)
        glow.move_to(key_neuron.get_center())
        
        # Value Vector
        # Adjusted end to C3 to make room for Paris label at C4 (Issue 35)
        value_vector = Arrow(
            start=self.grid["C2"],
            end=self.grid["C3"],
            buff=0.4,
            color="#90EE90",
            stroke_width=6
        )
        
        self.play(Create(key_neuron), Create(glow), run_time=1)
        self.play(GrowArrow(value_vector), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Label the Value vector 'Paris' with SVG (Issue 28)
        # Position at C4 (Issue 35)
        paris_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paris.svg")
        paris_icon.set_color("#90EE90")
        
        paris_text = Text("Paris", font_size=18, color="#90EE90")
        paris_label = VGroup(paris_icon, paris_text).arrange(DOWN, buff=0.1)
        self.place_at_grid(paris_label, "C4", scale_factor=0.6)
        
        self.play(FadeIn(paris_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Pulling effect: Pulse moves from Key along the Value vector
        pulse = Dot(color=YELLOW).move_to(key_neuron.get_center())
        self.play(
            pulse.animate.move_to(value_vector.get_end()),
            glow.animate.scale(1.2).set_stroke(opacity=0.8),
            run_time=1.5,
            rate_func=slow_into
        )
        self.play(FadeOut(pulse), glow.animate.scale(1/1.2).set_stroke(opacity=0.5), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Equation: Output = Activation * Value
        equation = MathTex(
            "Output", "=", "Activation", "\\times", "Value",
            color=WHITE,
            font_size=36
        )
        # Position equation in area E2 to E5 (Issue 36)
        self.place_in_area(equation, "E2", "E5")
        
        # Highlight 'Value' in the equation to match the vector color
        equation[4].set_color("#90EE90")
        
        self.play(Write(equation), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Final visual: Flash the 'Paris' label and the 'Output' part of the equation
        self.play(
            Indicate(paris_label, color="#90EE90", scale_factor=1.2),
            Indicate(equation[0], color=YELLOW, scale_factor=1.2),
            run_time=2
        )
        self.wait(2)
        
        # Cleanup colors
        self.lecture[4].set_color(WHITE)
        self.wait(1)
