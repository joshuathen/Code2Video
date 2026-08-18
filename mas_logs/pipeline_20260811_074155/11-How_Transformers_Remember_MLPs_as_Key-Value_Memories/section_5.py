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
        self.setup_layout("Step-by-Step: Extracting a Fact", [
            "An input token vector enters the MLP layer.",
            "It is compared against all stored \"Key\" vectors.",
            "A \"Knowledge Neuron\" activates when it detects a match.",
            "Non-matching noise is filtered out by activation functions.",
            "The corresponding \"Value\" vector enriches the token's meaning."
        ])

        # === Animation for Lecture Line 1 ===
        # Use BLUE for the W1 matrix
        self.lecture[0].set_color(BLUE)
        
        token_vec = Text("[Elon Musk]", font_size=24, color=WHITE)
        self.place_at_grid(token_vec, 'A3') # Issue 40: Moved to A3
        
        w1_matrix = Rectangle(height=2, width=1.5, color=BLUE, fill_opacity=0.1)
        w1_label = MathTex("W_1", color=BLUE, font_size=30)
        w1_group = VGroup(w1_matrix, w1_label)
        self.place_in_area(w1_group, 'B3', 'C4') # Issue 40: Moved to B3-C4
        
        self.play(FadeIn(token_vec), Create(w1_group))
        # Move token_vec into the matrix area
        self.play(token_vec.animate.move_to(self.grid['B3']).scale(0.8))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Use BLUE_A for Key vectors
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE_A)
        
        key_lines = VGroup(*[Line(LEFT*0.5, RIGHT*0.5, color=BLUE_A) for _ in range(5)]).arrange(DOWN, buff=0.2)
        self.place_in_area(key_lines, 'B3', 'C4', scale_factor=0.8) # Issue 41: Moved to B3-C4
        
        scanner = Line(LEFT*0.7, RIGHT*0.7, color=WHITE, stroke_width=4).set_opacity(0.5)
        scanner.move_to(key_lines[0].get_center())
        
        self.play(FadeIn(key_lines))
        self.play(scanner.animate.move_to(key_lines[-1].get_center()), run_time=2, rate_func=linear)
        self.play(FadeOut(scanner))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use GREEN for Knowledge Neuron
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        neurons = VGroup(*[Circle(radius=0.15, color=WHITE) for _ in range(5)]).arrange(DOWN, buff=0.3)
        neuron_labels = VGroup(*[Text(f"#{i}", font_size=12) for i in [1022, 1023, 1024, 1025, 1026]]).arrange(DOWN, buff=0.45)
        self.place_at_grid(neurons, 'D4') # Issue 42: Moved to D4
        self.place_at_grid(neuron_labels, 'D5') # Issue 42: Moved to D5
        
        neuron_1024 = neurons[2]
        self.play(FadeIn(neurons), FadeIn(neuron_labels))
        self.play(neuron_1024.animate.set_fill("#00FF00", opacity=1).set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Use RED for ReLU
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        axes = Axes(x_range=[-1, 2, 1], y_range=[-0.5, 2, 1], axis_config={"include_tip": False}).scale(0.3)
        relu_plot = axes.plot(lambda x: max(0, x), x_range=[-1, 2], color=RED)
        relu_label = MathTex(r"\sigma(x)", font_size=24, color=RED)
        relu_group = VGroup(axes, relu_plot, relu_label)
        self.place_in_area(relu_group, 'E4', 'F6')
        
        dot = Dot(axes.c2p(-1, 0), color=WHITE)
        
        self.play(FadeIn(relu_group), FadeIn(dot))
        self.play(dot.animate.move_to(axes.c2p(1.5, 1.5)), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Use PINK for Value vector
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#EE82EE")
        
        value_vec = Text("[CEO of Tesla, SpaceX]", font_size=18, color="#EE82EE")
        # Start emerging from neuron 1024
        value_vec.scale(0.8)
        value_vec.move_to(neuron_1024.get_center())
        
        enriched_vec = Text("[Elon Musk, CEO of Tesla...]", font_size=20, color=GREEN)
        self.place_at_grid(enriched_vec, 'A5')
        
        self.play(FadeIn(value_vec))
        self.play(value_vec.animate.move_to(self.grid['C5']))
        self.play(
            ReplacementTransform(VGroup(token_vec, value_vec), enriched_vec),
            neuron_1024.animate.set_fill(opacity=0.3),
            FadeOut(relu_group),
            FadeOut(dot),
            FadeOut(neurons),
            FadeOut(neuron_labels),
            FadeOut(key_lines),
            FadeOut(w1_group)
        )
        self.wait(2)

        # Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
