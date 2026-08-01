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
        title = "Anatomy of an MLP: The Two-Step Mechanism"
        lines = [
            "Input vectors interact with the first matrix, W1.",
            "W1 acts as a set of specific pattern detectors.",
            "A non-linear activation determines if a pattern matches.",
            "If matched, the second matrix, W2, provides information.",
            "W2 injects a \"value\" vector back into the model."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        W1_COLOR = "#00FFFF"
        W2_COLOR = "#FF00FF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Replaced MathTex with VGroup of Text to avoid 'latex' dependency error
        formula_parts = ["y", "=", "ReLU(", "x", "·", "W1", ")", "·", "W2"]
        formula = VGroup(*[Text(p, font_size=36) for p in formula_parts]).arrange(RIGHT, buff=0.1)
        formula[5].set_color(W1_COLOR) # W1
        formula[8].set_color(W2_COLOR) # W2
        
        self.place_in_area(formula, "A2", "A5")
        
        self.play(self.lecture[0].animate.set_color(W1_COLOR))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # W1 matrix rows as detectors
        key_box = Rectangle(width=1.2, height=1.5, color=W1_COLOR, fill_opacity=0.2)
        key_label = Text("W1 (Keys)", font_size=18, color=W1_COLOR)
        key_label.next_to(key_box, UP, buff=0.2)
        key_group = VGroup(key_box, key_label)
        self.place_at_grid(key_group, "C1", scale_factor=0.6)

        detector_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/detector.svg", color=WHITE)
        self.place_at_grid(detector_icon, "C2", scale_factor=0.4)

        neuron_group = VGroup(*[Circle(radius=0.2, color=WHITE) for _ in range(4)]).arrange(DOWN, buff=0.2)
        self.place_at_grid(neuron_group, "C3", scale_factor=0.7)

        self.play(self.lecture[1].animate.set_color(W1_COLOR))
        self.play(FadeIn(key_group), FadeIn(detector_icon))
        self.play(Create(neuron_group))
        
        # Glow effect for detectors
        glow_circles = neuron_group.copy().set_color(W1_COLOR).set_stroke(width=8)
        self.play(
            glow_circles.animate.scale(1.2).set_opacity(0),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # ReLU graph
        axes = Axes(x_range=[-2, 2, 1], y_range=[-1, 2, 1], axis_config={"include_tip": False}).scale(0.3)
        relu_plot = axes.plot(lambda x: max(0, x), x_range=[-2, 2], color=WHITE)
        relu_label = Text("ReLU", font_size=16).next_to(axes, DOWN, buff=0.1)
        relu_group = VGroup(axes, relu_plot, relu_label)
        self.place_at_grid(relu_group, "D3", scale_factor=0.8)

        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Create(relu_group))
        
        # Block signals animation
        block_line = Line(start=axes.c2p(-1.5, -0.5), end=axes.c2p(0, 0), color=RED)
        pass_line = Line(start=axes.c2p(0, 0), end=axes.c2p(1.5, 1.5), color=GREEN)
        self.play(Create(block_line), Create(pass_line))
        self.play(FadeOut(block_line), FadeOut(pass_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # W2 matrix columns as knowledge retrieval
        value_box = Rectangle(width=1.2, height=1.5, color=W2_COLOR, fill_opacity=0.2)
        value_label = Text("W2 (Values)", font_size=18, color=W2_COLOR)
        value_label.next_to(value_box, UP, buff=0.2)
        value_group = VGroup(value_box, value_label)
        self.place_at_grid(value_group, "C5", scale_factor=0.6)

        flashlight_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/flashlight.svg", color=WHITE)
        self.place_at_grid(flashlight_icon, "C4", scale_factor=0.4)

        self.play(self.lecture[3].animate.set_color(W2_COLOR))
        self.play(FadeIn(value_group), FadeIn(flashlight_icon))
        
        # Arrow from detectors to values
        arrow = Arrow(neuron_group.get_right(), value_box.get_left(), color=WHITE)
        self.play(GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Visual sum: Scaling the retrieved Value
        final_arrow = Arrow(value_box.get_right(), self.grid["C6"], color=W2_COLOR)
        value_vector = Vector([0, 1, 0], color=W2_COLOR)
        self.place_at_grid(value_vector, "C6", scale_factor=0.5)

        self.play(self.lecture[4].animate.set_color(W2_COLOR))
        self.play(GrowArrow(final_arrow))
        self.play(
            value_group.animate.set_color(GOLD),
            value_vector.animate.scale(1.5).set_color(GOLD),
            Indicate(formula[8]) # W2 in formula (9th element of VGroup)
        )
        
        self.wait(2)
