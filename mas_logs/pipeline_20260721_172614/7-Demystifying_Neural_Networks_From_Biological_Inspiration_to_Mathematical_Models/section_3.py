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
        # Data from shared state
        title = "The Mathematical Engine: Summation and Bias"
        lecture_lines = [
            "The neuron multiplies each input by its corresponding weight.",
            "It sums these products together into a single value.",
            "A bias term is added to adjust the activation threshold.",
            "Bias represents how much signal is needed to fire.",
            "This calculation produces the neuron's raw mathematical output."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        NODE_COLOR = "#3498DB"
        WEIGHT_COLOR = "#E74C3C"
        BIAS_COLOR = "#FFD700"
        EQUATION_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Central Node - Fix Issue 28: area B3 to D5
        node = Circle(radius=1.2, color=NODE_COLOR, fill_opacity=0.1)
        self.place_in_area(node, "B3", "D5")
        
        # Inputs and Weights - Fix Issue 28: x1 at B2, x2 at D2
        x1 = MathTex("x_1", color=WHITE)
        x2 = MathTex("x_2", color=WHITE)
        w1 = MathTex("w_1", color=WEIGHT_COLOR)
        w2 = MathTex("w_2", color=WEIGHT_COLOR)
        
        self.place_at_grid(x1, "B2")
        self.place_at_grid(x2, "D2")
        
        # Connections
        edge1 = Line(x1.get_right(), node.get_left(), buff=0.1)
        edge2 = Line(x2.get_right(), node.get_left(), buff=0.1)
        
        w1.next_to(edge1, UP, buff=0.1)
        w2.next_to(edge2, DOWN, buff=0.1)
        
        self.play(Create(node), FadeIn(x1, x2), Create(edge1), Create(edge2))
        self.play(Write(w1), Write(w2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Summation equation inside node - Fix Issue 29: area B3 to D5, scale 0.7
        sum_eq = MathTex("z = \\sum(w \\cdot x)", color=EQUATION_COLOR)
        self.place_in_area(sum_eq, "B3", "D5", scale_factor=0.7)
        
        self.play(Write(sum_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Add + b to the equation - Fix Issue 29: area B3 to D5, scale 0.7
        bias_eq = MathTex("z = \\sum(w \\cdot x) + b", color=EQUATION_COLOR)
        bias_eq.set_color_by_tex("b", BIAS_COLOR)
        self.place_in_area(bias_eq, "B3", "D5", scale_factor=0.7)
        
        # Vertical Slider - Positioned in Col 6 to stay clear of node and notes
        slider_track = Line(self.grid["B6"], self.grid["C6"], color=GRAY)
        slider_knob = Dot(color=BIAS_COLOR)
        slider_knob.move_to(slider_track.get_center())
        slider_label = Text("Bias Slider", font_size=12, color=BIAS_COLOR).next_to(slider_track, LEFT, buff=0.1)
        
        # Small Graph
        axes = Axes(x_range=[-1, 1], y_range=[-1, 1], x_length=1.5, y_length=1.5, axis_config={"include_tip": False}).scale(0.7)
        self.place_at_grid(axes, "E5")
        
        bias_tracker = ValueTracker(0)
        
        # Persistent mobject with updater for graph line
        line_visual = Line(
            axes.c2p(-0.5, -0.5),
            axes.c2p(0.5, 0.5),
            color=BIAS_COLOR
        )
        def line_updater(obj):
            val = bias_tracker.get_value()
            obj.put_start_and_end_on(
                axes.c2p(-0.5, -0.5 + val),
                axes.c2p(0.5, 0.5 + val)
            )
        line_visual.add_updater(line_updater)
        
        # Slider knob updater
        slider_knob.add_updater(lambda m: m.move_to(
            interpolate(slider_track.get_start(), slider_track.get_end(), (bias_tracker.get_value() + 0.5))
        ))

        self.play(Transform(sum_eq, bias_eq))
        self.play(Create(slider_track), FadeIn(slider_knob), FadeIn(slider_label), Create(axes), Create(line_visual))
        self.play(bias_tracker.animate.set_value(0.5), run_time=1)
        self.play(bias_tracker.animate.set_value(-0.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # 'Master Volume' knob turns - Fix Issue 30: grid D6, scale 0.8
        knob_circle = Circle(radius=0.3, color=GRAY, fill_opacity=0.2)
        knob_line = Line(ORIGIN, UP * 0.3, color=BIAS_COLOR)
        knob = VGroup(knob_circle, knob_line)
        self.place_at_grid(knob, "D6", scale_factor=0.8)
        knob_label = Text("Master Bias", font_size=14, color=BIAS_COLOR).next_to(knob, DOWN)
        
        def knob_updater(m):
            angle = interpolate(-PI/2, PI/2, (bias_tracker.get_value() + 0.5))
            m[1].set_angle(angle + PI/2)
        knob.add_updater(knob_updater)
        
        self.play(FadeIn(knob), Write(knob_label))
        self.play(bias_tracker.animate.set_value(0.3), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Final output z
        # Adjust out_arrow to start from node and end near the text z
        out_arrow = Arrow(node.get_right(), self.grid["E6"], color=WHITE)
        out_text = MathTex("z", color=EQUATION_COLOR).next_to(out_arrow, RIGHT)
        
        self.play(Create(out_arrow), Write(out_text))
        self.play(Indicate(sum_eq)) 
        
        self.wait(2)
