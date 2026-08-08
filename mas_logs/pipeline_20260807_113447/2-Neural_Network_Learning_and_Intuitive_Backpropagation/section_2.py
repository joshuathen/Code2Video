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
        title = "Prerequisite: The Anatomy of a Neuron"
        lines = [
            "Weights determine the strength of a connection.",
            "Biases set the threshold for a neuron firing.",
            "Thick lines represent higher mathematical weights."
        ]
        self.setup_layout(title, lines)

        # Pre-define colors for lines
        color_w = "#FFFF00"  # Yellow for Weights
        color_b = "#00FFFF"  # Cyan for Biases
        color_t = "#00FF00"  # Green for Thickness/Pulse

        # === Animation for Lecture Line 1 ===
        # "Weights determine the strength of a connection."
        self.play(self.lecture[0].animate.set_color(color_w))
        
        # Create Nodes using Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg
        neuron_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg"
        node_input = SVGMobject(neuron_asset, color=WHITE)
        node_output = SVGMobject(neuron_asset, color=WHITE)
        self.place_at_grid(node_input, "B2", scale_factor=0.5)
        self.place_at_grid(node_output, "B5", scale_factor=0.5)
        
        # Labels for nodes
        label_input = Text("Input", font_size=20, color=WHITE)
        label_output = Text("Output", font_size=20, color=WHITE)
        self.place_at_grid(label_input, "A2")
        self.place_at_grid(label_output, "A5")
        
        # Connection line
        connection = Line(node_input.get_right(), node_output.get_left(), color=color_w, stroke_width=2)
        
        # Weight label (Issue 39: Fix positioning and scale)
        label_weight = Text("Weight", font_size=18, color=color_w)
        self.place_at_grid(label_weight, 'A3', scale_factor=0.8)

        self.play(
            DrawBorderThenFill(node_input),
            DrawBorderThenFill(node_output),
            Write(label_input),
            Write(label_output)
        )
        self.play(
            Create(connection),
            Write(label_weight)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Biases set the threshold for a neuron firing."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_b)
        )
        
        # Bias label at the output node
        label_bias = Text("Bias", font_size=18, color=color_b)
        self.place_at_grid(label_bias, "C5") # Below output node
        
        # Symbol for bias (Issue 40: Fix scale to avoid cramped look)
        bias_symbol = MathTex("+b", font_size=24, color=color_b)
        self.place_at_grid(bias_symbol, 'B5', scale_factor=0.6)

        self.play(
            Write(label_bias),
            node_output.animate.set_color(color_b),
            Write(bias_symbol)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Thick lines represent higher mathematical weights."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_t)
        )
        
        # Animate connection thickness changing
        self.play(
            connection.animate.set_stroke_width(12).set_color(color_t),
            label_weight.animate.set_color(color_t)
        )
        
        # A pulse of light travels from Input to Output
        pulse = Dot(point=node_input.get_center(), radius=0.15, color=WHITE)
        pulse.set_z_index(connection.z_index + 1)
        
        # Add glow effect to pulse
        glow = Dot(point=node_input.get_center(), radius=0.3, color=WHITE, fill_opacity=0.3)
        pulse_group = VGroup(pulse, glow)

        self.play(
            MoveAlongPath(pulse_group, connection),
            run_time=1.5,
            rate_func=linear
        )
        self.play(FadeOut(pulse_group))
        
        self.wait(2)

        # Reset colors for final state
        self.play(
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
