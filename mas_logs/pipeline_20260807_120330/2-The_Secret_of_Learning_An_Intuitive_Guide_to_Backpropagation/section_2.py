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
        self.setup_layout(
            "Prerequisite: The Forward Flow",
            [
                "In the forward flow, data moves through layers.",
                "Inputs are weighted, summed, and activated to produce guesses.",
                "This guess is compared to reality to measure mistakes."
            ]
        )

        # Colors
        NEURON_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"
        OUTPUT_COLOR = "#FF4500"
        REALITY_COLOR = "#00FF00"
        CONNECTION_COLOR = GRAY_A

        # === Animation for Lecture Line 1 ===
        # Represent input data as a cat icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png] moving into the network.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Asset integration (Issue 21)
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        # Issue 27: place_at_grid(cat_icon, 'C2', scale_factor=0.8)
        self.place_at_grid(cat_icon, "C2", scale_factor=0.8)
        
        # Grid positions for the network
        # Issue 28: Move input neurons to B3, D3
        input_neurons = VGroup(
            Circle(radius=0.2, color=NEURON_COLOR),
            Circle(radius=0.2, color=NEURON_COLOR)
        )
        self.place_at_grid(input_neurons[0], "B3")
        self.place_at_grid(input_neurons[1], "D3")
        
        # Hidden Layer neurons (adjusted slightly to Column 4)
        hidden_neurons = VGroup(
            Circle(radius=0.2, color=NEURON_COLOR),
            Circle(radius=0.2, color=NEURON_COLOR),
            Circle(radius=0.2, color=NEURON_COLOR)
        )
        self.place_at_grid(hidden_neurons[0], "B4")
        self.place_at_grid(hidden_neurons[1], "C4")
        self.place_at_grid(hidden_neurons[2], "D4")
        
        # Output Layer: C6
        output_neuron = Circle(radius=0.2, color=NEURON_COLOR)
        self.place_at_grid(output_neuron, "C6")

        # Initial Network Display
        self.play(FadeIn(input_neurons), FadeIn(hidden_neurons), FadeIn(output_neuron))
        
        # Animation: cat icon moves into network
        self.play(FadeIn(cat_icon))
        self.play(cat_icon.animate.shift(RIGHT * 0.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show neurons lighting up sequentially from left to right (#FFFF00).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Define connections
        connections1 = VGroup()
        for i_n in input_neurons:
            for h_n in hidden_neurons:
                connections1.add(Line(i_n.get_right(), h_n.get_left(), stroke_width=1.5, color=CONNECTION_COLOR))
        
        connections2 = VGroup()
        for h_n in hidden_neurons:
            connections2.add(Line(h_n.get_right(), output_neuron.get_left(), stroke_width=1.5, color=CONNECTION_COLOR))

        # Animation Flow: Sequential lighting
        self.play(Create(connections1))
        self.play(
            input_neurons.animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        self.play(Create(connections2))
        self.play(
            hidden_neurons.animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        self.play(
            output_neuron.animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display final output 70% Dog (#FF4500) next to label Cat.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # Issue 29: Scaling guess and reality text to 0.6
        guess_text = Text("Guess: 70% Dog", font_size=20, color=OUTPUT_COLOR)
        reality_text = Text("Reality: Cat", font_size=20, color=REALITY_COLOR)
        
        # Position labels at B6 and D6 (1 unit away from output neuron at C6)
        self.place_at_grid(guess_text, "B6", scale_factor=0.6)
        self.place_at_grid(reality_text, "D6", scale_factor=0.6)
        
        self.play(Write(guess_text))
        self.wait(0.5)
        self.play(Write(reality_text))
        
        # Visualize mistake measurement
        comparison_indicator = DoubleArrow(
            guess_text.get_bottom() + DOWN*0.1, 
            reality_text.get_top() + UP*0.1, 
            color=RED, 
            stroke_width=2,
            tip_length=0.1
        )
        self.play(Create(comparison_indicator))
        
        self.wait(2)
