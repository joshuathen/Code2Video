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
        # Configuration
        COLOR_KEYS = "#ADD8E6"  # Light Blue
        COLOR_INPUT = "#FFFFE0" # Light Yellow
        COLOR_PULSE = "#00FFFF" # Cyan
        COLOR_NEUTRAL = "#888888"
        COLOR_VALUE = "#90EE90"  # Light Green
        
        title_text = "The First Layer: The Key Detectors"
        lecture_lines = [
            "The MLP consists of two main linear layers.",
            "The first layer acts as a set of \"Keys\".",
            "Each neuron in this layer detects a specific concept.",
            "When input matches a Key, the neuron activates strongly.",
            "This activation signals that a specific fact was found."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Assets
        # Eiffel icon for input vector
        try:
            eiffel_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eiffel.svg", color=COLOR_INPUT)
        except:
            eiffel_svg = Triangle(color=COLOR_INPUT, fill_opacity=1).scale(0.15)

        # Tower icon for neuron concept
        try:
            tower_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg", color=COLOR_KEYS)
        except:
            tower_svg = Square(color=COLOR_KEYS, fill_opacity=1).scale(0.15)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))

        # Create two layers of neurons
        layer1_neurons = VGroup(*[Circle(radius=0.15, color=COLOR_NEUTRAL, stroke_width=2) for _ in range(5)])
        layer2_neurons = VGroup(*[Circle(radius=0.15, color=COLOR_NEUTRAL, stroke_width=2) for _ in range(5)])
        
        # Position neurons in columns 3 and 5
        rows_labels = ["B", "C", "D", "E", "F"]
        for i, row in enumerate(rows_labels):
            self.place_at_grid(layer1_neurons[i], f"{row}3")
            self.place_at_grid(layer2_neurons[i], f"{row}5")

        # Create connections (MLP-like)
        connections = VGroup()
        for n1 in layer1_neurons:
            for n2 in layer2_neurons:
                line = Line(n1.get_right(), n2.get_left(), stroke_width=0.5, color=GREY_E, stroke_opacity=0.3)
                connections.add(line)

        self.play(
            Create(layer1_neurons), 
            Create(layer2_neurons), 
            Create(connections), 
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(COLOR_KEYS)
        )

        # Label first layer as "Keys"
        keys_label = Text("Keys", font_size=20, color=COLOR_KEYS)
        self.place_at_grid(keys_label, "A3")
        
        self.play(
            Write(keys_label),
            layer1_neurons.animate.set_color(COLOR_KEYS),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(COLOR_KEYS)
        )

        # Target neuron is D3 (3rd neuron in layer 1)
        target_neuron = layer1_neurons[2]
        
        # Issue 33: Place concept label near the neuron
        concept_text = Text("French Landmarks", font_size=16, color=COLOR_KEYS)
        self.place_at_grid(concept_text, "D3")
        concept_text.shift(RIGHT * 1.5) # Shifted right to avoid covering neuron

        # Issue 32: Place tower icon at D1
        tower_icon = tower_svg.copy().scale(0.5)
        self.place_at_grid(tower_icon, "D1")

        self.play(
            Write(concept_text),
            FadeIn(tower_icon),
            target_neuron.animate.set_fill(COLOR_KEYS, opacity=0.4),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(COLOR_INPUT)
        )

        # Create input vector "Eiffel Tower"
        # Starting at B1 to avoid overlap with tower_icon at D1
        input_label = Text("Eiffel Tower", font_size=16, color=COLOR_INPUT)
        eiffel_icon = eiffel_svg.copy().scale(0.5)
        input_vector_group = VGroup(eiffel_icon, input_label).arrange(RIGHT, buff=0.1)
        self.place_at_grid(input_vector_group, "B1")

        self.play(FadeIn(input_vector_group))
        
        # Move input vector to the target neuron D3
        self.play(input_vector_group.animate.move_to(self.grid["D3"]), run_time=1.5)

        # Cyan Pulse on contact
        pulse = Circle(radius=0.15, color=COLOR_PULSE, stroke_width=4).move_to(self.grid["D3"])
        self.play(
            pulse.animate.scale(3).set_style(stroke_opacity=0),
            target_neuron.animate.set_fill(COLOR_PULSE, opacity=0.8).set_color(COLOR_PULSE),
            FadeOut(input_vector_group),
            run_time=1
        )
        self.remove(pulse)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(COLOR_PULSE)
        )

        # Issue 34: Add "Match Found!" label at C3
        match_label = Text("Match Found!", font_size=18, color=COLOR_PULSE)
        self.place_at_grid(match_label, "C3")

        # Show activation signaling (Arrow to Value layer neuron at D5)
        activation_arrow = Arrow(self.grid["D3"], self.grid["D5"], color=COLOR_PULSE, buff=0.2)
        
        self.play(
            FadeIn(match_label),
            GrowArrow(activation_arrow),
            layer2_neurons[2].animate.set_stroke(color=COLOR_VALUE, width=4),
            run_time=1
        )
        
        self.wait(2)
