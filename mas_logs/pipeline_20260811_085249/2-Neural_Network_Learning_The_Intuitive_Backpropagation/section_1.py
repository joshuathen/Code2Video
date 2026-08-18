from manim import *

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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Neural networks act as decision-making chains.",
            "Neurons take inputs, multiplying them by weights.",
            "Weights represent the importance of each input.",
            "We sum these to get a signal.",
            "Think of weights as adjustable decision knobs."
        ]
        self.setup_layout("The Neuron Metaphor: Input & Weights", lecture_lines)
        
        # Setup Assets
        knob_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg"
        knob1 = SVGMobject(knob_path)
        knob2 = SVGMobject(knob_path)
        
        # Define objects
        neuron = Circle(radius=0.5, color=WHITE)
        input_label = Text("Input", font_size=20, color=WHITE)
        neuron_group = VGroup(neuron, knob1, input_label).arrange(DOWN)
        
        node = Circle(radius=0.5, color=WHITE)
        node_label = Text("Node", font_size=20, color=WHITE)
        sum_label = Text("Σ", font_size=30, color=WHITE)
        node_group = VGroup(node, sum_label, node_label).arrange(DOWN)
        
        # Combined group for critic request
        neuron_input_node_group = VGroup(neuron_group, node_group)
        
        # Positioning
        self.place_in_area(neuron_input_node_group, 'B3', 'E5', scale_factor=0.9)
        
        # Weights
        weight1 = Arrow(start=neuron.get_right(), end=node.get_left(), color="#FFD700")
        weight2 = Arrow(start=neuron.get_right() + DOWN*0.5, end=node.get_left(), color="#00CED1")
        w1_label = Text("w1", font_size=18, color="#FFD700")
        w2_label = Text("w2", font_size=18, color="#00CED1")
        
        # Fixes from VideoCritic
        self.place_at_grid(w1_label, 'B4', scale_factor=0.7)
        self.place_at_grid(w2_label, 'D4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), FadeIn(neuron_group))
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]), Create(weight1), Create(weight2), FadeIn(w1_label), FadeIn(w2_label))
        
        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]), FadeIn(node_group))
        
        # === Animation for Lecture Line 5 ===
        # Corrected: Use .animate to perform transformations
        self.play(
            FadeIn(self.lecture[4]), 
            knob2.animate.move_to(node.get_center()).set_color("#32CD32").scale(0.5)
        )
