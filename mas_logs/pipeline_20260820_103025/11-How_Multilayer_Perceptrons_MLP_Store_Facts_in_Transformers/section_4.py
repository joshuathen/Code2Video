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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Inputs activate dedicated fact neurons.", "Information projects to output space.", "Weights allow surgical model editing."]
        self.setup_layout("The Memory Retrieval Process", lecture_lines)
        
        # Define semantic colors
        INPUT_COLOR = "#ADD8E6"
        NEURON_COLOR = "#FFD700"
        
        # Assets
        input_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/token.svg").set_color(INPUT_COLOR)
        neuron_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg").set_color(NEURON_COLOR)
        
        # Mobjects
        input_circle = Circle(radius=0.4, color=INPUT_COLOR, fill_opacity=0.3).add(input_icon)
        self.place_at_grid(input_circle, 'D4', scale_factor=0.6)
        
        fact_neurons_group = VGroup(*[Circle(radius=0.2, color=NEURON_COLOR, fill_opacity=0.5) for _ in range(3)])
        self.place_in_area(fact_neurons_group, 'B2', 'C3', scale_factor=0.7)
        
        output_space_node = neuron_icon
        self.place_at_grid(output_space_node, 'F4', scale_factor=0.6)
        
        # Labels (tethered to objects)
        input_label = Text("Input", font_size=20, color=INPUT_COLOR).scale(0.75)
        input_label.next_to(input_circle, DOWN, buff=0.1)
        
        neuron_label = Text("Fact Neurons", font_size=20, color=NEURON_COLOR).scale(0.75)
        neuron_label.next_to(fact_neurons_group, UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(INPUT_COLOR)
        self.play(FadeIn(input_circle), Write(input_label))
        self.play(input_circle.animate.set_fill(INPUT_COLOR, opacity=0.8), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(NEURON_COLOR)
        self.play(FadeIn(fact_neurons_group), Write(neuron_label))
        self.play(input_circle.animate.move_to(fact_neurons_group.get_center()))
        self.play(FadeIn(output_space_node))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        self.play(Indicate(fact_neurons_group, color=WHITE))
        self.wait(1)
