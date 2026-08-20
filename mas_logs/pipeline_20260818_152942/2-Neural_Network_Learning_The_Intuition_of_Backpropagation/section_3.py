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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Intuition of Backpropagation", [
            "Backpropagation assigns blame for the error.",
            "We start adjusting from the output layer.",
            "We work backward through each layer.",
            "Weights are tuned to reduce total loss.",
            "This refines the model's overall prediction."
        ])
        
        # Create a simple feed-forward network representation
        layers = [3, 4, 2]
        network = VGroup()
        for i, count in enumerate(layers):
            layer = VGroup()
            for j in range(count):
                dot = Dot(color="#00FFFF")
                layer.add(dot)
            layer.arrange(DOWN, buff=0.4)
            network.add(layer)
        network.arrange(RIGHT, buff=1.5)
        
        # Positioning based on feedback (using final requested parameters)
        self.place_in_area(network, "B4", "E6", scale_factor=0.5)
        
        # Draw edges (simplified)
        edges = VGroup()
        for i in range(len(network)-1):
            for node1 in network[i]:
                for node2 in network[i+1]:
                    edge = Line(node1.get_center(), node2.get_center(), stroke_width=1, color=WHITE)
                    edges.add(edge)
        self.add(edges)
        self.add(network)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Highlight path from output to input
        path = VGroup(network[2], network[1], network[0])
        self.play(Indicate(path, color="#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        # Signal flow backward animation
        signal = Dot(color=YELLOW)
        for i in range(len(network)-1, 0, -1):
            for node in network[i]:
                self.add(signal)
                self.play(signal.animate.move_to(node.get_center()), run_time=0.2)
        self.remove(signal)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.play(edges.animate.set_stroke(color="#00FF00", width=2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFA500"))
        self.play(network.animate.set_color(WHITE))
        self.wait(2)
