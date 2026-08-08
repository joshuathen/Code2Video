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
        lecture_lines = [
            "Data moves forward through the network layers.",
            "Matrix operations and functions process this input data.",
            "The result is a final prediction score."
        ]
        self.setup_layout("The Forward Pass: Making a Guess", lecture_lines)
        
        # Assets
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color=WHITE)
        chip_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chip.svg", color=WHITE)
        
        # Setup visual elements
        input_node = Circle(radius=0.3, color=WHITE, fill_opacity=0.5)
        hidden_nodes = VGroup(*[Circle(radius=0.25, color=WHITE, fill_opacity=0.5) for _ in range(3)])
        output_node = Circle(radius=0.3, color=WHITE, fill_opacity=0.5)
        
        # Positioning as requested by VideoCritic
        self.place_at_grid(input_node, 'B3', scale_factor=0.6)
        for i, node in enumerate(hidden_nodes):
            self.place_at_grid(node, f"{['B', 'C', 'D'][i]}4")
        self.place_at_grid(output_node, 'B5', scale_factor=0.6)
        
        edges = VGroup()
        for h in hidden_nodes:
            edges.add(Line(input_node.get_center(), h.get_center(), color=WHITE))
        for h in hidden_nodes:
            edges.add(Line(h.get_center(), output_node.get_center(), color=WHITE))
        
        network_group = VGroup(edges, input_node, hidden_nodes, output_node, server_icon)
        self.place_in_area(network_group, 'B3', 'D5', scale_factor=0.7)
        self.place_at_grid(server_icon, 'A3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(server_icon), FadeIn(input_node), FadeIn(hidden_nodes), FadeIn(output_node), Create(edges))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        data_flow = Dot(color="#00FF00")
        self.play(MoveAlongPath(data_flow, Line(input_node.get_center(), hidden_nodes[1].get_center())), run_time=1.5)
        self.play(MoveAlongPath(data_flow, Line(hidden_nodes[1].get_center(), output_node.get_center())), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        prediction = Text("0.6", font_size=24, color="#FFFF00")
        self.place_at_grid(chip_icon, 'C6', scale_factor=0.5)
        prediction.next_to(output_node, RIGHT)
        self.play(Write(prediction), FadeIn(chip_icon))
        self.wait(1)
