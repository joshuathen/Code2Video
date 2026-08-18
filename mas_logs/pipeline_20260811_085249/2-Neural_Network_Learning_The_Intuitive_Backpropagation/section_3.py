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
            "Backpropagation is a blame game.",
            "Distribute error back through the network.",
            "Adjust weights based on their contribution.",
            "We reduce error via gradient descent.",
            "The network learns from its mistakes."
        ])
        
        # Assets
        net_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg")
        node_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg")
        weight_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weight.svg")
        correction_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/correction.svg")

        # Network Diagram
        nodes = VGroup(
            self.place_at_grid(node_icon.copy(), "B2", 0.5),
            self.place_at_grid(node_icon.copy(), "B4", 0.5),
            self.place_at_grid(node_icon.copy(), "D3", 0.5)
        )
        weights = VGroup(
            Line(nodes[0].get_right(), nodes[2].get_left(), color=WHITE),
            Line(nodes[1].get_left(), nodes[2].get_right(), color=WHITE)
        )
        network_diagram = VGroup(nodes, weights)
        self.place_in_area(network_diagram, "A2", "C4", scale_factor=1.0)
        self.add(network_diagram)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6347"))
        pulse = self.place_at_grid(net_icon.copy(), "D3", 0.3)
        pulse.set_color("#FF6347")
        self.play(FadeIn(pulse), run_time=1)
        self.play(FadeOut(pulse))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        error_flow = self.place_at_grid(node_icon.copy(), "D3", 0.3)
        self.play(error_flow.animate.move_to(self.grid["B2"]), run_time=1)
        self.play(FadeOut(error_flow))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        weights[0].set_color("#FFD700")
        correction = self.place_at_grid(correction_icon.copy(), "C2", 0.2)
        self.play(Create(weights[0]), FadeIn(correction))
        self.play(weights[0].animate.set_color("#32CD32"), FadeOut(correction))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#32CD32"))
        dot = Dot(color=YELLOW)
        gradient_path = ArcBetweenPoints(self.grid["A6"], self.grid["F6"], angle=-TAU/12)
        self.place_in_area(gradient_path, "A4", "F5", scale_factor=0.8)
        self.play(Create(dot), Create(gradient_path))
        self.play(MoveAlongPath(dot, gradient_path), run_time=2)
        self.play(FadeOut(dot), FadeOut(gradient_path))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#32CD32"))
        self.wait(1)
