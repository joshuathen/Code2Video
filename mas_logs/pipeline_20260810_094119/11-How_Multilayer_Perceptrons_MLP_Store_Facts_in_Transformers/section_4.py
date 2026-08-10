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
        self.setup_layout("Visualization: The Fact-Retrieval Process", [
            "Information flows through the network layers.",
            "Weights activate to match patterns.",
            "Facts emerge from these weight connections.",
            "Retrieval is a distributed process.",
            "Not one neuron holds the fact."
        ])
        
        # Asset Loading
        neuron_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg"
        
        # Create components
        input_node = SVGMobject(neuron_asset, color=BLUE).scale(0.3)
        mlp_group = VGroup(*[SVGMobject(neuron_asset, color=GREY_B).scale(0.15) for _ in range(12)])
        mlp_group.arrange_in_grid(3, 4, buff=0.2)
        output_node = SVGMobject(neuron_asset, color=GREEN).scale(0.3)
        
        chain = VGroup(input_node, mlp_group, output_node).arrange(RIGHT, buff=0.8)
        
        # Applying layout per Critic feedback (Issue 32)
        self.place_in_area(chain, 'B3', 'E5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(chain))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        pulses = VGroup(*[Dot(color=YELLOW, radius=0.05).move_to(mlp_group[i]) for i in [0, 5, 11]])
        self.play(FadeIn(pulses))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        connections = VGroup(*[Line(input_node.get_right(), mlp_group[i].get_left(), color=RED, stroke_width=2) for i in [0, 5, 11]])
        self.play(Create(connections))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(PURPLE)
        connections2 = VGroup(*[Line(mlp_group[i].get_right(), output_node.get_left(), color=PURPLE, stroke_width=2) for i in range(12)])
        self.play(Create(connections2), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(ORANGE)
        self.play(Indicate(mlp_group))
        self.wait(1)
