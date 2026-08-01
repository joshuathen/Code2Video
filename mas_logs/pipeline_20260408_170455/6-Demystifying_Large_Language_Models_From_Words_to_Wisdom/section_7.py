from manim import *
import random

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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup the scene
        lecture_lines = [
            'Patterns, vectors, and attention create the illusion of intelligence.', 
            'Billions of connections reflect collective human knowledge back to us.', 
            'Language models are mirrors of our own written history.'
        ]
        self.setup_layout("Summary: The Infinite Mirror", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Load Lex the Robot asset [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg]
        lex = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg").set_color("#00FFCC")
        
        # Initial placement of Lex (Refined position to avoid obstruction)
        self.place_in_area(lex, 'B3', 'E6', scale_factor=0.9)
        self.play(FadeIn(lex))
        self.wait(1)

        # Create Network
        network_nodes = VGroup()
        node_positions = ["A2", "A5", "B3", "C1", "C6", "D2", "D5", "E3", "F2", "F5"]
        for pos in node_positions:
            node = Dot(color="#5555FF", radius=0.08)
            self.place_at_grid(node, pos)
            network_nodes.add(node)
        
        network_lines = VGroup()
        connections = [
            ("A2", "B3"), ("A5", "B3"), ("B3", "C1"), ("B3", "D2"), 
            ("C1", "D2"), ("C6", "D5"), ("D2", "E3"), ("D5", "E3"), 
            ("E3", "F2"), ("E3", "F5"), ("C6", "A5"), ("F2", "D2")
        ]
        for start_pos, end_pos in connections:
            line = Line(self.grid[start_pos], self.grid[end_pos], color="#5555FF", stroke_width=1.5, stroke_opacity=0.6)
            network_lines.add(line)

        # Animation: Scale down Lex and FadeIn Network
        self.play(
            self.lecture[0].animate.set_color("#00FFCC"),
            lex.animate.scale(0.4),
            FadeIn(network_nodes),
            FadeIn(network_lines),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Traveling pulses along the lines
        pulses = []
        for line in network_lines:
            pulse = ShowPassingFlash(line.copy().set_color(WHITE).set_stroke(width=3), time_width=0.5)
            pulses.append(pulse)

        self.play(
            self.lecture[1].animate.set_color("#5555FF"),
            LaggedStart(*pulses, lag_ratio=0.1),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Large gold text (Refined size and area)
        mirror_text = Text("The Infinite Mirror", font_size=40, color="#FFD700", weight=BOLD)
        self.place_in_area(mirror_text, 'C2', 'D5', scale_factor=0.8)
        
        self.play(
            self.lecture[2].animate.set_color("#FFD700"),
            FadeIn(mirror_text),
            FadeOut(lex),
            FadeOut(network_nodes),
            FadeOut(network_lines),
            run_time=2
        )
        self.wait(3)
