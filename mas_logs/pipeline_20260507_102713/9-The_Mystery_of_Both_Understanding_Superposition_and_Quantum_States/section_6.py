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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Classical bits are either zero or one, like switches.',
            'Qubits use superposition to be both states at once.',
            'Quantum computers solve complex problems exponentially faster.'
        ]
        self.setup_layout("Application: Quantum Computing Power", lecture_lines)
        
        # Initialize lecture colors (all gray first, then highlight)
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Description: A classical switch [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/switch.svg] flips between 0 and 1 (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        switch_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/switch.svg").set_color(WHITE)
        label_0 = Text("0", font_size=36, color=WHITE)
        label_1 = Text("1", font_size=36, color=WHITE)
        
        # Position labels relative to the SVG
        label_0.next_to(switch_svg, LEFT, buff=0.3)
        label_1.next_to(switch_svg, RIGHT, buff=0.3)
        
        switch_container = VGroup(switch_svg, label_0, label_1)
        
        # Grid positioning: 'A2' to 'C5', scale 0.8 (Issue 52/59)
        self.place_in_area(switch_container, "A2", "C5", scale_factor=0.8)
        
        self.play(FadeIn(switch_container))
        self.wait(0.5)
        
        # Simulation of flipping state
        self.play(switch_svg.animate.flip(UP), run_time=0.8)
        self.wait(0.5)
        self.play(switch_svg.animate.flip(UP), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: A green glowing Qubit (#00FF00) displays 0 and 1 overlapping.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#00FF00")
        )
        # Dim the switch to focus on the qubit (Issue 52/53 spatial distinction)
        self.play(switch_container.animate.set_color(GRAY))
        
        qubit_circle = Circle(radius=1.0, color="#00FF00", stroke_width=6)
        # Semi-transparent fill for glow
        qubit_glow = Circle(radius=1.2, color="#00FF00", fill_opacity=0.2, stroke_width=0)
        
        # Superposition labels
        label_sup_0 = Text("0", font_size=48, color="#00FF00").shift(LEFT * 0.2)
        label_sup_1 = Text("1", font_size=48, color="#00FF00").shift(RIGHT * 0.2)
        
        qubit_group = VGroup(qubit_circle, qubit_glow, label_sup_0, label_sup_1)
        
        # Grid positioning: 'D2' to 'F5', scale 0.8 (Issue 53/59)
        self.place_in_area(qubit_group, "D2", "F5", scale_factor=0.8)
        
        self.play(FadeIn(qubit_group))
        
        # Animated pulsing for superposition
        for _ in range(2):
            self.play(
                qubit_glow.animate.scale(1.2).set_opacity(0.4),
                label_sup_0.animate.set_opacity(0.5),
                label_sup_1.animate.set_opacity(1.0),
                run_time=0.8,
                rate_func=there_and_back
            )
            self.play(
                qubit_glow.animate.scale(1.2).set_opacity(0.4),
                label_sup_0.animate.set_opacity(1.0),
                label_sup_1.animate.set_opacity(0.5),
                run_time=0.8,
                rate_func=there_and_back
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Maze [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/maze.svg] paths highlight simultaneously in glowing yellow (#FFFF00).
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FFFF00"),
            FadeOut(switch_container),
            FadeOut(qubit_group)
        )
        
        maze_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/maze.svg").set_color(GRAY_E)
        
        # Centering the maze in the visualization area
        self.place_in_area(maze_svg, "A2", "F5", scale_factor=1.0)
        
        self.play(FadeIn(maze_svg))
        self.wait(0.5)
        
        # Quantum exploration: highlight maze paths simultaneously in yellow
        quantum_highlight = maze_svg.copy().set_color("#FFFF00").set_stroke(width=4)
        quantum_glow = maze_svg.copy().set_color("#FFFF00").set_stroke(width=10).set_opacity(0.2)
        
        self.play(
            Create(quantum_highlight),
            FadeIn(quantum_glow),
            run_time=2
        )
        
        # Final glow pulse
        self.play(
            quantum_glow.animate.scale(1.1).set_opacity(0.4),
            rate_func=there_and_back,
            run_time=1.5
        )

        self.wait(3)
