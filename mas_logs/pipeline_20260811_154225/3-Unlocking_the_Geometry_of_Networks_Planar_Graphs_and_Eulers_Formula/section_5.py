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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: Network Reliability", ["Dual graphs aid network reliability.", "Cutting the original blocks paths.", "This represents flow in duals."])
        
        # Load Assets
        maze = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg", color="#33FF57")
        network = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg", color="#FF5733")
        
        # Define Group for animation
        animation_group = VGroup(maze, network)
        self.place_in_area(animation_group, 'C3', 'D5', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(maze), self.lecture[0].animate.set_color("#33FF57"))
        
        # === Animation for Lecture Line 2 ===
        grid_label = Text("Dual Graph", font_size=20, color="#FF5733")
        self.place_at_grid(grid_label, 'A4', scale_factor=0.7)
        self.play(FadeIn(network), Write(grid_label), self.lecture[1].animate.set_color("#FF5733"))
        
        # === Animation for Lecture Line 3 ===
        # Represent flow in duals
        cut_highlight = maze.copy().set_color("#FF0000")
        path_highlight = network.copy().set_color("#FFFF00")
        flow_dot = Dot(color=YELLOW)
        self.place_in_area(flow_dot, 'D4', 'D4', scale_factor=0.5)
        
        self.play(
            self.lecture[2].animate.set_color(YELLOW),
            ReplacementTransform(maze.copy(), cut_highlight),
            ReplacementTransform(network.copy(), path_highlight),
            FadeIn(flow_dot)
        )
        self.play(flow_dot.animate.shift(RIGHT * 0.5))
        self.wait(2)
