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
        # Setup layout with title and lecture lines
        lecture_lines = [
            "Duality simplifies complex layouts in circuit board design.",
            "Mazes use duality to define walls and paths.",
            "If walls form a tree, the path always exits."
        ]
        self.setup_layout("Application: Mazes and Circuitry", lecture_lines)

        # Assets
        maze_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg"
        circuit_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/circuit.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Load and place maze walls
        maze_walls = SVGMobject(maze_asset_path)
        maze_walls.set_color("#0000FF")
        self.place_in_area(maze_walls, "A2", "F6", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(maze_walls), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Load and place dual graph (circuit)
        dual_graph = SVGMobject(circuit_asset_path)
        dual_graph.set_color("#FFFF00")
        self.place_in_area(dual_graph, "A2", "F6", scale_factor=0.8)
        
        # Transition from walls to dual graph paths
        self.play(FadeIn(dual_graph), maze_walls.animate.set_stroke(opacity=0.3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Traverser (Yellow Dot) - representing the exit path
        traverser_dot = Dot(color="#FFFF00")
        # Scale as per issue 29
        self.place_at_grid(traverser_dot, "B2", scale_factor=0.3)
        
        # Path sequence for the traverser based on the grid
        path_points = [
            self.grid["B2"],
            self.grid["B5"],
            self.grid["E5"],
            self.grid["E3"],
            self.grid["F3"],
            self.grid["F6"]
        ]
        
        self.play(FadeIn(traverser_dot))
        
        # Animate movement along the path
        for target_pos in path_points[1:]:
            self.play(traverser_dot.animate.move_to(target_pos), run_time=0.6, rate_func=linear)
            
        self.wait(2)
