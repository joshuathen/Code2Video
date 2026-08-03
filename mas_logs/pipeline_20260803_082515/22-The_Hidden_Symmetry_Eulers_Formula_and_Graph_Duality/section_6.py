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
        # Data from shared state
        title = "Application: The Maze Runner's Secret"
        lecture_lines = [
            "Graph duality helps solve complex maze navigation problems.",
            "Dual edges represent boundaries or paths between regions.",
            "Understanding these connections reveals the maze's hidden structure."
        ]
        
        self.setup_layout(title, lecture_lines)

        # Colors from storyboard
        MAZE_COLOR = "#888888"
        DUAL_COLOR = "#0000FF"

        # === Pre-construction of elements for anchoring ===
        
        # Define maze walls relative to the grid
        maze = VGroup(
            Line(self.grid["B2"], self.grid["B5"], color=MAZE_COLOR),
            Line(self.grid["B5"], self.grid["E5"], color=MAZE_COLOR),
            Line(self.grid["E5"], self.grid["E2"], color=MAZE_COLOR),
            Line(self.grid["E2"], self.grid["B2"], color=MAZE_COLOR),
            Line(self.grid["B3"], self.grid["D3"], color=MAZE_COLOR),
            Line(self.grid["C4"], self.grid["E4"], color=MAZE_COLOR),
            Line(self.grid["C2"], self.grid["C3"], color=MAZE_COLOR),
            Line(self.grid["D4"], self.grid["D5"], color=MAZE_COLOR)
        )

        # Create dual vertices (dots) at cell centers
        def get_cell_center(tl, br):
            return (self.grid[tl] + self.grid[br]) / 2

        dots = VGroup(
            Dot(get_cell_center("B2", "C3"), color=DUAL_COLOR), # Cell 1
            Dot(get_cell_center("B3", "C4"), color=DUAL_COLOR), # Cell 2
            Dot(get_cell_center("B4", "C5"), color=DUAL_COLOR), # Cell 3
            Dot(get_cell_center("C2", "D3"), color=DUAL_COLOR), # Cell 4
            Dot(get_cell_center("C3", "D4"), color=DUAL_COLOR), # Cell 5
            Dot(get_cell_center("C4", "D5"), color=DUAL_COLOR), # Cell 6
            Dot(get_cell_center("D2", "E3"), color=DUAL_COLOR), # Cell 7
            Dot(get_cell_center("D3", "E4"), color=DUAL_COLOR), # Cell 8
            Dot(get_cell_center("D4", "E5"), color=DUAL_COLOR)  # Cell 9
        )
        
        # Initial edges connection based on initial centers
        edges = VGroup(
            Line(dots[0].get_center(), dots[3].get_center(), color=DUAL_COLOR),
            Line(dots[3].get_center(), dots[6].get_center(), color=DUAL_COLOR),
            Line(dots[6].get_center(), dots[7].get_center(), color=DUAL_COLOR),
            Line(dots[7].get_center(), dots[4].get_center(), color=DUAL_COLOR),
            Line(dots[4].get_center(), dots[1].get_center(), color=DUAL_COLOR),
            Line(dots[1].get_center(), dots[2].get_center(), color=DUAL_COLOR),
            Line(dots[2].get_center(), dots[3].get_center(), color=DUAL_COLOR), # Cross-path
            Line(dots[3].get_center(), dots[4].get_center(), color=DUAL_COLOR)  # Connect to path
        ).set_stroke(width=3)
        # Re-defining path to be more sequential for "solving" the maze
        path_edges = VGroup(
            Line(dots[0].get_center(), dots[3].get_center(), color=DUAL_COLOR),
            Line(dots[3].get_center(), dots[6].get_center(), color=DUAL_COLOR),
            Line(dots[6].get_center(), dots[7].get_center(), color=DUAL_COLOR),
            Line(dots[7].get_center(), dots[4].get_center(), color=DUAL_COLOR),
            Line(dots[4].get_center(), dots[1].get_center(), color=DUAL_COLOR),
            Line(dots[1].get_center(), dots[2].get_center(), color=DUAL_COLOR),
            Line(dots[2].get_center(), dots[5].get_center(), color=DUAL_COLOR),
            Line(dots[5].get_center(), dots[8].get_center(), color=DUAL_COLOR)
        ).set_stroke(width=3)

        # Fix Issue 31: Enhance dots visibility and centering
        self.place_in_area(dots, 'B2', 'E5', scale_factor=1.1)
        
        # Redefine path_edges based on adjusted dot positions
        path_edges = VGroup(
            Line(dots[0].get_center(), dots[3].get_center(), color=DUAL_COLOR),
            Line(dots[3].get_center(), dots[6].get_center(), color=DUAL_COLOR),
            Line(dots[6].get_center(), dots[7].get_center(), color=DUAL_COLOR),
            Line(dots[7].get_center(), dots[4].get_center(), color=DUAL_COLOR),
            Line(dots[4].get_center(), dots[1].get_center(), color=DUAL_COLOR),
            Line(dots[1].get_center(), dots[2].get_center(), color=DUAL_COLOR),
            Line(dots[2].get_center(), dots[5].get_center(), color=DUAL_COLOR),
            Line(dots[5].get_center(), dots[8].get_center(), color=DUAL_COLOR)
        )

        # Fix Issue 30: Group elements and anchor to grid area
        maze_group = VGroup(maze, dots, path_edges)
        self.place_in_area(maze_group, 'B1', 'F6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(MAZE_COLOR)
        self.play(Create(maze), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(DUAL_COLOR)
        self.play(Create(dots), run_time=1)
        self.play(Create(path_edges), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(DUAL_COLOR)
        # Highlight the solved path
        self.play(path_edges.animate.set_stroke(width=8), run_time=1)
        self.play(Indicate(path_edges, color=DUAL_COLOR, scale_factor=1.1), run_time=2)
        self.wait(2)

        # Final reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
