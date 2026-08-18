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
            "Classical bits are limited to zero or one.",
            "Qubits explore multiple computational paths at the same time.",
            "This massive parallelism solves complex problems exponentially faster."
        ]
        self.setup_layout("Application: The Power of Parallelism", lecture_lines)
        
        # Paths for maze assets
        maze_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg"

        # === Animation for Lecture Line 1 ===
        # Show a single red dot (#FF0000) moving through a simple maze grid [Asset: maze.svg].
        self.lecture[0].set_color(YELLOW)
        
        maze_structure = SVGMobject(maze_asset_path, color=WHITE)
        self.place_in_area(maze_structure, "B2", "E5", scale_factor=0.8)
        
        start_label = Text("START", font_size=20)
        self.place_at_grid(start_label, "A2", scale_factor=0.5)
        
        exit_label = Text("EXIT", font_size=20)
        self.place_at_grid(exit_label, "F5", scale_factor=0.5)
        
        self.play(DrawBorderThenFill(maze_structure), Write(start_label), Write(exit_label))
        
        red_dot = Dot(color="#FF0000")
        self.place_at_grid(red_dot, "B2")
        
        self.play(FadeIn(red_dot))
        
        # Sequential path movement (simulating classical search)
        classical_path = ["B3", "C3", "C2", "C3", "B3", "B4", "B5", "B4", "B3", "D3", "D4", "E4", "E5"]
        for pos in classical_path:
            self.play(red_dot.animate.move_to(self.grid[pos]), run_time=0.15, rate_func=linear)
        
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Show a wave of blue dots (#0000FF) expanding to fill all maze paths [Asset: maze.svg] at once.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(FadeOut(red_dot))
        
        # Create multiple blue dots at start
        num_dots = 12
        blue_dots = VGroup(*[Dot(color="#0000FF", radius=0.08) for _ in range(num_dots)])
        for dot in blue_dots:
            dot.move_to(self.grid["B2"])
        
        self.play(FadeIn(blue_dots))
        
        # Define paths for the quantum expansion (all branches)
        target_positions = ["C2", "C3", "C4", "C5", "B4", "B5", "D2", "D3", "D4", "E3", "E4", "E5"]
        
        animations = []
        for i, pos in enumerate(target_positions):
            animations.append(blue_dots[i].animate.move_to(self.grid[pos]))
            
        self.play(*animations, run_time=1.5, rate_func=smooth)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the exit in gold (#FFD700) as the blue wave reaches it instantly within the maze [Asset: maze.svg].
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        exit_highlight = Circle(radius=0.4, color="#FFD700", stroke_width=4).move_to(self.grid["E5"])
        solved_label = Text("SOLVED!", font_size=20, color="#FFD700")
        self.place_at_grid(solved_label, "F6", scale_factor=0.6)
        
        # Target dot at exit (last one in the group)
        winning_dot = blue_dots[-1]
        
        self.play(
            Create(exit_highlight),
            Write(solved_label),
            winning_dot.animate.scale(2).set_color("#FFD700"),
            Flash(self.grid["E5"], color="#FFD700", flash_radius=0.5),
            run_time=1
        )
        
        self.wait(2)
