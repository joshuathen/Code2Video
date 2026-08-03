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
        self.setup_layout("Quantum vs. Classical: The Maze Solver", [
            "Classical computers process information one step at a time.",
            "Quantum computers explore all possible paths simultaneously.",
            "Superposition enables massive parallel processing for complex problems."
        ])

        # Define maze nodes and paths based on issues 28, 29, 30
        # Starting point shifted to B2 to avoid overlap with lecture notes
        # Exit point shifted to E6 to avoid edge cramping
        path_dead_coords = ["B2", "A2", "A3", "A4", "B4", "B3"]
        path_correct_coords = ["B2", "C2", "C3", "D3", "D4", "D5", "E5", "E6"]
        
        p_dead = VMobject()
        p_dead.set_points_as_corners([self.grid[c] for c in path_dead_coords])
        
        p_correct = VMobject()
        p_correct.set_points_as_corners([self.grid[c] for c in path_correct_coords])
        
        # Maze visual construction
        maze_lines = VGroup()
        for i in range(len(path_dead_coords)-1):
            maze_lines.add(Line(self.grid[path_dead_coords[i]], self.grid[path_dead_coords[i+1]], color=WHITE, stroke_width=2))
        for i in range(len(path_correct_coords)-1):
            maze_lines.add(Line(self.grid[path_correct_coords[i]], self.grid[path_correct_coords[i+1]], color=WHITE, stroke_width=2))
        
        exit_marker = Triangle(color=YELLOW, fill_opacity=1).scale(0.15).rotate(-PI/2)
        self.place_at_grid(exit_marker, "E6") 
        exit_marker.shift(RIGHT*0.3)
        exit_label = Text("EXIT", font_size=14, color=YELLOW)
        exit_label.next_to(exit_marker, UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        # Classical computers process information one step at a time.
        self.lecture[0].set_color(RED)
        self.play(Create(maze_lines), FadeIn(exit_marker), Write(exit_label))
        
        red_dot = Dot(color=RED).scale(0.7)
        self.place_at_grid(red_dot, "B2") 
        self.play(FadeIn(red_dot))
        
        # Classical search: tries dead end first
        self.play(MoveAlongPath(red_dot, p_dead), run_time=2.5, rate_func=linear)
        # Backtracks
        p_dead_rev = p_dead.copy().reverse_points()
        self.play(MoveAlongPath(red_dot, p_dead_rev), run_time=1.5, rate_func=linear)
        # Starts correct path but slowly
        p_correct_start = VMobject().set_points_as_corners([self.grid[c] for c in path_correct_coords[:4]])
        self.play(MoveAlongPath(red_dot, p_correct_start), run_time=1, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        # Quantum computers explore all possible paths simultaneously.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # Quantum dots enter
        blue_dot_1 = Dot(color=BLUE).scale(0.7)
        blue_dot_2 = Dot(color=BLUE).scale(0.7)
        self.place_at_grid(blue_dot_1, "B2") 
        self.place_at_grid(blue_dot_2, "B2") 
        self.play(FadeIn(blue_dot_1), FadeIn(blue_dot_2))
        
        # Simultaneous exploration
        # Red dot continues slowly along path_correct
        # It was at index 3 (D3), now moves to index 5 (D5)
        p_red_slow = VMobject().set_points_as_corners([self.grid[c] for c in path_correct_coords[3:6]])
        
        self.play(
            MoveAlongPath(blue_dot_1, p_dead),
            MoveAlongPath(blue_dot_2, p_correct),
            MoveAlongPath(red_dot, p_red_slow),
            run_time=4,
            rate_func=linear
        )

        # === Animation for Lecture Line 3 ===
        # Superposition enables massive parallel processing for complex problems.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        # blue_dot_2 reached the exit, red_dot is at D5 (middle of path)
        self.play(Indicate(blue_dot_2, color=BLUE, scale_factor=1.5), run_time=1)
        self.play(Flash(blue_dot_2, color=BLUE, line_length=0.2), run_time=1)
        
        # Red dot still moving from D5 to E6
        p_red_finish = VMobject().set_points_as_corners([self.grid[c] for c in path_correct_coords[5:]])
        self.play(MoveAlongPath(red_dot, p_red_finish), run_time=3, rate_func=linear)
        
        self.play(Indicate(red_dot, color=RED), run_time=1)
        self.wait(2)
