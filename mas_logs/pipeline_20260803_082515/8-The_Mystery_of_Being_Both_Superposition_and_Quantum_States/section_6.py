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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        self.setup_layout(
            "Application: Quantum Computing",
            [
                "Classical bits process one path at a time.",
                "Qubits explore every possible path simultaneously.",
                "This allows computers to solve complex problems instantly."
            ]
        )

        # Maze Setup
        # Using Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg
        maze = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg")
        maze.set_color(WHITE)
        self.place_in_area(maze, "B2", "E5", scale_factor=1.5)
        
        start_label = Text("START", font_size=18, color=WHITE)
        self.place_at_grid(start_label, 'F5', scale_factor=0.8) # Resolved Issue 47
        
        exit_label = Text("EXIT", font_size=18, color=WHITE)
        self.place_at_grid(exit_label, 'B6', scale_factor=0.8) # Resolved Issue 48

        self.add(maze, start_label, exit_label)

        # Define points for the classical path and quantum spread
        # Maze occupies roughly B2 to E5 area
        p_start = self.grid["E5"]
        path_points = [
            self.grid["E4"],
            self.grid["D4"],
            self.grid["D3"],
            self.grid["C3"],
            self.grid["C4"],
            self.grid["B4"],
            self.grid["B5"]
        ]
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)
        
        # White dot representing a classical bit
        white_dot = Dot(color=WHITE, radius=0.1)
        white_dot.move_to(p_start)
        self.add(white_dot)
        
        # Sequential path movement
        for pos in path_points:
            self.play(white_dot.animate.move_to(pos), run_time=0.4, rate_func=linear)
        
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Reset color and highlight second line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF") # Cyan-like for Qubits
        
        # Reset white dot for comparison
        self.play(white_dot.animate.move_to(p_start), run_time=0.5)

        # Create many blue dots to represent a superposition "gas"
        # We scatter them across the grid points inside the maze area
        blue_dots = VGroup(*[Dot(color="#00FFFF", radius=0.07) for _ in range(16)])
        for dot in blue_dots:
            dot.move_to(p_start)
        self.add(blue_dots)

        # Destinations for dots to simulate "filling" the maze
        scatter_points = [
            self.grid["B2"], self.grid["B3"], self.grid["B4"], self.grid["B5"],
            self.grid["C2"], self.grid["C3"], self.grid["C4"], self.grid["C5"],
            self.grid["D2"], self.grid["D3"], self.grid["D4"], self.grid["D5"],
            self.grid["E2"], self.grid["E3"], self.grid["E4"], self.grid["E5"]
        ]

        # Animate blue dots spreading simultaneously
        self.play(
            LaggedStart(
                *[dot.animate.move_to(dest) for dot, dest in zip(blue_dots, scatter_points)],
                lag_ratio=0.03
            ),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Reset color and highlight third line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        # The dot at the exit (B5) pulses
        exit_qubit = blue_dots[3] # Index 3 is B5 in the scatter_points list
        
        self.play(
            Indicate(exit_qubit, color=GREEN, scale_factor=2.0),
            white_dot.animate.move_to(path_points[0]),
            run_time=1.5
        )
        
        # Show classical dot still trying while quantum is done
        self.play(white_dot.animate.move_to(path_points[1]), run_time=0.5)
        
        self.wait(2)
