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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: Why Duality Matters", [
            "Duality simplifies complex geometric problems.",
            "Problems about faces become problems about vertices.",
            "This symmetry reveals the deep structure of networks."
        ])

        # Colors
        WHITE_COLOR = "#FFFFFF"
        MAGENTA_COLOR = "#FF00FF"
        YELLOW_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Show a complex maze [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg] 
        # representing a "Face" problem. Color the walls (edges) #FFFFFF.
        self.lecture[0].set_color(WHITE_COLOR)
        
        # Load and place maze asset
        maze = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg")
        maze.set_color(WHITE_COLOR)
        # Position maze on the right side, shifting slightly right to accommodate text
        self.place_in_area(maze, "B3", "E6", scale_factor=1.8)
            
        self.play(Create(maze), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform the maze into its dual graph, where "Rooms" become "Vertices" 
        # and "Passages" become "Edges" in #FF00FF.
        self.lecture[1].set_color(MAGENTA_COLOR)

        # Dual Vertices positioned to avoid crowding (Issues 28, 29, 30)
        dual_positions = {
            # Row 1 shifted: 'B3'-'C4', 'B4'-'C5', 'B5'-'C6'
            "R11": self.place_in_area(Dot(radius=0.01), "B3", "C4").get_center(),
            "R12": self.place_in_area(Dot(radius=0.01), "B4", "C5").get_center(),
            "R13": self.place_in_area(Dot(radius=0.01), "B5", "C6").get_center(),
            # Row 2 shifted: 'C3'-'D4', 'C4'-'D5', 'C5'-'D6'
            "R21": self.place_in_area(Dot(radius=0.01), "C3", "D4").get_center(),
            "R22": self.place_in_area(Dot(radius=0.01), "C4", "D5").get_center(),
            "R23": self.place_in_area(Dot(radius=0.01), "C5", "D6").get_center(),
            # Row 3 shifted: 'D3'-'E4', 'D4'-'E5', 'D5'-'E6'
            "R31": self.place_in_area(Dot(radius=0.01), "D3", "E4").get_center(),
            "R32": self.place_in_area(Dot(radius=0.01), "D4", "E5").get_center(),
            "R33": self.place_in_area(Dot(radius=0.01), "D5", "E6").get_center(),
            # External node centered: 'A4'-'B5'
            "EXT": self.place_in_area(Dot(radius=0.01), "A4", "B5").get_center()
        }
        
        dual_vertices = VGroup(*[
            Dot(pos, color=MAGENTA_COLOR, radius=0.08) 
            for pos in dual_positions.values()
        ])

        # Dual Edges (connecting rooms that share a passage)
        dual_edges = VGroup()
        edge_config = [
            ("EXT", "R11"), ("R11", "R21"), ("R21", "R31"), 
            ("R31", "R32"), ("R32", "R22"), ("R22", "R12"),
            ("R12", "R13"), ("R13", "R23"), ("R23", "R33")
        ]
        
        for start_key, end_key in edge_config:
            edge = Line(dual_positions[start_key], dual_positions[end_key], color=MAGENTA_COLOR, stroke_width=3)
            dual_edges.add(edge)

        self.play(
            maze.animate.set_opacity(0.3),
            Create(dual_vertices),
            run_time=1.5
        )
        self.play(Create(dual_edges), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate a simple path through the dual vertices to demonstrate solving the problem in the dual space.
        self.lecture[2].set_color(YELLOW_COLOR)

        # Define a solution path through the dual graph
        path_keys = ["EXT", "R11", "R21", "R31", "R32", "R33"]
        path_points = [dual_positions[k] for k in path_keys]
        
        solver_path = VMobject(color=YELLOW_COLOR, stroke_width=6)
        solver_path.set_points_as_corners(path_points)
        
        robot_logic_dot = Dot(path_points[0], color=YELLOW_COLOR, radius=0.12)
        
        self.play(FadeIn(robot_logic_dot))
        self.play(
            MoveAlongPath(robot_logic_dot, solver_path),
            Create(solver_path),
            run_time=4,
            rate_func=linear
        )
        
        self.wait(2)
