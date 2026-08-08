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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Introduction: The Perspective Problem"
        lines = [
            "One point can have many different coordinate descriptions.",
            "An explorer uses a standard North-East grid.",
            "A pirate uses a tilted treasure map grid.",
            "They see the same \"X\" using different numbers.",
            "Coordinates depend entirely on your chosen basis."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_STANDARD = WHITE
        COLOR_PIRATE = YELLOW
        COLOR_X = RED
        
        # Core positions in the 6x6 system
        origin_pos = self.grid['D2']
        x_pos = self.grid['B5']  # Point "X" at (3, 2) relative to D2
        
        # === Animation for Lecture Line 1 ===
        # "One point can have many different coordinate descriptions."
        self.play(self.lecture[0].animate.set_color(COLOR_X))
        
        # Draw a red 'X'
        point_x = Cross(stroke_width=6).scale(0.2).set_color(COLOR_X)
        point_x.move_to(x_pos)
        point_x.set_z_index(10)
        
        self.play(Create(point_x))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "An explorer uses a standard North-East grid."
        self.play(self.lecture[1].animate.set_color(COLOR_STANDARD))
        
        # Standard white grid
        # x_range from -1 to 5, y_range from -1 to 4 ensures origin (0,0) is shifted
        # x_length and y_length match the number of units in range (1 unit per grid cell)
        standard_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_color": COLOR_STANDARD, "stroke_opacity": 0.3},
            axis_config={"stroke_color": COLOR_STANDARD, "stroke_width": 2},
            x_length=6,
            y_length=5
        )
        # Shift origin to D2
        standard_grid.shift(origin_pos - standard_grid.get_origin())
        
        explorer_label = Text("Explorer", font_size=24, color=COLOR_STANDARD)
        # Fixed: Move from A1 to A2 per VideoCritic (Issue 27)
        self.place_at_grid(explorer_label, 'A2', scale_factor=0.8)
        
        self.play(Create(standard_grid), Write(explorer_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A pirate uses a tilted treasure map grid."
        self.play(self.lecture[2].animate.set_color(COLOR_PIRATE))
        
        # Tilted yellow grid
        pirate_grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_color": COLOR_PIRATE, "stroke_opacity": 0.3},
            axis_config={"stroke_color": COLOR_PIRATE, "stroke_width": 2},
            x_length=6,
            y_length=5
        )
        # Transformation matrix: tilted and stretched
        matrix = [[1.2, -0.3], [0.5, 1.1]]
        pirate_grid.apply_matrix(matrix)
        # Shift origin to D2
        pirate_grid.shift(origin_pos - pirate_grid.get_origin())
        
        pirate_label = Text("Pirate", font_size=24, color=COLOR_PIRATE)
        # Fixed: Move from A3 to A5 per VideoCritic (Issue 28)
        self.place_at_grid(pirate_label, 'A5', scale_factor=0.8)
        
        self.play(Create(pirate_grid), Write(pirate_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "They see the same \"X\" using different numbers."
        self.play(self.lecture[3].animate.set_color(COLOR_X))
        
        # Standard projection lines (Horizontal/Vertical)
        std_proj_x = DashedLine(x_pos, [x_pos[0], origin_pos[1], 0], color=COLOR_STANDARD)
        std_proj_y = DashedLine(x_pos, [origin_pos[0], x_pos[1], 0], color=COLOR_STANDARD)
        
        # Pirate projection lines (Along basis vectors)
        # Basis vectors: b1 = [1.2, 0.5], b2 = [-0.3, 1.1]
        p_b1 = np.array([1.2, 0.5, 0])
        p_b2 = np.array([-0.3, 1.1, 0])
        c1, c2 = 2.65, 0.61 # Derived coefficients for (3,2) in Pirate basis
        
        pirate_proj_1 = DashedLine(x_pos, origin_pos + c1*p_b1, color=COLOR_PIRATE)
        pirate_proj_2 = DashedLine(x_pos, origin_pos + c2*p_b2, color=COLOR_PIRATE)
        
        self.play(Create(std_proj_x), Create(std_proj_y))
        self.play(Create(pirate_proj_1), Create(pirate_proj_2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Coordinates depend entirely on your chosen basis."
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        std_coords = MathTex(r"\text{Standard: } \begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=COLOR_STANDARD)
        pirate_coords = MathTex(r"\text{Pirate: } \begin{bmatrix} 2.65 \\ 0.61 \end{bmatrix}", color=COLOR_PIRATE)
        
        # Fixed: Move from F row to E row per VideoCritic (Issue 29)
        self.place_in_area(std_coords, 'E1', 'E3', scale_factor=0.7)
        self.place_in_area(pirate_coords, 'E4', 'E6', scale_factor=0.7)
        
        self.play(Write(std_coords), Write(pirate_coords))
        self.wait(2)
