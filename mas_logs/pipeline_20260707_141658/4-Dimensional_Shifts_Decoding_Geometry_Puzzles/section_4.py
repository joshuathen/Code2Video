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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Unfolding Puzzle: Nets and Topography"
        lecture_lines = [
            "Nets flatten 3D surfaces into 2D paper maps.",
            "Folding these maps reconstructs the original geometric solid.",
            "Many patterns exist, but only some form a cube."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        pink_color = "#FF69B4"
        
        # === Animation for Lecture Line 1 ===
        # Color matching: Pink text for pink net
        self.lecture[0].set_color(pink_color)
        
        # T-shaped net of six squares
        sq_size = 0.6
        s3 = Square(side_length=sq_size, fill_opacity=0.5, fill_color=pink_color, stroke_color=WHITE)
        s1 = s3.copy().next_to(s3, UP, buff=0)
        s2 = s3.copy().next_to(s3, LEFT, buff=0)
        s4 = s3.copy().next_to(s3, RIGHT, buff=0)
        s5 = s3.copy().next_to(s3, DOWN, buff=0)
        s6 = s3.copy().next_to(s5, DOWN, buff=0)
        
        net = VGroup(s1, s2, s3, s4, s5, s6)
        self.place_in_area(net, "B2", "E5", scale_factor=1.0)
        
        self.play(Create(net))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(pink_color)
        
        # Function to create an isometric cube in 2D
        def get_isometric_cube(side=0.6, color=pink_color):
            # Top face
            f1 = Polygon(
                [0, 0, 0], 
                [side * np.cos(30*DEGREES), side * np.sin(30*DEGREES), 0],
                [0, 2 * side * np.sin(30*DEGREES), 0],
                [-side * np.cos(30*DEGREES), side * np.sin(30*DEGREES), 0],
                fill_opacity=0.8, fill_color=color, stroke_color=WHITE
            )
            # Front-left face
            f2 = Polygon(
                [0, 0, 0],
                [-side * np.cos(30*DEGREES), side * np.sin(30*DEGREES), 0],
                [-side * np.cos(30*DEGREES), side * np.sin(30*DEGREES) - side, 0],
                [0, -side, 0],
                fill_opacity=0.7, fill_color=color, stroke_color=WHITE
            )
            # Front-right face
            f3 = Polygon(
                [0, 0, 0],
                [side * np.cos(30*DEGREES), side * np.sin(30*DEGREES), 0],
                [side * np.cos(30*DEGREES), side * np.sin(30*DEGREES) - side, 0],
                [0, -side, 0],
                fill_opacity=0.6, fill_color=color, stroke_color=WHITE
            )
            cube = VGroup(f1, f2, f3)
            return cube

        cube_2d = get_isometric_cube(side=sq_size, color=pink_color)
        # Position cube in the same area
        # Issue 32 fix: scale_factor changed from 1.5 to 1.1
        self.place_in_area(cube_2d, "B2", "E5", scale_factor=1.1)
        
        # Transform the net squares into the isometric cube
        self.play(
            ReplacementTransform(net, cube_2d),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(pink_color)
        
        # Transition to faulty net
        self.play(FadeOut(cube_2d))
        
        # Faulty net arrangement: 5 in a row + 1 on the side
        # [1][2][3][4][5]
        #       [6]
        sq_row = [Square(side_length=sq_size, fill_opacity=0.5, fill_color=pink_color, stroke_color=WHITE) for _ in range(5)]
        faulty_row = VGroup(*sq_row).arrange(RIGHT, buff=0)
        fs6 = Square(side_length=sq_size, fill_opacity=0.5, fill_color=pink_color, stroke_color=WHITE)
        fs6.next_to(faulty_row[2], DOWN, buff=0)
        
        faulty_net = VGroup(faulty_row, fs6)
        # Issue 31 fix: position from B1-E6 to B2-E5, scale from 0.9 to 0.8
        self.place_in_area(faulty_net, "B2", "E5", scale_factor=0.8)
        
        self.play(Create(faulty_net))
        self.wait(1)
        
        # Highlight overlap
        # In a 5-in-row fold, the 1st and 5th squares would overlap.
        overlap_label = Text("OVERLAP!", color=RED, font_size=24)
        # Issue 33 fix: added scale_factor=0.8
        self.place_at_grid(overlap_label, "F4", scale_factor=0.8)
        
        self.play(
            faulty_row[0].animate.set_fill(RED, opacity=0.8),
            faulty_row[4].animate.set_fill(RED, opacity=0.8),
            Write(overlap_label),
            faulty_row.animate.set_stroke(RED)
        )
        self.wait(2)
        
        # Wrap up
        self.lecture[2].set_color(WHITE)
        self.play(FadeOut(faulty_net), FadeOut(overlap_label))
        self.wait(1)
