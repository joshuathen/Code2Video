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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Hilbert curves maintain local spatial relationships.",
            "Points near on curve remain near in 2D.",
            "This locality property optimizes data storage mapping.",
            "A robot gardener minimizes travel with this path.",
            "Complexity emerges from simple recursive rules."
        ]
        self.setup_layout("The Hilbert Curve & Locality", lecture_lines)
        
        # Create objects
        grid_square = Square(side_length=3.2, color=WHITE)
        label = Text("Hilbert Curve", font_size=24, color=WHITE)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(grid_square, 'B2', 'D4', scale_factor=1.0)
        self.place_at_grid(label, 'B2', scale_factor=0.9)
        self.place_at_grid(robot, 'A6', scale_factor=0.5)
        self.play(Create(grid_square), Write(label), FadeIn(robot))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        # Define Hilbert path (simplistic version)
        path = VGroup(
            Line(grid_square.get_corner(DL), grid_square.get_corner(UL), color="#00FF00"),
            Line(grid_square.get_corner(UL), grid_square.get_corner(UR), color="#00FF00"),
            Line(grid_square.get_corner(UR), grid_square.get_corner(DR), color="#00FF00")
        )
        self.play(Create(path))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        index_label = Text("Index Mapping: 1D to 2D", font_size=20, color="#FFFF00")
        self.place_at_grid(index_label, 'E3', scale_factor=0.8)
        self.play(Write(index_label), robot.animate.move_to(self.grid['E4']))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        self.wait(1)
