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
        # Setup background and lecture lines
        self.setup_layout(
            "Why Order Matters (Non-Commutativity)", 
            [
                'In matrix math, the order of actions matters.', 
                'Rotating then shearing leads to a specific destination.', 
                'Shearing then rotating ends in a completely different spot.', 
                'Since BA and AB differ, multiplication is non-commutative.', 
                'Visualizing these paths proves order is everything.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Create two small grids (NumberPlanes) for the right side
        # Use Text for labels to avoid LaTeX dependency
        # Issue 38: Move planes to Row B to avoid title overlap
        plane_left = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1], 
            x_length=2.5, y_length=2.5,
            background_line_style={"stroke_color": BLUE_D, "stroke_width": 1, "stroke_opacity": 0.5}
        ).add_coordinates(label_constructor=Text)
        plane_right = plane_left.copy()
        
        self.place_in_area(plane_left, 'B1', 'D3', scale_factor=0.8)
        self.place_in_area(plane_right, 'B4', 'D6', scale_factor=0.8)
        
        # Issue 37: Scale labels and place at Row A to avoid overlap with tick marks
        label_left = Text("B then A", font_size=20, color=WHITE)
        label_right = Text("A then B", font_size=20, color=WHITE)
        self.place_at_grid(label_left, 'A2', scale_factor=0.8)
        self.place_at_grid(label_right, 'A5', scale_factor=0.8)
        
        # Issue 26: Asset integration - Robot icons
        # Robots: Start at (1, 1) in their respective planes
        robot_left = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        robot_left.set_color(GREEN).set_height(0.4)
        robot_left.move_to(plane_left.c2p(1, 1))

        robot_right = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        robot_right.set_color(TEAL).set_height(0.4)
        robot_right.move_to(plane_right.c2p(1, 1))
        
        self.add(plane_left, plane_right, label_left, label_right, robot_left, robot_right)
        self.play(
            FadeIn(plane_left), FadeIn(plane_right), 
            FadeIn(label_left), FadeIn(label_right),
            FadeIn(robot_left), FadeIn(robot_right)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        
        # Path 1: Rotate 90 degrees then Horizontal Shear
        shear_matrix = [[1, 1], [0, 1]]
        
        self.play(
            Rotate(robot_left, angle=PI/2, about_point=plane_left.c2p(0, 0)),
            run_time=1.5
        )
        self.play(
            robot_left.animate.apply_matrix(shear_matrix, about_point=plane_left.c2p(0, 0)),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(TEAL)
        
        # Path 2: Horizontal Shear then Rotate 90 degrees
        self.play(
            robot_right.animate.apply_matrix(shear_matrix, about_point=plane_right.c2p(0, 0)),
            run_time=1.5
        )
        self.play(
            Rotate(robot_right, angle=PI/2, about_point=plane_right.c2p(0, 0)),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED_B)
        
        # Highlight different final positions with red 'X'
        x_mark_left = Text("X", color="#FF0000", font_size=36)
        x_mark_right = Text("X", color="#FF0000", font_size=36)
        
        x_mark_left.move_to(robot_left.get_center())
        x_mark_right.move_to(robot_right.get_center())
        
        # Issue 39: Place inequality text in area F2-F5
        inequality_text = Text("BA \u2260 AB", font_size=40, color=WHITE)
        self.place_in_area(inequality_text, 'F2', 'F5', scale_factor=1.0)
        
        self.play(
            Create(x_mark_left),
            Create(x_mark_right),
            FadeIn(inequality_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.play(Indicate(inequality_text, color=YELLOW))
        self.wait(2)
