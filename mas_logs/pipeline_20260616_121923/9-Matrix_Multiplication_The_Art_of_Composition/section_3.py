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
        # 1. Setup layout and Title
        self.setup_layout(
            "Defining the 'Shortcut' (The Product)", 
            [
                'We want to skip the intermediate movement step.', 
                'Matrix C is the product of applying B after A.', 
                'This Master Matrix teleports the Robot in one leap.'
            ]
        )
        
        # Helper for creating matrices with Text
        def create_text_matrix(values, color=WHITE, bracket_color=WHITE):
            grid_m = VGroup(*[
                VGroup(*[Text(str(item), font_size=24, color=color) for item in row]).arrange(RIGHT, buff=0.4)
                for row in values
            ]).arrange(DOWN, buff=0.3)
            
            l_bracket = Text("[", font_size=40, color=bracket_color)
            r_bracket = Text("]", font_size=40, color=bracket_color)
            
            l_bracket.stretch_to_fit_height(grid_m.height + 0.1)
            r_bracket.stretch_to_fit_height(grid_m.height + 0.1)
            l_bracket.next_to(grid_m, LEFT, buff=0.1)
            r_bracket.next_to(grid_m, RIGHT, buff=0.1)
            
            return VGroup(l_bracket, grid_m, r_bracket)

        # Colors used
        COLOR_ROBOT = YELLOW
        COLOR_MATRIX_C = "#FF00FF" # Magenta
        COLOR_MATRIX_A = BLUE
        COLOR_MATRIX_B = GREEN

        # === Animation for Lecture Line 1 ===
        # Show the Robot at start (1, 0) and final end (1, 1) positions simultaneously on the grid.
        plane = NumberPlane(
            x_range=[-1, 2, 1], 
            y_range=[-1, 2, 1], 
            background_line_style={"stroke_opacity": 0.3}
        )
        
        robot_start_dot = Dot(plane.c2p(1, 0), color=COLOR_ROBOT, radius=0.1)
        robot_end_dot = Dot(plane.c2p(1, 1), color=COLOR_ROBOT, radius=0.1)
        start_label = Text("(1,0)", font_size=16, color=COLOR_ROBOT).next_to(robot_start_dot, DOWN, buff=0.1)
        end_label = Text("(1,1)", font_size=16, color=COLOR_ROBOT).next_to(robot_end_dot, UP, buff=0.1)
        
        robot_leap_visual = VGroup(plane, robot_start_dot, robot_end_dot, start_label, end_label)
        
        # Resolve Issue 34: Positioning the Robot leap visual in E2-F5
        self.place_in_area(robot_leap_visual, 'E2', 'F5', scale_factor=0.9)
        
        self.play(self.lecture[0].animate.set_color(COLOR_ROBOT))
        self.play(FadeIn(robot_leap_visual))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the multiplication BA = C, where C is [[0, -1], [1, 1]] in magenta (#FF00FF).
        mat_b = create_text_matrix([["b11", "b12"], ["b21", "b22"]], color=COLOR_MATRIX_B)
        mat_a = create_text_matrix([["a11", "a12"], ["a21", "a22"]], color=COLOR_MATRIX_A)
        equals = Text("=", font_size=28)
        mat_c = create_text_matrix([[0, -1], [1, 1]], color=COLOR_MATRIX_C)
        
        matrix_composition = VGroup(mat_b, mat_a, equals, mat_c).arrange(RIGHT, buff=0.3)
        
        # Resolve Issue 32: Positioning the full matrix equation B x A = C in B1-C6
        self.place_in_area(matrix_composition, 'B1', 'C6', scale_factor=0.6)
        
        master_label = Text("Master Matrix C", font_size=22, color=COLOR_MATRIX_C)
        # Resolve Issue 33: Positioning the 'Master Matrix' label at D5
        self.place_at_grid(master_label, 'D5', scale_factor=0.8)
        
        self.play(self.lecture[1].animate.set_color(COLOR_MATRIX_C))
        self.play(Write(matrix_composition))
        self.play(FadeIn(master_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the Robot teleporting directly from (1, 0) to (1, 1) using the transformation Matrix C.
        teleporting_robot = Dot(plane.c2p(1, 0), color=COLOR_ROBOT, radius=0.12)
        
        self.play(self.lecture[2].animate.set_color(COLOR_MATRIX_C))
        self.add(teleporting_robot)
        # Robot "teleports" (quick animation) using theconsolidated instruction manual C
        self.play(
            teleporting_robot.animate.move_to(plane.c2p(1, 1)), 
            rate_func=slow_into, 
            run_time=1.5
        )
        self.play(Indicate(mat_c), Flash(robot_end_dot, color=COLOR_ROBOT))
        self.wait(2)
