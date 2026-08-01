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

class Section2Scene(TeachingScene):
    def construct(self):
        # 1. Setup title and lecture lines
        title_text = "The Sequential Challenge"
        lines = [
            "Let's restart our Robot at its original position.",
            "First, Matrix A rotates the Robot ninety degrees.",
            "Now, we introduce Matrix B to add a shear.",
            "Matrix B shifts the Robot to its final destination.",
            "We can track the full path of this journey."
        ]
        self.setup_layout(title_text, lines)

        # 2. Define Colors and Assets
        ROBOT_COLOR = "#00FF00"
        GRID_COLOR = "#555555"
        HIGHLIGHT_COLOR = "#FFFF00"
        MATRIX_B_COLOR = "#00FFFF"
        PATH_COLOR = "#FFFFFF"

        # 3. Create Scene Elements
        # Background Plane
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": GRID_COLOR, "stroke_width": 2, "stroke_opacity": 0.6},
            axis_config={"stroke_color": GRID_COLOR, "include_tip": True}
        )
        self.place_in_area(plane, 'B3', 'F6', scale_factor=0.8)

        # Robot Mobject
        robot = VGroup(
            Square(side_length=0.4, color=ROBOT_COLOR, fill_opacity=0.5),
            Dot(radius=0.03, color=WHITE).shift(UP*0.08 + LEFT*0.08),
            Dot(radius=0.03, color=WHITE).shift(UP*0.08 + RIGHT*0.08),
            Line(LEFT*0.1, RIGHT*0.1, color=WHITE).shift(DOWN*0.1)
        )
        # Position Robot at (1,0) relative to plane
        robot.move_to(plane.c2p(1, 0))

        # Matrices
        matrix_a = Text("A = [[0, -1], [1, 0]]", font_size=20)
        matrix_b = Text("B = [[1, 1], [0, 1]]", font_size=20, color=MATRIX_B_COLOR)
        # Resolved Issue 29: Position and scale matrices
        self.place_at_grid(matrix_a, 'A1', scale_factor=0.8)
        self.place_at_grid(matrix_b, 'A4', scale_factor=0.8)

        # Labels (positioned via grid to meet constraints)
        start_label = Text("(1,0)", font_size=16, color=ROBOT_COLOR)
        mid_label = Text("(0,1)", font_size=16, color=ROBOT_COLOR)
        final_label = Text("(1,1)", font_size=16, color=ROBOT_COLOR)
        
        # Resolved Issue 31: Move start_label to E6
        self.place_at_grid(start_label, 'E6', scale_factor=0.8)
        self.place_at_grid(mid_label, 'C4', scale_factor=1.0)
        # Resolved Issue 30: Move final_label to B6
        self.place_at_grid(final_label, 'B6', scale_factor=0.8)

        # Scene coordinates for path tracking
        origin_abs = plane.c2p(0, 0)
        p_start_abs = plane.c2p(1, 0).copy()

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(FadeIn(plane), FadeIn(robot), FadeIn(start_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.play(FadeIn(matrix_a))
        self.wait(0.5)
        
        # Apply Rotation
        self.play(
            Rotate(plane, PI/2, about_point=origin_abs),
            Rotate(robot, PI/2, about_point=origin_abs),
            FadeOut(start_label),
            run_time=2
        )
        self.play(FadeIn(mid_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.play(FadeIn(matrix_b))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Apply Shear Matrix B
        shear_matrix = [[1, 1], [0, 1]]
        self.play(
            plane.animate.apply_matrix(shear_matrix, about_point=origin_abs),
            robot.animate.apply_matrix(shear_matrix, about_point=origin_abs),
            FadeOut(mid_label),
            run_time=2
        )
        self.play(FadeIn(final_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Calculate path points in scene space
        # Rotation 90 deg around origin_abs
        rel_start = p_start_abs - origin_abs
        rel_mid = np.array([-rel_start[1], rel_start[0], 0])
        p_mid_abs = origin_abs + rel_mid
        
        # Shear rel_mid by Matrix B
        rel_final = np.array([rel_mid[0] + rel_mid[1], rel_mid[1], 0])
        p_final_abs = origin_abs + rel_final
        
        # Visualize Path
        radius_val = np.linalg.norm(rel_start)
        arc_path = ArcBetweenPoints(p_start_abs, p_mid_abs, radius=radius_val, stroke_color=PATH_COLOR)
        dashed_arc = DashedVMobject(arc_path)
        shear_path = DashedLine(p_mid_abs, p_final_abs, color=PATH_COLOR)
        
        self.play(Create(dashed_arc))
        self.play(Create(shear_path))
        self.wait(2)
        
        # Cleanup highlighting
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
