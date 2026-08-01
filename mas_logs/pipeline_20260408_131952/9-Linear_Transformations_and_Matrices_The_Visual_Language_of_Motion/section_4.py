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
        # Setup layout
        title = "Encoding Motion into a Matrix"
        lines = [
            'We start with the standard basis vectors on the grid.',
            'Watch as i-hat and j-hat move to new locations.',
            'A 2x2 matrix records these final basis positions.',
            'The first column captures the coordinates of transformed i-hat.',
            'The second column captures the coordinates of transformed j-hat.'
        ]
        self.setup_layout(title, lines)

        # Colors
        I_HAT_COLOR = "#FF0000"
        J_HAT_COLOR = "#00FF00"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Grid area for the plane: B1 to E4 (Issue 40)
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, 'B1', 'E4', scale_factor=0.8)
        
        # Basis vectors
        i_hat = Vector([1, 0], color=I_HAT_COLOR)
        j_hat = Vector([0, 1], color=J_HAT_COLOR)
        
        # Shift vectors to plane position
        plane_center = plane.get_center()
        i_hat.shift(plane_center)
        j_hat.shift(plane_center)
        
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Capture target world positions before plane transforms
        target_i_world = plane.coords_to_point(2, 1)
        target_j_world = plane.coords_to_point(-1, 1)
        
        # Transformation matrix
        matrix_vals = [[2, -1], [1, 1]]
        
        self.play(
            plane.animate.apply_matrix(matrix_vals),
            i_hat.animate.put_start_and_end_on(plane_center, target_i_world),
            j_hat.animate.put_start_and_end_on(plane_center, target_j_world),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Empty brackets appearing at grid positions (Issue 41, 42)
        bracket_l = Text("[", font_size=80)
        bracket_r = Text("]", font_size=80)
        self.place_at_grid(bracket_l, 'C5', scale_factor=1.5)
        self.place_at_grid(bracket_r, 'C6', scale_factor=1.5)
        
        self.play(Write(bracket_l), Write(bracket_r))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # First column coordinates for i-hat
        coord_i = Text("(2, 1)", color=I_HAT_COLOR, font_size=20).next_to(i_hat.get_end(), UR, buff=0.1)
        self.play(Write(coord_i))
        
        val_i1 = Text("2", color=I_HAT_COLOR, font_size=36)
        val_i2 = Text("1", color=I_HAT_COLOR, font_size=36)
        
        # Determine target positions in matrix column 1
        target_pos_i1 = self.grid["B5"].copy()
        target_pos_i2 = self.grid["D5"].copy()
        
        val_i1.move_to(coord_i.get_center())
        val_i2.move_to(coord_i.get_center())
        
        self.play(
            val_i1.animate.move_to(target_pos_i1),
            val_i2.animate.move_to(target_pos_i2),
            coord_i.animate.set_opacity(0),
            run_time=1.5
        )
        self.add(val_i1, val_i2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Second column coordinates for j-hat
        coord_j = Text("(-1, 1)", color=J_HAT_COLOR, font_size=20).next_to(j_hat.get_end(), UL, buff=0.1)
        self.play(Write(coord_j))
        
        val_j1 = Text("-1", color=J_HAT_COLOR, font_size=36)
        val_j2 = Text("1", color=J_HAT_COLOR, font_size=36)
        
        # Determine target positions in matrix column 2
        target_pos_j1 = self.grid["B6"].copy()
        target_pos_j2 = self.grid["D6"].copy()
        
        val_j1.move_to(coord_j.get_center())
        val_j2.move_to(coord_j.get_center())
        
        self.play(
            val_j1.animate.move_to(target_pos_j1),
            val_j2.animate.move_to(target_pos_j2),
            coord_j.animate.set_opacity(0),
            run_time=1.5
        )
        self.add(val_j1, val_j2)
        self.wait(2)
