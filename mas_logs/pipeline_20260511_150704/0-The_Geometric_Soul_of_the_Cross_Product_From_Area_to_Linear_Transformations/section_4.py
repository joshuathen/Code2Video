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
        lines = [
            "Compute the cross product using a symbolic 3x3 determinant.",
            "The top row holds the unit vectors i, j, k.",
            "Use the cofactor method to find each component.",
            "This result relates to the volume of a parallelepiped.",
            "Specifically, the triple product dotting with a third vector."
        ]
        self.setup_layout("The Matrix Connection: Determinants in 3D", lines)

        # Colors for consistency
        L1_COLOR = YELLOW_A
        L2_COLOR = BLUE_A
        L3_COLOR = GREEN_A
        L4_COLOR = ORANGE
        L5_COLOR = RED_A

        # === Animation for Lecture Line 1 ===
        # Create a symbolic 3x3 matrix using Text (avoiding LaTeX)
        # vals: row1, row2, row3
        vals = [
            ["i", "j", "k"],
            ["ux", "uy", "uz"],
            ["vx", "vy", "vz"]
        ]
        
        # Build matrix manually
        rows_vg = VGroup()
        for row in vals:
            row_items = VGroup(*[Text(char, font_size=24) for char in row]).arrange(RIGHT, buff=0.7)
            rows_vg.add(row_items)
        rows_vg.arrange(DOWN, buff=0.5)

        # Draw vertical lines for determinant
        line_left = Line(rows_vg.get_corner(UL), rows_vg.get_corner(DL)).shift(LEFT*0.2)
        line_right = Line(rows_vg.get_corner(UR), rows_vg.get_corner(DR)).shift(RIGHT*0.2)
        m_matrix = VGroup(rows_vg, line_left, line_right)
        
        # Fixed position: Issue 40
        self.place_at_grid(m_matrix, 'B2', scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(L1_COLOR),
            FadeIn(m_matrix)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight top row
        top_row_rect = SurroundingRectangle(rows_vg[0], color=L2_COLOR, buff=0.1)
        
        self.play(
            self.lecture[1].animate.set_color(L2_COLOR),
            Create(top_row_rect)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show cofactor formula components
        expansion = Text("i(uy vz - uz vy) - j(ux vz - uz vx) + k(ux vy - uy vx)", font_size=18, color=L3_COLOR)
        # Fixed position: Issue 41
        self.place_in_area(expansion, 'B4', 'B6', scale_factor=0.8)
        
        # Visual "cover-up" for first component (i)
        # Vertical highlight for col 1
        col1_rect = SurroundingRectangle(VGroup(rows_vg[0][0], rows_vg[1][0], rows_vg[2][0]), color=L3_COLOR, buff=0.1)
        
        self.play(
            self.lecture[2].animate.set_color(L3_COLOR),
            Write(expansion),
            Create(col1_rect)
        )
        self.play(FadeOut(col1_rect), FadeOut(top_row_rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Volume connection label
        vol_text = Text("Cross Product Magnitude = Area (3D Volume context)", font_size=18, color=L4_COLOR)
        self.place_at_grid(vol_text, 'C5', scale_factor=1.0)
        
        self.play(
            self.lecture[3].animate.set_color(L4_COLOR),
            Write(vol_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Vector visualization: Issue 42
        # Using Axes to create a 3D-like perspective coordinate system in 2D
        axes = Axes(
            x_range=[-2, 2], y_range=[-2, 2],
            x_length=3, y_length=3,
            axis_config={"color": GREY, "stroke_width": 2}
        )
        
        # Pseudo-3D vectors
        u_vec = Arrow(axes.c2p(0,0), axes.c2p(1.2, 0.4), color=BLUE, buff=0, stroke_width=4)
        v_vec = Arrow(axes.c2p(0,0), axes.c2p(0.8, -0.6), color=GREEN, buff=0, stroke_width=4)
        # Cross product (roughly perpendicular to u, v in this projection)
        cross_vec = Arrow(axes.c2p(0,0), axes.c2p(-0.4, 1.5), color=YELLOW, buff=0, stroke_width=4)
        
        u_label = Text("u", font_size=16, color=BLUE).next_to(u_vec.get_end(), UR, buff=0.1)
        v_label = Text("v", font_size=16, color=GREEN).next_to(v_vec.get_end(), DR, buff=0.1)
        cp_label = Text("u x v", font_size=16, color=YELLOW).next_to(cross_vec.get_end(), UL, buff=0.1)
        
        viz_group = VGroup(axes, u_vec, v_vec, cross_vec, u_label, v_label, cp_label)
        # Fixed position: Issue 42
        self.place_in_area(viz_group, 'D2', 'F5', scale_factor=0.8)
        
        self.play(
            self.lecture[4].animate.set_color(L5_COLOR),
            FadeIn(viz_group)
        )
        self.wait(2)
