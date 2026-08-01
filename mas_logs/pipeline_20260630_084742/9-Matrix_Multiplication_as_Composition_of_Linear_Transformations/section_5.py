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
        # Setup layout
        lecture_lines = [
            'Track i-hat through the first rotation.', 
            'Next, apply the shear to its new position.', 
            'This final point becomes our first product column.', 
            'Follow j-hat to find the second column.', 
            'Now we have our composed transformation matrix.'
        ]
        self.setup_layout("Visual Derivation of the Product", lecture_lines)
        
        # Setup Coordinate System - use Text instead of MathTex for labels to avoid LaTeX dependency
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4.0,
            y_length=4.0,
            background_line_style={"stroke_opacity": 0.3}
        ).add_coordinates(label_constructor=Text)
        # Issue 41: Moved plane to B2-F6 and increased scale
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.85)
        self.add(plane)

        # Colors
        i_color = "#FF0000"  # Red for i-hat
        j_color = "#00FF00"  # Green for j-hat
        
        # Helper function for generating vectors on the plane
        def get_plane_vector(coords, color):
            return Arrow(
                plane.c2p(0, 0), 
                plane.c2p(coords[0], coords[1]), 
                buff=0, 
                color=color, 
                stroke_width=5
            )

        # Initial i-hat at (1,0)
        i_vec = get_plane_vector([1, 0], i_color)
        i_label = Text("i", slant=ITALIC, color=i_color, font_size=24)
        i_label.next_to(i_vec.get_end(), RIGHT, buff=0.1)

        # Construct Matrix manually using Text to avoid LaTeX dependency
        # res_matrix = [[1, -1], [1, 0]]
        col1_mobs = VGroup(Text("1", font_size=24), Text("1", font_size=24)).arrange(DOWN, buff=0.2)
        col2_mobs = VGroup(Text("-1", font_size=24), Text("0", font_size=24)).arrange(DOWN, buff=0.2)
        matrix_content = VGroup(col1_mobs, col2_mobs).arrange(RIGHT, buff=0.3)
        l_bracket = Text("[", font_size=36).next_to(matrix_content, LEFT, buff=0.1)
        r_bracket = Text("]", font_size=36).next_to(matrix_content, RIGHT, buff=0.1)
        res_matrix = VGroup(matrix_content, l_bracket, r_bracket)
        # Add helper to mimic Matrix class behavior for compatibility
        res_matrix.get_columns = lambda: [col1_mobs, col2_mobs]

        # Issue 40: Moved res_matrix to area A5-A6 and increased scale
        self.place_in_area(res_matrix, 'A5', 'A6', scale_factor=0.8)
        res_matrix.set_opacity(0)
        self.add(res_matrix)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(GrowArrow(i_vec), Write(i_label))
        self.wait(1)
        
        # Transformation 1: Rotation (90 deg CCW): (1,0) -> (0,1)
        rotated_i = get_plane_vector([0, 1], i_color)
        self.play(
            ReplacementTransform(i_vec, rotated_i),
            i_label.animate.next_to(plane.c2p(0, 1), UP, buff=0.1)
        )
        i_vec = rotated_i
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Transformation 2: Shear (x' = x+y, y'=y): (0,1) -> (1,1)
        sheared_i = get_plane_vector([1, 1], i_color)
        self.play(
            ReplacementTransform(i_vec, sheared_i),
            i_label.animate.next_to(plane.c2p(1, 1), UR, buff=0.1)
        )
        i_vec = sheared_i
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Show Matrix and highlight first column
        self.play(res_matrix.animate.set_opacity(1))
        col1 = res_matrix.get_columns()[0]
        rect1 = SurroundingRectangle(col1, color=i_color, buff=0.05)
        self.play(Create(rect1))
        self.wait(1)
        self.play(FadeOut(rect1))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Initial j-hat at (0,1)
        j_vec = get_plane_vector([0, 1], j_color)
        j_label = Text("j", slant=ITALIC, color=j_color, font_size=24)
        j_label.next_to(j_vec.get_end(), UP, buff=0.1)
        self.play(GrowArrow(j_vec), Write(j_label))
        
        # Transformation 1: Rotation: (0,1) -> (-1,0)
        rotated_j = get_plane_vector([-1, 0], j_color)
        self.play(
            ReplacementTransform(j_vec, rotated_j),
            j_label.animate.next_to(plane.c2p(-1, 0), LEFT, buff=0.1)
        )
        j_vec = rotated_j
        
        # Transformation 2: Shear: (-1,0) -> (-1+0, 0) = (-1,0)
        self.play(Indicate(j_vec, color=j_color))
        
        # Highlight second column
        col2 = res_matrix.get_columns()[1]
        rect2 = SurroundingRectangle(col2, color=j_color, buff=0.05)
        self.play(Create(rect2))
        self.wait(1)
        self.play(FadeOut(rect2))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Final emphasis on the composed matrix
        self.play(Indicate(res_matrix, scale_factor=1.1))
        self.wait(2)
