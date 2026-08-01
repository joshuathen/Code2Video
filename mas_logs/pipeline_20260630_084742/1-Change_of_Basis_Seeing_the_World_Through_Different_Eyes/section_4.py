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
        # Setup background and initial lecture lines
        lecture_lines = [
            "Bob's coordinates equal P times Z-4's coordinates.",
            "Multiplying by P rotates and stretches the vectors.",
            "If Z-4 says (1, 0), Bob calculates (1, 1).",
            "The math bridges two different perspectives perfectly.",
            "The tilted grid now aligns with Bob's grid."
        ]
        self.setup_layout("The Calculation: Translating Coordinates", lecture_lines)

        # Transformation matrix P = [[1, -1], [1, 1]]
        matrix_p = [[1, -1], [1, 1]]
        
        # Define Coordinate System
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1, "stroke_opacity": 0.5}
        )
        
        # Standard basis vectors for reference (Bob's grid)
        i_hat = Vector([1, 0], color=GREEN)
        j_hat = Vector([0, 1], color=RED)
        i_label = Text("i", color=GREEN)
        j_label = Text("j", color=RED)

        # Z-4's basis vectors as Bob sees them (columns of P)
        b1 = Vector([1, 1], color=YELLOW)
        b2 = Vector([-1, 1], color=ORANGE)
        b1_label = Text("b1", color=YELLOW)
        b2_label = Text("b2", color=ORANGE)

        # Grouping for layout
        grid_group = VGroup(plane, i_hat, j_hat)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        formula = Text("[x]_Bob = P * [x]_Z-4", font_size=32, color=WHITE)
        self.place_at_grid(formula, 'A3', scale_factor=0.7)
        
        # Applying requested fix: self.place_in_area(grid_group, 'A1', 'F6', scale_factor=1.0)
        self.place_in_area(grid_group, 'A1', 'F6', scale_factor=1.0)
        
        self.play(Write(formula), Create(grid_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Applying requested fixes for labels
        self.place_at_grid(i_label, 'D4', scale_factor=0.8)
        self.place_at_grid(j_label, 'B3', scale_factor=0.8)
        
        self.play(Write(i_label), Write(j_label))
        
        # Show transformation of Bob's grid into Z-4's (Bob's view of Z-4)
        # Note: We apply P to the standard plane to show how Bob sees Z-4's world
        self.play(
            plane.animate.apply_matrix(matrix_p),
            i_hat.animate.become(b1),
            j_hat.animate.become(b2),
            i_label.animate.next_to(b1.get_end(), RIGHT, buff=0.1),
            j_label.animate.next_to(b2.get_end(), LEFT, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        calc_text = Text("P * [1, 0]^T = [1, 1]^T", font_size=28, color=YELLOW)
        self.place_at_grid(calc_text, 'B5', scale_factor=0.8)
        
        # The vector [1, 0] in Z-4 system is b1 in Bob's system
        # Highlight b1
        self.play(Write(calc_text), b1.animate.set_stroke(width=8))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Warping from tilted grid back to Bob's square grid (Aligning)
        # This is the inverse of the matrix_p transformation
        inv_p = np.linalg.inv(matrix_p)
        
        self.play(
            plane.animate.apply_matrix(inv_p),
            i_hat.animate.become(Vector([1, 0], color=GREEN)),
            j_hat.animate.become(Vector([0, 1], color=RED)),
            FadeOut(calc_text),
            FadeOut(b1),
            run_time=2
        )
        # Final adjustment of labels to original fixed grid positions as requested
        self.play(
            i_label.animate.move_to(self.grid['D4']).scale(1.0),
            j_label.animate.move_to(self.grid['B3']).scale(1.0)
        )
        self.wait(2)
