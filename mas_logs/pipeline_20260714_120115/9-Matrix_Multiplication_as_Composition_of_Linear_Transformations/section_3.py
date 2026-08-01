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
        # Data
        title_text = "Defining the 'Shortcut' Matrix"
        lecture_lines = [
            "We can find one matrix to do both steps.",
            "This combined matrix C represents the total effect.",
            "Mathematically, we write this as the product BA.",
            "Matrix multiplication represents this shortcut through space.",
            "The product matrix maps original positions to final spots."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_A = "#FF69B4"  # Pink for Matrix A
        COLOR_B = "#87CEFA"  # Light Blue for Matrix B
        COLOR_V = "#FFFFFF"  # White for Vector
        COLOR_C = "#FFD700"  # Gold for Shortcut Matrix C
        
        # Helper: Create a mini grid
        def create_mini_grid(label):
            axes = Axes(
                x_range=[-2, 2, 1],
                y_range=[-2, 2, 1],
                x_length=2,
                y_length=2,
                axis_config={"include_tip": False, "color": GREY_C}
            )
            grid_title = Text(label, font_size=16, color=WHITE).next_to(axes, UP, buff=0.1)
            return VGroup(axes, grid_title)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        left_grid_vgroup = create_mini_grid("Two-Step Process")
        right_grid_vgroup = create_mini_grid("Shortcut Matrix")
        
        self.place_in_area(left_grid_vgroup, "C2", "D3", scale_factor=0.8)
        self.place_in_area(right_grid_vgroup, "C5", "D6", scale_factor=0.8)
        
        axes_l = left_grid_vgroup[0]
        # Initial vector
        v_l = Vector(axes_l.c2p(1, 0) - axes_l.c2p(0, 0), color=COLOR_V).move_to(axes_l.c2p(0,0), aligned_edge=DL)
        
        self.play(
            FadeIn(left_grid_vgroup),
            FadeIn(right_grid_vgroup),
            GrowArrow(v_l)
        )
        
        # Two steps on the left
        v_l_mid = Vector(axes_l.c2p(0, 1) - axes_l.c2p(0, 0), color=COLOR_A).move_to(axes_l.c2p(0,0), aligned_edge=DL)
        v_l_end = Vector(axes_l.c2p(0, 1.5) - axes_l.c2p(0, 0), color=COLOR_B).move_to(axes_l.c2p(0,0), aligned_edge=DL)
        
        self.play(Transform(v_l, v_l_mid))
        self.play(Transform(v_l, v_l_end))
        self.wait(1)
        self.next_section()

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Use Text instead of MathTex to avoid LaTeX requirement
        formula = Text("B(A(v)) = (BA)v", font_size=32, color=WHITE)
        self.place_in_area(formula, "A2", "A5", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)
        self.next_section()

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        matrix_c_def = Text("C = BA", font_size=32, color=WHITE)
        self.place_in_area(matrix_c_def, "E5", "E6", scale_factor=1.0)
        
        formula_with_c = Text("B(A(v)) = Cv", font_size=32, color=WHITE)
        self.place_in_area(formula_with_c, "A2", "A5", scale_factor=0.8)
        
        self.play(
            FadeIn(matrix_c_def),
            Transform(formula, formula_with_c)
        )
        self.wait(1)
        self.next_section()

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        axes_r = right_grid_vgroup[0]
        v_r_start = Vector(axes_r.c2p(1, 0) - axes_r.c2p(0, 0), color=COLOR_V).move_to(axes_r.c2p(0,0), aligned_edge=DL)
        v_r_end = Vector(axes_r.c2p(0, 1.5) - axes_r.c2p(0, 0), color=COLOR_C).move_to(axes_r.c2p(0,0), aligned_edge=DL)
        
        self.play(GrowArrow(v_r_start))
        self.play(Transform(v_r_start, v_r_end))
        self.wait(1)
        self.next_section()
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        highlight_box = SurroundingRectangle(matrix_c_def, color=COLOR_C)
        
        final_note = Text("Shortcut mapping complete", font_size=20, color=COLOR_C)
        self.place_in_area(final_note, "F4", "F6", scale_factor=0.9)
        
        self.play(
            matrix_c_def.animate.set_color(COLOR_C),
            Create(highlight_box),
            Write(final_note)
        )
        self.wait(2)
