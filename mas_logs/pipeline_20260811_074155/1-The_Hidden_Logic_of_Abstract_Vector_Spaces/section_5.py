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
        # Setup title and lecture lines
        title_text = "The Big Picture: Why Abstraction Wins"
        lecture_lines = [
            "Generalizing rules allows us to solve many problems once.",
            "Abstract theorems apply to arrows, functions, and data.",
            "Abstraction is the universal language of the linear universe."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_REMOTE = "#FF8C00"
        COLOR_ARROWS = "#00FF00"
        COLOR_MATRICES = "#FF00FF"
        COLOR_FUNCTIONS = "#00FFFF"
        COLOR_THEOREM = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Show an orange 'Universal Remote' (color #FF8C00) labeled 'Vector Space Axioms'.
        self.lecture[0].set_color(COLOR_REMOTE)
        
        remote_body = RoundedRectangle(corner_radius=0.2, height=1.8, width=1.1, color=COLOR_REMOTE, fill_opacity=0.3)
        remote_label = Text("Vector Space\nAxioms", font_size=16, color=COLOR_REMOTE)
        
        # Add some visual "buttons" to the remote to emphasize the 'Universal Remote' metaphor
        buttons = VGroup(*[Circle(radius=0.08, color=COLOR_REMOTE, fill_opacity=0.8) for _ in range(4)])
        buttons.arrange_in_grid(2, 2, buff=0.1)
        
        remote_content = VGroup(remote_label, buttons).arrange(DOWN, buff=0.2)
        remote = VGroup(remote_body, remote_content)
        
        # Issue 28: Move remote from C3-D4 to C4-D5
        self.place_in_area(remote, "C4", "D5")
        
        self.play(FadeIn(remote))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display colorful icons for Arrows (#00FF00), Matrices (#FF00FF), and Functions (#00FFFF) around the remote.
        self.lecture[1].set_color(COLOR_ARROWS)

        # Arrows Icon
        arrow_icon = Arrow(LEFT, RIGHT, color=COLOR_ARROWS, buff=0).scale(0.8)
        arrow_label = Text("Arrows", font_size=18, color=COLOR_ARROWS)
        arrows_vgroup = VGroup(arrow_icon, arrow_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(arrows_vgroup, "B2")

        # Matrices Icon (Representing 'data' as mentioned in lecture)
        matrix_box = Square(side_length=0.4, color=COLOR_MATRICES, fill_opacity=0.2)
        matrix_cells = VGroup(*[Square(side_length=0.15, color=COLOR_MATRICES) for _ in range(4)])
        matrix_cells.arrange_in_grid(2, 2, buff=0.05)
        matrix_icon = VGroup(matrix_box, matrix_cells)
        matrix_label = Text("Matrices", font_size=18, color=COLOR_MATRICES)
        matrices_vgroup = VGroup(matrix_icon, matrix_label).arrange(DOWN, buff=0.1)
        # Issue 27: Move matrices_vgroup from E2 to D2
        self.place_at_grid(matrices_vgroup, "D2")

        # Functions Icon
        func_icon = Text("f(x)", font_size=20, color=COLOR_FUNCTIONS, slant=ITALIC)
        func_label = Text("Functions", font_size=18, color=COLOR_FUNCTIONS)
        functions_vgroup = VGroup(func_icon, func_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(functions_vgroup, "B5")

        self.play(
            FadeIn(arrows_vgroup),
            FadeIn(matrices_vgroup),
            FadeIn(functions_vgroup)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash a yellow 'Theorem' box (color #FFFF00) that sends pulses to all icons simultaneously.
        self.lecture[2].set_color(COLOR_THEOREM)

        theorem_box = Rectangle(height=0.6, width=1.4, color=COLOR_THEOREM, fill_opacity=0.3)
        theorem_text = Text("Theorem", font_size=20, color=COLOR_THEOREM)
        theorem_group = VGroup(theorem_box, theorem_text)
        # Issue 26: Move theorem_group from A3 to A4
        self.place_at_grid(theorem_group, "A4")

        # Define pulse paths from Theorem to icons
        line_to_arrows = Line(theorem_group.get_center(), arrows_vgroup.get_center(), color=COLOR_THEOREM)
        line_to_matrices = Line(theorem_group.get_center(), matrices_vgroup.get_center(), color=COLOR_THEOREM)
        line_to_functions = Line(theorem_group.get_center(), functions_vgroup.get_center(), color=COLOR_THEOREM)

        self.play(FadeIn(theorem_group))
        self.play(
            ShowPassingFlash(line_to_arrows.copy().set_stroke(width=6), run_time=1.5, time_width=0.5),
            ShowPassingFlash(line_to_matrices.copy().set_stroke(width=6), run_time=1.5, time_width=0.5),
            ShowPassingFlash(line_to_functions.copy().set_stroke(width=6), run_time=1.5, time_width=0.5),
            Flash(theorem_group, color=COLOR_THEOREM, flash_radius=0.7),
            Flash(arrows_vgroup, color=COLOR_THEOREM, flash_radius=0.5),
            Flash(matrices_vgroup, color=COLOR_THEOREM, flash_radius=0.5),
            Flash(functions_vgroup, color=COLOR_THEOREM, flash_radius=0.5)
        )
        self.wait(2)
