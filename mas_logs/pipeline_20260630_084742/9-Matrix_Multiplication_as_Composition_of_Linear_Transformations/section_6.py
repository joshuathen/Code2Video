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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the specific title and lecture lines
        self.setup_layout(
            "Summary and Non-Commutativity",
            [
                "Matrix multiplication is just combining functions.",
                "Does the order of transformations change the result?",
                "Rotating then shearing is not shearing then rotating."
            ]
        )

        # Helper to create a 'Leo' representation using the SVG asset
        def create_leo():
            # Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg
            leo = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg")
            leo.set_height(1.0)
            leo.set_color(BLUE_B)
            return leo

        # Colors for highlights
        COLOR_1 = "#FFFF00" # Yellow
        COLOR_2 = "#00FFFF" # Cyan
        COLOR_3 = "#FF8C00" # Orange
        NOT_EQUAL_COLOR = "#FF0000" # Red

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_1)
        
        # Symbolic representation of functional composition
        comp_symbol = Text("C = B × A ⇔ (f_B ∘ f_A)(v)", color=COLOR_1)
        # Fix for Issue 42: Reposition comp_symbol to row B and adjust scale
        self.place_in_area(comp_symbol, "B1", "B6", scale_factor=0.8)
        self.play(Write(comp_symbol))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_2)
        
        # Comparison question
        order_q = Text("A × B ≟ B × A", color=COLOR_2)
        self.place_in_area(order_q, "C1", "C6", scale_factor=0.9)
        self.play(FadeIn(order_q))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_3)

        # Transformation Matrices
        # Rotate 90 (R) and Shear (S)
        rot_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        shear_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])

        # Create two Leos and their grids
        grid_left = NumberPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1], background_line_style={"stroke_opacity": 0.3})
        leo_left = create_leo()
        scene_left = VGroup(grid_left, leo_left)
        # Fix for Issue 43: Move scene_left to E1-F2 to avoid overlap with label in Row D
        self.place_in_area(scene_left, "E1", "F2", scale_factor=0.4)
        
        label_left = Text("Rotate then Shear", font_size=18, color=WHITE)
        self.place_at_grid(label_left, "D2", scale_factor=1.0)

        grid_right = NumberPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1], background_line_style={"stroke_opacity": 0.3})
        leo_right = create_leo()
        scene_right = VGroup(grid_right, leo_right)
        # Fix for Issue 43: Move scene_right to E5-F6
        self.place_in_area(scene_right, "E5", "F6", scale_factor=0.4)
        
        label_right = Text("Shear then Rotate", font_size=18, color=WHITE)
        self.place_at_grid(label_right, "D5", scale_factor=1.0)

        self.play(
            FadeIn(scene_left), FadeIn(label_left),
            FadeIn(scene_right), FadeIn(label_right)
        )
        self.wait(1)

        # Apply transformations sequentially
        # Animation: Step 1
        self.play(
            leo_left.animate.apply_matrix(rot_matrix[:2, :2]),
            leo_right.animate.apply_matrix(shear_matrix[:2, :2]),
            run_time=1.5
        )
        self.wait(0.5)

        # Animation: Step 2
        self.play(
            leo_left.animate.apply_matrix(shear_matrix[:2, :2]),
            leo_right.animate.apply_matrix(rot_matrix[:2, :2]),
            run_time=1.5
        )
        self.wait(1)

        # Conclusion: Not Equal
        not_equal_sign = Text("≠", color=NOT_EQUAL_COLOR, font_size=72)
        # Fix for Issue 44: Position between final states with reduced scale factor
        self.place_in_area(not_equal_sign, "E3", "E4", scale_factor=0.8)
        
        self.play(FadeIn(not_equal_sign))
        self.play(Indicate(not_equal_sign))
        self.wait(3)
