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
        # Setup title and lecture lines
        lecture_lines_text = [
            "One 'Master Matrix' can perform both steps instantly.",
            "This single jump is the composition of transformations.",
            "We calculate Matrix C by multiplying B and A.",
            "Order is right-to-left because A occurs first.",
            "Composition C equals B times A for direct movement."
        ]
        self.setup_layout("The 'Aha!' Moment: Composition via Multiplication", lecture_lines_text)

        # Colors for highlights
        COLOR_L1 = "#00FFFF" # Cyan
        COLOR_L2 = "#FF00FF" # Magenta
        COLOR_L3 = "#FFFF00" # Yellow
        COLOR_L4 = "#FF8800" # Orange
        COLOR_L5 = "#00FF00" # Green
        
        ASSET_PATH = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png"

        # Matrix Definitions
        # A = Rotation 90, B = Shear
        matrix_a = [[0, -1], [1, 0]]
        matrix_b = [[1, 1], [0, 1]]
        # C = B * A = [[1, 1], [0, 1]] * [[0, -1], [1, 0]] = [[1, -1], [1, 0]]
        matrix_c = [[1, -1], [1, 0]]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_L1)
        
        # Original Grid
        grid_original = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(grid_original, "B2", "D3", scale_factor=0.6)
        
        # Final Grid
        grid_final = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(grid_final, "B5", "D6", scale_factor=0.6)
        grid_final.apply_matrix(matrix_c)

        label_orig = Text("Initial", font_size=18)
        self.place_in_area(label_orig, "A2", "A3")
        
        label_final = Text("Final (B * A)", font_size=18)
        self.place_in_area(label_final, "A5", "A6")

        self.play(
            Create(grid_original),
            Create(grid_final),
            Write(label_orig),
            Write(label_final),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_L2)
        
        # Ghost Robo-Cat [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png]
        try:
            robo_ghost = ImageMobject(ASSET_PATH).scale(0.15)
        except:
            robo_ghost = Circle(radius=0.2, color=BLUE, fill_opacity=0.8) # Fallback
            
        robo_ghost.move_to(grid_original.get_center())
        robo_ghost.set_opacity(0.5)
        
        # Visual indicator for the jump
        path_arc = ArcBetweenPoints(
            grid_original.get_center(), 
            grid_final.get_center(), 
            angle=-TAU/4
        ).set_stroke(COLOR_L2, opacity=0.3)
        
        self.play(Create(path_arc))
        
        # Step sequence: apply A, then apply B
        temp_cat = robo_ghost.copy().set_opacity(0.8)
        self.play(FadeIn(temp_cat))
        
        # Intermediate state A
        self.play(temp_cat.animate.apply_matrix(matrix_a), run_time=1)
        # Move and Final state B
        self.play(
            temp_cat.animate.move_to(grid_final.get_center()).apply_matrix(matrix_b), 
            run_time=1.5
        )
        self.wait(0.5)
        self.play(FadeOut(temp_cat))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_L3)
        
        # Equation C = B * A
        eq_c = VGroup(
            Text("C"), Text("="), Text("B"), Text("⋅"), Text("A")
        ).arrange(RIGHT, buff=0.15).set_color(COLOR_L3)
        self.place_in_area(eq_c, "F2", "F5", scale_factor=0.8)
        
        self.play(Write(eq_c))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_L4)
        
        # Highlighting order: A happens first
        box_a = SurroundingRectangle(eq_c[4], color=WHITE, buff=0.1)
        box_b = SurroundingRectangle(eq_c[2], color=WHITE, buff=0.1)
        
        self.play(Create(box_a))
        self.play(Indicate(eq_c[4]))
        self.wait(0.5)
        self.play(ReplacementTransform(box_a, box_b))
        self.play(Indicate(eq_c[2]))
        self.wait(0.5)
        self.play(FadeOut(box_b))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_L5)
        
        # Solid Robo-Cat [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png]
        try:
            robo_solid = ImageMobject(ASSET_PATH).scale(0.15)
        except:
            robo_solid = Circle(radius=0.2, color=YELLOW, fill_opacity=1.0)
            
        robo_solid.move_to(grid_original.get_center())
        
        self.add(robo_solid)
        # Direct jump animation
        self.play(
            MoveAlongPath(robo_solid, path_arc),
            robo_solid.animate.apply_matrix(matrix_c),
            run_time=2,
            rate_func=slow_into
        )
        
        # Final flash on Matrix C
        self.play(Indicate(eq_c[0], color=COLOR_L5))
        self.wait(2)
