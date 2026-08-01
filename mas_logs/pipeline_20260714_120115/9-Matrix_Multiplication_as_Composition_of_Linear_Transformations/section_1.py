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

class Section1Scene(TeachingScene):
    def construct(self):
        # --- Metadata ---
        # Title: Prerequisite: The Matrix as an Action
        # Line 1: A matrix is more than a grid of numbers.
        # Line 2: It is a function that transforms 2D space.
        # Line 3: Matrix A moves basis vectors to new locations.

        lecture_lines = [
            "A matrix is more than a grid of numbers.",
            "It is a function that transforms 2D space.",
            "Matrix A moves basis vectors to new locations."
        ]
        self.setup_layout("Prerequisite: The Matrix as an Action", lecture_lines)

        # Colors
        color_i = "#FF0000"
        color_j = "#00FF00"
        color_v = "#FFFF00"
        highlight_color = YELLOW

        # Define Transformation Matrix
        # Matrix A = [[1, 1], [0, 1]] (Horizontal Shear)
        # This moves (1,0) -> (1,0) and (0,1) -> (1,1)
        shear_matrix = [[1, 1], [0, 1]]

        # Asset path
        grid_asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/grid.svg"

        # === Animation for Lecture Line 1 ===
        # Display 2D grid [Asset: grid.svg] with basis vectors i-hat (#FF0000) and j-hat (#00FF00) labeled.
        self.lecture[0].set_color(highlight_color)
        
        # Load grid asset - Addressing Issue 25
        grid_asset = SVGMobject(grid_asset_path)
        # Apply Issue 27 fix: use area B1 to F5
        self.place_in_area(grid_asset, 'B1', 'F5', scale_factor=0.7)
        
        # Hidden NumberPlane to manage vector positioning, aligned with grid_asset
        # Grid range adjusted to fit typical icon density
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0},
            axis_config={"stroke_opacity": 0}
        )
        plane.move_to(grid_asset.get_center())
        # Scale plane to match the grid_asset scale from place_in_area
        # NumberPlane defaults to 1 grid unit = 1 manim unit.
        plane.scale(0.7) 

        # Basis vectors
        i_vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=color_i, stroke_width=4)
        j_vec = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=color_j, stroke_width=4)
        
        # Initial labels
        # Applying Issue 29 fix: label_i at E4
        label_i = Text("i(1,0)", color=color_i, font_size=24)
        self.place_at_grid(label_i, 'E4', scale_factor=0.6)
        
        label_j = Text("j(0,1)", color=color_j, font_size=24)
        # Initial j label position (above the tip at C3)
        self.place_at_grid(label_j, 'B3', scale_factor=0.6)

        self.add(grid_asset)
        self.play(Create(i_vec), Create(j_vec), Write(label_i), Write(label_j))
        self.next_section("Grid Basics")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the grid [Asset: grid.svg] shearing to the right, moving basis vectors to (1,0) and (1,1).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)

        # Perform the shear transformation
        self.play(
            grid_asset.animate.apply_matrix(shear_matrix),
            i_vec.animate.apply_matrix(shear_matrix),
            j_vec.animate.apply_matrix(shear_matrix),
            run_time=2
        )
        
        # Update j label to show new coordinates (1,1)
        label_j_new = Text("j(1,1)", color=color_j, font_size=24)
        # Position updated j label at B4 (above the new tip at C4)
        self.place_at_grid(label_j_new, 'B4', scale_factor=0.6)
        
        self.play(Transform(label_j, label_j_new))
        self.next_section("Linear Transformation")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a vector v and label its transformed position as A(v).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)

        # Vector v = [1, 1] transformed by shear is [2, 1]
        # (calculated as A * [1, 1]^T = [[1, 1], [0, 1]] * [1, 1] = [2, 1])
        v_vec = Arrow(plane.c2p(0, 0), plane.c2p(2, 1), buff=0, color=color_v, stroke_width=4)
        
        # Applying Issue 28 fix: label_Av at D5
        label_Av = Text("A(v)", slant=ITALIC, color=color_v, font_size=28)
        self.place_at_grid(label_Av, 'D5', scale_factor=0.7)

        self.play(
            Create(v_vec), 
            Write(label_Av)
        )
        self.next_section("Matrix Mapping")
        self.wait(3)
