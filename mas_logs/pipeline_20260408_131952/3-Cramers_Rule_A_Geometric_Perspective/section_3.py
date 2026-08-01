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
        # Setup layout with prompt-specified lecture lines
        title = "The Vector Combination Problem"
        lines = [
            "Think of V1 and V2 as a new grid.",
            "Target B sits at a specific spot here.",
            "We need B's coordinates in this custom basis."
        ]
        self.setup_layout(title, lines)

        # Colors
        V1_COLOR = "#0000FF"  # Blue
        V2_COLOR = "#FF0000"  # Red
        B_COLOR = "#00FF00"   # Green
        COORD_COLOR = "#FFFF00" # Yellow

        # 1. Create the basis grid (NumberPlane)
        # We use a restricted range so it doesn't spill over too much, 
        # and we'll apply the matrix transformation.
        basis_grid = NumberPlane(
            x_range=[-2, 10, 1],
            y_range=[-2, 14, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        
        # Issue 34: Place grid correctly to avoid obstructing lecture notes
        self.place_in_area(basis_grid, 'A1', 'F6', scale_factor=0.9)
        
        # Helper to create vectors aligned with the transformed plane
        def get_v_in_plane(plane, coords, color):
            start = plane.get_origin()
            end = plane.c2p(coords[0], coords[1])
            return Vector(end - start, color=color).shift(start)

        # Initial basis vectors (i, j)
        i_vec = get_v_in_plane(basis_grid, [1, 0], V1_COLOR)
        j_vec = get_v_in_plane(basis_grid, [0, 1], V2_COLOR)
        
        # Transformed vectors (V1, V2)
        v1_vec_final = get_v_in_plane(basis_grid, [2, 1], V1_COLOR)
        v2_vec_final = get_v_in_plane(basis_grid, [1, 3], V2_COLOR)

        # Labels
        v1_label = Text("V1", color=V1_COLOR, font_size=24, slant=ITALIC)
        v2_label = Text("V2", color=V2_COLOR, font_size=24, slant=ITALIC)
        
        # Issue 35: Place V2 label to avoid overlap
        self.place_at_grid(v2_label, 'D1', scale_factor=0.6)
        self.place_at_grid(v1_label, 'F6', scale_factor=0.6) # Placing V1 label at bottom right

        # Target B and Assets
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/target.svg
        # Issue 25: Asset integration
        target_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/target.svg")
        target_icon.set_color(B_COLOR)
        
        b_vec = get_v_in_plane(basis_grid, [7, 11], B_COLOR)
        # Place target icon near the B vector's endpoint coordinates
        self.place_at_grid(target_icon, 'B3', scale_factor=0.4)

        # Issue 36: Target coordinates label at B4
        b_coords_label = Text("(2, 3)", color=COORD_COLOR, font_size=32)
        self.place_at_grid(b_coords_label, 'B4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.add(basis_grid, i_vec, j_vec)
        self.wait(1)

        # Transform standard basis to V1, V2 grid
        matrix = [[2, 1], [1, 3]]
        self.play(
            basis_grid.animate.apply_matrix(matrix),
            Transform(i_vec, v1_vec_final),
            Transform(j_vec, v2_vec_final),
            FadeIn(v1_label),
            FadeIn(v2_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Show target vector B and the target asset
        self.play(Create(b_vec))
        self.play(FadeIn(target_icon))
        self.play(Indicate(target_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Show coordinates (2, 3) in custom basis
        self.play(Write(b_coords_label))
        self.wait(2)

        # Reset lecture color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
