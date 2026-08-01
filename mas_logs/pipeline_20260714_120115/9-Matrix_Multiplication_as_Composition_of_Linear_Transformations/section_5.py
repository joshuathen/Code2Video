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
        # Define the title and lecture lines
        title_text = "Why Order Matters (Non-Commutativity)"
        lecture_lines = [
            "Does the order of transformations actually matter?",
            "Rotating then shifting is different from shifting then rotating.",
            "Usually, the matrix product AB is not equal to BA.",
            "We say that matrix multiplication is not commutative.",
            "Changing the order produces a different final result."
        ]
        
        # Set up the layout
        self.setup_layout(title_text, lecture_lines)
        
        # Custom Colors based on storyboard requirements
        ROTATE_COLOR = "#FFFFE0"  # Light Yellow for Rotation
        SHIFT_COLOR = "#ADD8E6"   # Light Blue for Shift
        X_COLOR = "#FF0000"       # Red for the "X"
        ASSET_PATH = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/based.svg"

        # === Animation for Lecture Line 1 ===
        # Step: Create two side-by-side grids: Label the left "BA" and the right "AB".
        # [Constraint L001/L010/Issue 39] Move labels to Row A for better space usage.
        self.lecture[0].set_color(WHITE)
        
        ba_label = Text("BA", color=WHITE, font_size=32)
        ab_label = Text("AB", color=WHITE, font_size=32)
        
        # Use A-row for labels as requested by Issue 39
        self.place_in_area(ba_label, 'A2', 'A3', scale_factor=1.0)
        self.place_in_area(ab_label, 'A5', 'A6', scale_factor=1.0)
        
        # [Issue 40] Move grids up to Rows B-D to fill space
        ba_grid = NumberPlane(
            x_range=[-1, 4, 1], y_range=[-1, 4, 1],
            x_length=3, y_length=3,
            background_line_style={"stroke_opacity": 0.3}
        ).set_color(GRAY)
        
        ab_grid = NumberPlane(
            x_range=[-1, 4, 1], y_range=[-1, 4, 1],
            x_length=3, y_length=3,
            background_line_style={"stroke_opacity": 0.3}
        ).set_color(GRAY)
        
        self.place_in_area(ba_grid, 'B2', 'D3', scale_factor=0.8)
        self.place_in_area(ab_grid, 'B5', 'D6', scale_factor=0.8)
        
        # Helper to get scene coordinates from local grid coords
        def g2s(grid, coords):
            return grid.c2p(*coords)
            
        # Initial vectors starting at the local origin (0,0) of each grid, pointing to (1,0)
        v_ba = Arrow(g2s(ba_grid, [0,0]), g2s(ba_grid, [1,0]), buff=0, color=WHITE)
        v_ab = Arrow(g2s(ab_grid, [0,0]), g2s(ab_grid, [1,0]), buff=0, color=WHITE)
        
        self.play(
            FadeIn(ba_label), FadeIn(ab_label),
            Create(ba_grid), Create(ab_grid),
            GrowArrow(v_ba), GrowArrow(v_ab)
        )
        self.next_section()

        # === Animation for Lecture Line 2 ===
        # Left (BA): Rotate 90 deg (#FFFFE0), then shift 2 units right (#ADD8E6).
        self.lecture[1].set_color(ROTATE_COLOR)
        
        # Part A: Rotate 90 degrees around origin
        self.play(
            Rotate(v_ba, angle=90*DEGREES, about_point=g2s(ba_grid, [0,0])),
            v_ba.animate.set_color(ROTATE_COLOR),
            run_time=1.5
        )
        self.wait(0.2)
        
        # Part B: Shift 2 units right
        shift_vec = g2s(ba_grid, [2,0]) - g2s(ba_grid, [0,0])
        self.play(
            v_ba.animate.shift(shift_vec).set_color(SHIFT_COLOR),
            run_time=1.5
        )
        self.next_section()

        # === Animation for Lecture Line 3 ===
        # Right (AB): Shift 2 units right (#ADD8E6), then rotate 90 deg (#FFFFE0).
        self.lecture[2].set_color(SHIFT_COLOR)
        
        # Part B: Shift 2 units right
        shift_vec_ab = g2s(ab_grid, [2,0]) - g2s(ab_grid, [0,0])
        self.play(
            v_ab.animate.shift(shift_vec_ab).set_color(SHIFT_COLOR),
            run_time=1.5
        )
        self.wait(0.2)
        
        # Part A: Rotate 90 degrees around origin
        self.play(
            Rotate(v_ab, angle=90*DEGREES, about_point=g2s(ab_grid, [0,0])),
            v_ab.animate.set_color(ROTATE_COLOR),
            run_time=1.5
        )
        self.next_section()

        # === Animation for Lecture Line 4 ===
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/based.svg]
        # Point out different final locations using the SVG asset at the tips.
        self.lecture[3].set_color(WHITE)
        
        # Create asset markers at the final vector tips (Issue 26)
        icon_ba = SVGMobject(ASSET_PATH).scale(0.15).move_to(v_ba.get_end())
        icon_ab = SVGMobject(ASSET_PATH).scale(0.15).move_to(v_ab.get_end())
        
        # Labels for the end coordinates (BA: (2,1), AB: (0,2) rotated is (0,3))
        # BA: (1,0) rot 90 -> (0,1), shift 2 right -> (2,1). Tip is at (2,1).
        # AB: (1,0) shift 2 right -> (3,0), rot 90 around origin -> (0,3). Tip is at (0,3).
        coord_ba = Text("(2, 1)", font_size=16).next_to(icon_ba, UR, buff=0.1)
        coord_ab = Text("(0, 3)", font_size=16).next_to(icon_ab, UL, buff=0.1)
        
        self.play(
            FadeIn(icon_ba), FadeIn(icon_ab),
            Write(coord_ba), Write(coord_ab)
        )
        self.wait(1)
        self.next_section()

        # === Animation for Lecture Line 5 ===
        # [Issue 41] Position eq_text at E3-E4 to avoid being too low.
        self.lecture[4].set_color(X_COLOR)
        
        eq_text = Text("BA = AB", font_size=32)
        self.place_in_area(eq_text, 'E3', 'E4', scale_factor=1.0)
        
        # Create the red X mark
        line1 = Line(eq_text.get_corner(UL), eq_text.get_corner(DR), color=X_COLOR, stroke_width=6)
        line2 = Line(eq_text.get_corner(UR), eq_text.get_corner(DL), color=X_COLOR, stroke_width=6)
        x_mark = VGroup(line1, line2)
        
        self.play(Write(eq_text))
        self.play(Create(x_mark))
        self.wait(2)
