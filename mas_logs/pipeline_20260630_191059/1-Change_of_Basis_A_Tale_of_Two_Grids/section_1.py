from manim import *
import numpy as np

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
        # Data from storyboard
        title = "The Hook: Same Point, Different Perspectives"
        lecture_lines = [
            "Alice uses a standard square coordinate grid.",
            "She sees a point at coordinates three-two.",
            "Bob uses a tilted and stretched coordinate grid.",
            "He sees the same point at coordinate one-one.",
            "Coordinates depend on the perspective you choose."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Basis Colors
        ALICE_GRID_COLOR = WHITE
        POINT_COLOR = RED
        BOB_GRID_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # Alice uses a standard square coordinate grid.
        self.play(self.lecture[0].animate.set_color(ALICE_GRID_COLOR))
        
        # Alice's Square Grid
        # We use a NumberPlane with symmetric ranges so its center is the origin.
        # This allows us to place the origin precisely at 'E2' using place_at_grid.
        # x_range covers roughly the right side area A1-F6 when centered at E2.
        alice_grid = NumberPlane(
            x_range=[-2, 5, 1], 
            y_range=[-2, 5, 1],
            x_length=7, y_length=7,
            background_line_style={"stroke_color": ALICE_GRID_COLOR, "stroke_width": 1, "stroke_opacity": 0.3},
            axis_config={"stroke_color": ALICE_GRID_COLOR, "stroke_width": 2, "stroke_opacity": 0.5},
            tips=False
        )
        # Note: Symmetric ranges [-3.5, 3.5] would make center=0.
        # Here x_range=[-2, 5] has center 1.5. To make origin at E2, we'll use symmetric ranges.
        alice_grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=8, y_length=8,
            background_line_style={"stroke_color": ALICE_GRID_COLOR, "stroke_width": 1, "stroke_opacity": 0.3},
            axis_config={"stroke_color": ALICE_GRID_COLOR, "stroke_width": 2, "stroke_opacity": 0.5},
            tips=False
        )
        self.place_at_grid(alice_grid, 'E2')
        
        self.play(Create(alice_grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # She sees a point at coordinates three-two.
        self.play(self.lecture[1].animate.set_color(POINT_COLOR))
        
        # Point at (3,2) relative to Alice's origin at E2 is C5.
        dot = Dot(color=POINT_COLOR, radius=0.1)
        self.place_at_grid(dot, "C5")
        
        # Label for Alice's coordinate view
        # Fix Issue 31: Use specific grid anchor for the label
        coord_label = Text("(3, 2)", font_size=20, color=WHITE)
        self.place_at_grid(coord_label, "C6", scale_factor=0.8)
        
        self.play(FadeIn(dot), Write(coord_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Bob uses a tilted and stretched coordinate grid.
        self.play(self.lecture[2].animate.set_color(BOB_GRID_COLOR))
        
        # Bob's Basis: b1 = (2, 1), b2 = (1, 1) in Alice's units.
        # This means Bob's (1,1) is Alice's (3,2).
        matrix = [[2, 1], [1, 1]]
        
        # Create Bob's grid by transforming a NumberPlane.
        # Fix Issue 29: Limit ranges to avoid excessive length and clutter.
        bob_grid = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=4, y_length=4,
            background_line_style={"stroke_color": BOB_GRID_COLOR, "stroke_width": 2, "stroke_opacity": 0.6},
            axis_config={"stroke_color": BOB_GRID_COLOR, "stroke_width": 3, "stroke_opacity": 0.8},
            tips=False
        )
        bob_grid.apply_matrix(matrix)
        
        # Origin remains at center of the transformed parallelogram.
        self.place_at_grid(bob_grid, 'E2')
        
        self.play(FadeOut(alice_grid), FadeIn(bob_grid))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # He sees the same point at coordinate one-one.
        self.play(self.lecture[3].animate.set_color(BOB_GRID_COLOR))
        
        # Transform the label to Bob's perspective (1,1)
        bob_label_text = Text("(1, 1)", font_size=20, color=BOB_GRID_COLOR)
        self.place_at_grid(bob_label_text, "C6", scale_factor=0.8)
        
        self.play(Transform(coord_label, bob_label_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Coordinates depend on the perspective you choose.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Final visual emphasis
        self.play(Flash(dot, color=WHITE, line_length=0.2, flash_radius=0.3))
        self.wait(2)
