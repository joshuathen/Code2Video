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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Matrices as Morphing Grids (0:45 - 1:45)"
        lecture_lines = [
            "A matrix isn't just numbers; it warps space.",
            "Imagine a grid sheet being stretched and rotated.",
            "The origin must always stay in the same place.",
            "Grid lines must remain parallel and evenly spaced.",
            "This transformation moves every vector on the grid."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_GRID = "#444444"
        COLOR_ORIGIN = "#FFFFFF"
        COLOR_V_LINES = "#ADD8E6"
        COLOR_VECTOR = "#FFA500"

        # === Animation for Lecture Line 1 ===
        # A matrix isn't just numbers; it warps space.
        self.lecture[0].set_color(COLOR_GRID)
        
        # Create NumberPlane
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": COLOR_GRID,
                "stroke_width": 2,
                "stroke_opacity": 0.8
            },
            axis_config={"include_tip": False, "stroke_color": COLOR_GRID}
        )
        # Resolved Issue 30 & 31: Adjust position and scale to avoid obstruction and clipping
        self.place_in_area(plane, 'A2', 'F6', scale_factor=0.65)
        
        origin_dot = Dot(plane.c2p(0, 0), color=COLOR_ORIGIN, radius=0.1)
        
        self.play(Create(plane), FadeIn(origin_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine a grid sheet being stretched and rotated.
        self.lecture[1].set_color(WHITE)
        
        # Shear matrix
        shear_matrix = [[1, 1], [0, 1]]
        
        self.play(
            plane.animate.apply_matrix(shear_matrix),
            # Origin dot stays at (0,0) in logical coordinates. 
            # Since the matrix is linear, plane.c2p(0,0) remains at the center.
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The origin must always stay in the same place.
        self.lecture[2].set_color(COLOR_ORIGIN)
        self.play(Indicate(origin_dot, color=COLOR_ORIGIN, scale_factor=2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Grid lines must remain parallel and evenly spaced.
        self.lecture[3].set_color(COLOR_V_LINES)
        
        # Vertical lines are at index 1 of background_lines
        v_lines = plane.background_lines[1]
        self.play(v_lines.animate.set_color(COLOR_V_LINES))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This transformation moves every vector on the grid.
        self.lecture[4].set_color(COLOR_VECTOR)
        
        # Setup vector [1, 1] on the currently sheared grid
        vector = Arrow(plane.c2p(0, 0), plane.c2p(1, 1), buff=0, color=COLOR_VECTOR, stroke_width=6)
        
        # Resolved Issue 32: Use grid system for the label and avoid screen boundary
        vector_label = MathTex(r"\begin{bmatrix} 1 \\ 1 \end{bmatrix}", color=COLOR_VECTOR, font_size=24)
        self.place_at_grid(vector_label, 'B5', scale_factor=0.6)

        self.play(GrowArrow(vector), FadeIn(vector_label))
        self.wait(1)
        
        # Further transformation to show vector moves with grid
        # Rotation by 30 degrees
        rotate_matrix = [[0.866, -0.5], [0.5, 0.866]]
        
        # To make the vector move with the grid, we can either add it to the plane 
        # or animate its start/end points using plane.c2p
        # We also need to move the label to follow the vector tip.
        
        def update_label(m):
            # Update label position to be near the vector tip in the transformed space
            m.move_to(vector.get_end() + np.array([0.4, 0.4, 0]))

        self.play(
            plane.animate.apply_matrix(rotate_matrix),
            vector.animate.put_start_and_end_on(plane.c2p(0, 0), plane.c2p(1, 1)),
            UpdateFromFunc(vector_label, update_label),
            run_time=2
        )
        self.wait(2)
