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
        # Initialize layout with title and lecture lines
        self.setup_layout("The Inverse Journey: Going Back", [
            "- Use the inverse matrix to return to the original basis.",
            "- Change of basis is a two-way mathematical street.",
            "- The Human translates his coordinates for the Owl's world."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the inverse formula '[x]_Owl = P^-1 * [x]_Human' in #FFFFFF.
        self.lecture[0].set_color("#FFFF00")
        
        formula = MathTex(
            r"[\mathbf{x}]_{\text{Owl}} = P^{-1} [\mathbf{x}]_{\text{Human}}",
            color="#FFFFFF"
        )
        # Position formula at the top of the workspace
        # Fix Issue 38: scale_factor=0.8
        self.place_in_area(formula, "A2", "A5", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show a static point on the standard grid and highlight its coordinates.
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#FFFF00")
        
        # Grid plane: Human's standard basis (Identity)
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_numbers": False},
            background_line_style={"stroke_color": "#4444FF", "stroke_opacity": 0.5}
        )
        # Fix Issue 39: scale_factor=0.9
        self.place_in_area(plane, "C2", "F5", scale_factor=0.9)
        
        # Absolute point in the standard basis (the snack)
        # Use a dot that will not be transformed by apply_matrix if added separately,
        # but here we want it to stay at the same visual position or follow the grid?
        # Storyboard: "Show a static point... Animate the standard grid warping..."
        # If the grid warps, the point (snack) remains at the same location in the world,
        # but its coordinates relative to the grid change.
        # However, apply_matrix on a plane usually transforms everything on it.
        # Let's use a Dot and keep it stationary relative to the camera to represent the "world point".
        
        world_point = [1, 1, 0]
        dot = Dot(plane.c2p(1, 1), color="#FF0000")
        
        human_coord_label = MathTex(
            r"[\mathbf{x}]_{\text{Human}} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}",
            color="#FF0000", font_size=28
        )
        # Fix Issue 37: B1, scale 0.7
        self.place_at_grid(human_coord_label, "B1", scale_factor=0.7)
        
        self.play(Create(plane))
        self.play(FadeIn(dot), Write(human_coord_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Animate the standard grid warping into the Owl's tilted grid #FFFF00.
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#FFFF00")
        
        # Define transformation matrix P (Owl basis in Human coordinates)
        matrix_P = [[1, 0.5], [0.5, 1]]
        
        # Owl coordinate label
        owl_coord_label = MathTex(
            r"[\mathbf{x}]_{\text{Owl}} = \begin{bmatrix} 2/3 \\ 2/3 \end{bmatrix}",
            color="#FFFF00", font_size=28
        )
        # Fix Issue 37: B6, scale 0.7
        self.place_at_grid(owl_coord_label, "B6", scale_factor=0.7)
        
        # Warp the grid. Note: dot is not part of the plane VGroup in the previous step.
        # To make the grid warp while the dot stays in the same place in "world space":
        # The plane's c2p(1,1) will move. We want the dot to stay at the same pixel position.
        
        self.play(
            plane.animate.apply_matrix(matrix_P).set_color("#FFFF00"),
            human_coord_label.animate.set_opacity(0.4),
            run_time=3
        )
        
        self.play(Write(owl_coord_label))
        self.wait(3)
        
        # Finalize
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
