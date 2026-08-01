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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup the layout with section title and lecture lines
        self.setup_layout(
            "Tall Matrices: Upgrading Dimensions (2D to 3D)",
            [
                "A three-by-two matrix takes two-D inputs to three-D.",
                "We are upgrading from a flat plane to volume.",
                "The original plane becomes a slanted sheet in space.",
                "Pixel now exists on a tilted three-D surface.",
                "His height depends on his original two-D position."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Show a 3x2 matrix A = [[1, 0], [0, 1], [1, 1]]
        # This matrix takes 2D (2 columns) to 3D (3 rows)
        matrix_a = Matrix([[1, 0], [0, 1], [1, 1]], 
                         left_bracket="[", right_bracket="]",
                         element_to_mobject_config={"color": WHITE})
        matrix_a.set_color(WHITE)
        
        # Label for the matrix
        matrix_label = MathTex("A =", color=WHITE).scale(0.8)
        matrix_label.next_to(matrix_a, LEFT)
        matrix_group = VGroup(matrix_label, matrix_a)
        
        # Positioning according to grid rules - Resolved Issue 26
        # Using Area B2-D3 to avoid overlap with upcoming 3D visualization
        self.place_in_area(matrix_group, "B2", "D3", scale_factor=0.8)
        
        self.play(FadeIn(matrix_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Start with a flat 2D grid to show the starting dimension
        grid_2d = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            x_length=3.5, y_length=3.5,
            background_line_style={"stroke_color": "#444444", "stroke_width": 1}
        )
        self.place_in_area(grid_2d, "C4", "F6", scale_factor=0.8)
        
        # Define 3D Axes for the volume view (introducing 3rd axis)
        axes = ThreeDAxes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[-2, 2, 1],
            x_length=3.5, y_length=3.5, z_length=3.5,
            axis_config={"stroke_color": "#888888", "stroke_width": 2}
        )
        # Apply orientation to simulate 3D perspective in the 2D scene
        axes.rotate(75 * DEGREES, axis=RIGHT).rotate(-45 * DEGREES, axis=OUT)
        # Set target position for axes - Resolved Issue 27
        self.place_in_area(axes, "C4", "F6", scale_factor=0.8)

        self.play(Create(grid_2d))
        self.wait(0.5)
        
        # Transition from 2D plane to 3D volume view
        self.play(
            matrix_group.animate.scale(0.7).move_to(self.grid["B2"]),
            ReplacementTransform(grid_2d, axes),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Visualize column vectors of the matrix: v1=[1,0,1], v2=[0,1,1]
        # These span the 'slanted sheet' in 3D space
        origin = axes.get_origin()
        v1_end = axes.c2p(1, 0, 1)
        v2_end = axes.c2p(0, 1, 1)
        
        v1 = Arrow(origin, v1_end, buff=0, color="#00FF00", stroke_width=4)
        v2 = Arrow(origin, v2_end, buff=0, color="#00FF00", stroke_width=4)
        
        # Tilted plane segment spanned by the columns
        p1 = axes.c2p(-1.5, -1.5, -3)
        p2 = axes.c2p(1.5, -1.5, 0)
        p3 = axes.c2p(1.5, 1.5, 3)
        p4 = axes.c2p(-1.5, 1.5, 0)
        
        plane_sheet = Polygon(p1, p2, p3, p4, color="#00FF00", fill_opacity=0.3, stroke_width=0)
        
        self.play(GrowArrow(v1), GrowArrow(v2))
        self.play(FadeIn(plane_sheet))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Pixel the Cat represented as a circle icon on the tilted plane
        # Position: x=0.8, y=0.4 => z = x + y = 1.2
        pixel_pos = axes.c2p(0.8, 0.4, 1.2)
        pixel_cat = Circle(radius=0.15, color="#FFD700", fill_opacity=1.0).move_to(pixel_pos)
        
        # Label for Pixel (positioned near the object per L003)
        pixel_text = Text("Pixel", font_size=20, color="#FFD700")
        pixel_text.next_to(pixel_cat, UP, buff=0.15)
        
        self.play(FadeIn(pixel_cat), Write(pixel_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Vertical dashed line highlighting height (Z) relative to XY ground
        xy_proj = axes.c2p(0.8, 0.4, 0)
        height_line = DashedLine(pixel_pos, xy_proj, color=WHITE, stroke_width=3)
        
        # Mathematical label explaining the Z-dependence
        height_label = MathTex("z = x + y", font_size=24, color=WHITE)
        # Position label carefully to avoid crowding and grid boundaries (L015)
        height_label.next_to(height_line, RIGHT, buff=0.2)
        
        self.play(Create(height_line))
        self.play(Write(height_label))
        
        # Mark issues as resolved
        # update_issue(26, under_review=True, resolution_note="Fixed matrix_group positioning to avoid 3D axes overlap.")
        # update_issue(27, under_review=True, resolution_note="Adjusted ThreeDAxes area to F6 and scaled for better visibility.")
        
        self.wait(3)

# Issue Resolution Calls (Internal record)
# update_issue(26, under_review=True, resolution_note="Relocated matrix_group to area B2-D3 and scaled to 0.8 to prevent overlap with axes.")
# update_issue(27, under_review=True, resolution_note="Repositioned 3D axes to area C4-F6 to utilize bottom-right space effectively.")
