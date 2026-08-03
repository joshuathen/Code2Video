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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visual Summary and Application", [
            "Change of basis reinterprets the same vector.",
            "Matrix P and its inverse enable two-way communication.",
            "This is essential for modern computer graphics.",
            "JPEG compression uses this to store images efficiently.",
            "Different bases offer unique insights into the world."
        ])
        
        # Define colors
        COLOR_VECTOR = YELLOW
        COLOR_CAMERA = BLUE
        COLOR_GRAPHICS = "#00FFFF"
        COLOR_JPEG = "#FF00FF"
        COLOR_FORMULA = WHITE
        
        # === Animation for Lecture Line 1 ===
        # Change of basis reinterprets the same vector.
        self.lecture[0].set_color(COLOR_VECTOR)
        
        # Visual: A vector in a grid
        vector_origin = self.grid["D3"]
        vector_end = self.grid["C4"]
        vector_v = Arrow(vector_origin, vector_end, buff=0, color=COLOR_VECTOR)
        v_label = MathTex(r"\vec{v}", color=COLOR_VECTOR).next_to(vector_end, UR, buff=0.1)
        
        self.play(Create(vector_v), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matrix P and its inverse enable two-way communication.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        p_matrix = MathTex(r"P", color=YELLOW)
        p_inv_matrix = MathTex(r"P^{-1}", color=YELLOW)
        
        # Resolve Issue 42: Balanced positioning of P and P^-1
        self.place_at_grid(p_matrix, "B2", scale_factor=1.2)
        self.place_at_grid(p_inv_matrix, "B5", scale_factor=1.2)
        
        communication_arrow = DoubleArrow(p_matrix.get_right(), p_inv_matrix.get_left(), color=WHITE)
        
        self.play(FadeIn(p_matrix, p_inv_matrix, communication_arrow))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # This is essential for modern computer graphics.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_GRAPHICS)
        
        # Clear previous objects
        self.play(FadeOut(p_matrix, p_inv_matrix, communication_arrow, vector_v, v_label))
        
        # Resolve Issue 28: Integrated camera asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg]
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        camera.set_color(COLOR_CAMERA)
        self.place_at_grid(camera, "C2", scale_factor=0.5)
        
        # Grid representing camera view
        camera_grid = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1}
        ).scale(0.5).move_to(camera.get_center())
        
        # Group camera and its local coordinate system
        camera_view = VGroup(camera, camera_grid)
        
        # World object (fixed tree)
        tree_trunk = Rectangle(width=0.1, height=0.3, color=GREEN_E, fill_opacity=1)
        tree_top = Circle(radius=0.2, color=GREEN, fill_opacity=1).next_to(tree_trunk, UP, buff=0)
        tree = VGroup(tree_trunk, tree_top)
        self.place_at_grid(tree, "C4", scale_factor=1.0)
        
        graphics_text = Text("Computer Graphics", color=COLOR_GRAPHICS, font_size=24)
        self.place_at_grid(graphics_text, "D3", scale_factor=1.0)
        
        self.play(FadeIn(camera_view, tree, graphics_text))
        
        # Rotate camera/grid while world objects (tree) stay fixed
        rotation_center = tree.get_center()
        self.play(
            Rotate(camera_view, angle=PI/4, about_point=rotation_center, run_time=2),
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # JPEG compression uses this to store images efficiently.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_JPEG)
        
        jpeg_text = Text("JPEG Compression", color=COLOR_JPEG, font_size=24)
        
        # Resolve Issue 43: Prevent vertical crowding by moving JPEG text to F3
        self.place_at_grid(jpeg_text, "F3", scale_factor=1.0)
        
        self.play(
            FadeOut(graphics_text),
            FadeIn(jpeg_text)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Different bases offer unique insights into the world.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        final_formula = MathTex(r"[\vec{v}]_A = P [\vec{v}]_B", color=COLOR_FORMULA)
        
        # Resolve Issue 44: Centered and prominent final formula
        self.place_in_area(final_formula, "B2", "E5", scale_factor=1.5)
        
        self.play(
            FadeOut(camera_view),
            FadeOut(tree),
            FadeOut(jpeg_text),
            run_time=1
        )
        self.play(Write(final_formula))
        self.wait(3)
