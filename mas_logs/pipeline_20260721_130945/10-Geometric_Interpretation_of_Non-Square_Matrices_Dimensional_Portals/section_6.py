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
        # Setup Title and Lecture Lines
        title_text = "Summary: The Column Space Perspective"
        lecture_lines = [
            "Matrices transform vectors into a destination subspace.",
            "This is just moving between different coordinate systems.",
            "Dimensions are portals to new geometric worlds."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_VEC1 = "#FF0000"
        COLOR_VEC2 = "#00FF00"
        COLOR_SPAN = "#444444"
        COLOR_PORTAL = "#8888FF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Visual: Two vectors in 3D space with a shaded region indicating the subspace they span.
        
        # Create a simulated 3D axes
        # Origin at C3
        axis_x = Arrow(self.grid["C3"], self.grid["C5"], buff=0, color=WHITE, stroke_width=2)
        axis_y = Arrow(self.grid["C3"], self.grid["A3"], buff=0, color=WHITE, stroke_width=2)
        axis_z = Arrow(self.grid["C3"], self.grid["B4"], buff=0, color=WHITE, stroke_width=2) # Perspective
        axes_3d = VGroup(axis_x, axis_y, axis_z)
        
        # Red and Green vectors representing basis of the column space
        vec1 = Arrow(self.grid["C3"], self.grid["C4"], buff=0, color=COLOR_VEC1)
        vec2 = Arrow(self.grid["C3"], self.grid["B3"], buff=0, color=COLOR_VEC2)
        
        # Shaded parallelogram for the span
        span_poly = Polygon(
            self.grid["C3"],
            self.grid["C4"],
            self.grid["B4"],
            self.grid["B3"],
            color=COLOR_SPAN, fill_opacity=0.4, stroke_width=1
        )

        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Create(axes_3d), run_time=1)
        self.play(GrowArrow(vec1), GrowArrow(vec2))
        self.play(FadeIn(span_poly))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: Morph a 2D coordinate pair into its corresponding 3D position
        coord_2d = MathTex(r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}", color=WHITE)
        # Fix: Positioning coord_2d at C4 and label_3d at B4 for alignment and proximity (Issue 34, 35)
        self.place_at_grid(coord_2d, "C4", scale_factor=0.8)
        
        # Target point on the span (roughly in the middle of the parallelogram)
        # Use a point halfway between the grid points to stay on the plane
        dot_pos = (self.grid["C3"] + self.grid["B4"]) / 2
        dot_3d = Dot(point=dot_pos, color=HIGHLIGHT_COLOR)
        
        label_3d = MathTex(r"A\mathbf{x}", color=HIGHLIGHT_COLOR)
        self.place_at_grid(label_3d, "B4", scale_factor=0.7)

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.play(Write(coord_2d))
        self.wait(0.5)
        self.play(
            ReplacementTransform(coord_2d, dot_3d),
            FadeIn(label_3d)
        )
        self.play(Indicate(dot_3d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visual: Split-screen split
        
        # Clear previous elements
        self.play(
            FadeOut(axes_3d), FadeOut(vec1), FadeOut(vec2), 
            FadeOut(span_poly), FadeOut(dot_3d), FadeOut(label_3d)
        )
        
        # Portals
        portal_l = RoundedRectangle(height=1.5, width=1.0, color=COLOR_PORTAL)
        self.place_in_area(portal_l, "C2", "E2", scale_factor=1.0)
        label_l = Text("3x2 Portal", font_size=18, color=COLOR_PORTAL)
        # Move label from A2 to B2 (L003: Row A is for titles)
        self.place_at_grid(label_l, "B2", scale_factor=0.8)
        
        portal_r = RoundedRectangle(height=1.5, width=1.0, color=COLOR_PORTAL)
        self.place_in_area(portal_r, "C5", "E5", scale_factor=1.0)
        label_r = Text("2x3 Portal", font_size=18, color=COLOR_PORTAL)
        # Move label from A5 to B5 (L003: Row A is for titles)
        self.place_at_grid(label_r, "B5", scale_factor=0.8)

        # Assets integration (Issue 20, L009)
        pixel_cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        self.place_at_grid(pixel_cat, "E2", scale_factor=0.5)
        
        bird_3d = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg")
        self.place_at_grid(bird_3d, "C5", scale_factor=0.6)
        
        # Shadow for bird
        shadow_2d = bird_3d.copy()
        shadow_2d.set_color(GRAY)
        shadow_2d.set_fill_opacity(0.5)
        shadow_2d.stretch(0.1, dim=1) 
        self.place_at_grid(shadow_2d, "E5", scale_factor=0.6)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.play(
            Create(portal_l), Create(portal_r),
            Write(label_l), Write(label_r)
        )
        self.play(FadeIn(pixel_cat), FadeIn(bird_3d))
        self.wait(0.5)
        
        # Pixel jumps into portal
        jump_path = ArcBetweenPoints(pixel_cat.get_center(), portal_l.get_center(), angle=-TAU/4)
        self.play(MoveAlongPath(pixel_cat, jump_path), run_time=1.5)
        self.play(FadeOut(pixel_cat, shift=IN))
        
        # Bird casts shadow
        self.play(FadeIn(shadow_2d, shift=DOWN))
        self.play(Indicate(shadow_2d))
        
        self.wait(2)
