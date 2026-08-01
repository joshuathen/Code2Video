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

class Section5Scene(TeachingScene):
    def construct(self):
        # Fetch storyboard data
        title = "The Fourth Dimension: Conceptual Projection"
        lines = [
            "A 3D cube casts a flat, 2D shadow.",
            "Similarly, 4D objects project into our 3D space.",
            "We study these shadows to understand higher dimensions."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CUBE = "#FFFF00"      # Yellow
        COLOR_SHADOW = "#808080"    # Grey
        COLOR_TESSERACT = "#00FFFF" # Cyan
        COLOR_HIGHLIGHT = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        # Rotate a yellow #FFFF00 cube casting a 2D grey #808080 shadow.
        self.lecture[0].set_color(COLOR_CUBE)
        
        # Wireframe Cube
        s1 = Square(side_length=1.0, color=COLOR_CUBE)
        s2 = Square(side_length=1.0, color=COLOR_CUBE).shift(0.3*RIGHT + 0.3*UP)
        l_conn = VGroup(*[Line(s1.get_vertices()[i], s2.get_vertices()[i], color=COLOR_CUBE) for i in range(4)])
        cube_wire = VGroup(s1, s2, l_conn)
        
        # Shadow (Grey)
        shadow = Polygon(
            [-0.5, -0.3, 0], [0.5, -0.3, 0], [0.8, 0.2, 0],
            [0.8, 0.8, 0], [-0.2, 0.8, 0], [-0.5, 0.3, 0],
            color=COLOR_SHADOW, fill_opacity=0.4
        ).set_fill(COLOR_SHADOW)
        
        # Layout fixes (Issue 64)
        self.place_in_area(cube_wire, 'B3', 'C4', scale_factor=0.8)
        self.place_in_area(shadow, 'D3', 'E4', scale_factor=0.8)
        
        self.play(Create(cube_wire), FadeIn(shadow))
        self.play(Rotate(cube_wire, angle=2*PI, axis=RIGHT+UP), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Similarly, 4D objects project into our 3D space.
        # Morphing Tesseract projection.
        self.lecture[1].set_color(COLOR_TESSERACT)
        
        self.play(FadeOut(cube_wire), FadeOut(shadow))
        
        # Tesseract (Schlegel diagram representation)
        inner_sq = Square(side_length=0.7, color=COLOR_TESSERACT)
        outer_sq = Square(side_length=1.8, color=COLOR_TESSERACT)
        inner_sq.shift(0.1*RIGHT + 0.1*UP) # Slight asymmetry for depth
        
        t_lines = VGroup(*[Line(inner_sq.get_vertices()[i], outer_sq.get_vertices()[i], color=COLOR_TESSERACT) for i in range(4)])
        tesseract = VGroup(inner_sq, outer_sq, t_lines)
        
        # Layout fixes (Issue 64)
        self.place_in_area(tesseract, 'B2', 'E5', scale_factor=0.9)
        
        self.play(Create(tesseract))
        
        # Updater for connected lines during morphing
        def update_t_lines(obj):
            for i in range(4):
                obj[i].put_start_and_end_on(inner_sq.get_vertices()[i], outer_sq.get_vertices()[i])
        
        t_lines.add_updater(update_t_lines)
        
        # Simulate 4D rotation by swapping inner and outer squares
        self.play(
            inner_sq.animate.scale(2.5).shift(0.2*LEFT + 0.2*DOWN),
            outer_sq.animate.scale(0.4).shift(0.1*RIGHT + 0.1*UP),
            run_time=3,
            rate_func=there_and_back
        )
        t_lines.remove_updater(update_t_lines)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We study these shadows to understand higher dimensions.
        # Flash vertices and untangle knot.
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Vertex Flash
        v_dots = VGroup(*[Dot(v, color=COLOR_HIGHLIGHT, radius=0.06) for v in inner_sq.get_vertices() + outer_sq.get_vertices()])
        self.add(v_dots)
        self.play(*[Flash(dot, color=COLOR_HIGHLIGHT, flash_radius=0.15) for dot in v_dots], run_time=1)
        self.remove(v_dots)
        
        self.play(FadeOut(tesseract))
        
        # Knot untangling using SVGMobject asset
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/knot.svg]
        knot_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/knot.svg")
        knot_asset.set_color(COLOR_TESSERACT)
        
        circle = Circle(radius=1.2, color=COLOR_TESSERACT)
        
        # Layout fixes (Issue 64)
        self.place_in_area(knot_asset, 'B2', 'E5', scale_factor=0.8)
        self.place_in_area(circle, 'B2', 'E5', scale_factor=0.8)
        
        self.play(DrawBorderThenFill(knot_asset))
        self.wait(1)
        # 4D logic "untangles" the knot into a circle
        self.play(ReplacementTransform(knot_asset, circle))
        self.wait(2)
