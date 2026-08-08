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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data
        title = "The 'Shear' Insight for x"
        lines = [
            "To find x, consider the area (b, v2).",
            "Substitute b with x v1 plus y v2.",
            "The y v2 component is parallel to v2.",
            "Shearing along v2 does not change the area.",
            "This leaves only the area of x v1."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_B = "#FF00FF"
        COLOR_V2 = "#00FF00"
        COLOR_V1 = "#FFFF00"
        COLOR_XV1 = "#FFCC00"
        COLOR_YV2 = "#00DD00"
        
        # Coordinate System setup
        origin_pos = self.grid["D3"]
        
        # Vectors
        v2_raw = np.array([0.5, 1.5, 0])
        v1_raw = np.array([1.5, 0.4, 0])
        x_val = 1.3
        y_val = 0.7
        
        xv1_raw = x_val * v1_raw
        yv2_raw = y_val * v2_raw
        b_raw = xv1_raw + yv2_raw
        
        # Create Mobjects
        v2 = Arrow(origin_pos, origin_pos + v2_raw, buff=0, color=COLOR_V2)
        v2_label = MathTex("\\vec{v}_2", color=COLOR_V2)
        self.place_at_grid(v2_label, "C3", scale_factor=0.8) # Issue 28: Fixed v2_label position
        
        b = Arrow(origin_pos, origin_pos + b_raw, buff=0, color=COLOR_B)
        b_label = MathTex("\\vec{b}", color=COLOR_B)
        self.place_at_grid(b_label, "B6", scale_factor=0.8) # Issue 29: Fixed b_label position
        
        para_b_v2 = Polygon(
            origin_pos,
            origin_pos + v2_raw,
            origin_pos + v2_raw + b_raw,
            origin_pos + b_raw,
            stroke_width=2,
            fill_opacity=0.3,
            fill_color=COLOR_B
        )
        
        # Equation for total area
        eq_final = MathTex("\\text{Area}(\\vec{b}, \\vec{v}_2) = x \\cdot \\text{Area}(\\vec{v}_1, \\vec{v}_2)", color=WHITE)
        self.place_in_area(eq_final, "F2", "F5", scale_factor=0.8) # Issue 27: Fixed eq_final position

        # === Animation for Lecture Line 1 ===
        # "To find x, consider the area (b, v2)."
        self.play(self.lecture[0].animate.set_color(COLOR_B))
        self.play(Create(v2), Create(v2_label))
        self.play(Create(b), Create(b_label))
        self.play(FadeIn(para_b_v2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Substitute b with x v1 plus y v2."
        self.play(self.lecture[1].animate.set_color(COLOR_V1))
        
        xv1 = Arrow(origin_pos, origin_pos + xv1_raw, buff=0, color=COLOR_XV1)
        xv1_label = MathTex("x\\vec{v}_1", color=COLOR_XV1)
        self.place_at_grid(xv1_label, "E5", scale_factor=0.7)
        
        yv2 = Arrow(origin_pos + xv1_raw, origin_pos + xv1_raw + yv2_raw, buff=0, color=COLOR_YV2)
        yv2_label = MathTex("y\\vec{v}_2", color=COLOR_YV2)
        self.place_at_grid(yv2_label, "C5", scale_factor=0.7) # Grid-aligned label
        
        self.play(Create(xv1), FadeIn(xv1_label))
        self.play(Create(yv2), FadeIn(yv2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The y v2 component is parallel to v2."
        self.play(self.lecture[2].animate.set_color(COLOR_V2))
        
        # Highlight parallelism by sliding yv2 to the origin
        yv2_ghost = yv2.copy().set_color(COLOR_V2).set_stroke(opacity=0.5)
        self.play(yv2_ghost.animate.shift(origin_pos - (origin_pos + xv1_raw)), run_time=1.5)
        self.play(Indicate(yv2), Indicate(v2))
        self.play(FadeOut(yv2_ghost))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Shearing along v2 does not change the area."
        self.play(self.lecture[3].animate.set_color(COLOR_V2))
        
        # Define the sheared parallelogram (xv1, v2)
        para_xv1_v2 = Polygon(
            origin_pos,
            origin_pos + v2_raw,
            origin_pos + v2_raw + xv1_raw,
            origin_pos + xv1_raw,
            stroke_width=2,
            fill_opacity=0.3,
            fill_color=COLOR_XV1
        )
        
        # Animate the shear: top side slides from (b + v2) to (xv1 + v2)
        self.play(
            Transform(para_b_v2, para_xv1_v2),
            b.animate.set_points_as_corners([origin_pos, origin_pos + xv1_raw]).set_color(COLOR_XV1),
            FadeOut(yv2), FadeOut(yv2_label), FadeOut(b_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This leaves only the area of x v1."
        self.play(self.lecture[4].animate.set_color(COLOR_XV1))
        
        self.play(Write(eq_final))
        self.play(Circumscribe(para_b_v2), Circumscribe(eq_final))
        self.wait(2)

        # Cleanup
        self.play(
            FadeOut(v2), FadeOut(v2_label), 
            FadeOut(xv1), FadeOut(xv1_label), 
            FadeOut(para_b_v2), FadeOut(eq_final)
        )
        self.wait(1)
