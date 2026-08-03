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
        self.setup_layout("The Parallel-Transposition Trick", [
            "Consider the parallelogram formed by vectors W and B.",
            "Since yB is parallel to B, its area is zero.",
            "Only the xA component contributes to this new area."
        ])
        
        # Grid Coordinates Reference
        # D2: Origin
        # D4: End of vector B
        # B3: End of vector W
        # B5: End of vector W+B
        # B2: End of vector xA
        # B4: End of vector xA+B
        
        origin = self.grid["D2"]
        b_end = self.grid["D4"]
        w_end = self.grid["B3"]
        wb_end = self.grid["B5"]
        xa_end = self.grid["B2"]
        xab_end = self.grid["B4"]
        
        # Asset integration: Background Grid
        grid_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_svg, "A1", "F6", scale_factor=5.0)
        grid_svg.set_opacity(0.2)
        self.add(grid_svg)

        # === Animation for Lecture Line 1 ===
        # Consider the parallelogram formed by vectors W and B.
        # Matching color: #FFAAAA (Parallelogram)
        
        vec_b = Arrow(origin, b_end, buff=0, color="#0000FF")
        vec_w = Arrow(origin, w_end, buff=0, color="#FF0000")
        label_b = MathTex(r"\vec{B}", color="#0000FF")
        self.place_at_grid(label_b, "E3", scale_factor=0.8)
        
        # Issue 23: Move label_w to B2
        label_w = MathTex(r"\vec{W}", color="#FF0000")
        self.place_at_grid(label_w, "B2", scale_factor=0.8)
        
        parallelogram = Polygon(
            origin, b_end, wb_end, w_end, 
            fill_color="#FFAAAA", fill_opacity=0.5, stroke_width=2, stroke_color="#FFAAAA"
        )
        
        self.lecture[0].set_color("#FFAAAA")
        self.play(GrowArrow(vec_b), GrowArrow(vec_w), FadeIn(label_b), FadeIn(label_w))
        self.play(FadeIn(parallelogram))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Since yB is parallel to B, its area is zero.
        # Matching color: #FFFF00
        
        # Decomposition of W: W = xA + yB
        # xA is vertical (D2 to B2), yB is horizontal (B2 to B3)
        vec_xa = Arrow(origin, xa_end, buff=0, color="#FFFFFF")
        vec_yb = Arrow(xa_end, w_end, buff=0, color="#FFFF00")
        
        # Issue 24: Move label_yb to B3
        label_yb = MathTex(r"y\vec{B}", color="#FFFF00")
        self.place_at_grid(label_yb, "B3", scale_factor=0.7)
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        self.play(Create(vec_xa), Create(vec_yb), FadeIn(label_yb))
        self.play(Indicate(vec_yb), Indicate(vec_b))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Only the xA component contributes to this new area.
        # Matching color: #00FF00
        
        # New parallelogram spanned by xA and B
        para_xa_b = Polygon(
            origin, b_end, xab_end, xa_end, 
            fill_color="#00FF00", fill_opacity=0.5, stroke_width=2, stroke_color="#00FF00"
        )
        
        # Issue 22: Move label_xa to C2
        label_xa = MathTex(r"x\vec{A}", color="#FFFFFF")
        self.place_at_grid(label_xa, "C2", scale_factor=0.7)

        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        self.play(
            FadeOut(parallelogram),
            FadeOut(vec_w),
            FadeOut(label_w),
            FadeIn(para_xa_b),
            FadeIn(label_xa)
        )
        
        self.wait(2)
