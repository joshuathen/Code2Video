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
        self.setup_layout(
            "Deriving the Ratio (Cramer's Formula)", 
            [
                "Area(b, v2) equals x times Area(v1, v2).",
                "Solving for x gives a ratio of areas.",
                "This is the determinant Ab over det A.",
                "Ab replaces the first column with vector b.",
                "We have geometrically derived Cramer's Rule!"
            ]
        )
        
        # Colors
        COLOR_V1 = RED
        COLOR_V2 = GREEN
        COLOR_B = BLUE
        COLOR_AREA_A = YELLOW
        COLOR_AREA_AB = ORANGE
        COLOR_MATH_1 = YELLOW_A
        COLOR_MATH_2 = PINK
        COLOR_MATH_3 = BLUE_A
        COLOR_MATH_4 = GOLD
        COLOR_MATH_5 = GREEN_A

        # Coordinate System for Geometry
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 8, 1],
            x_length=3.5,
            y_length=4.0,
            axis_config={"include_tip": True, "font_size": 18}
        ).add_coordinates()
        
        # Place axes in A1 to D3 area
        self.place_in_area(axes, 'A1', 'D3', scale_factor=0.9)

        # Vectors and Parallelograms
        # v1 = [2, 1], v2 = [1, 2], b = [4, 5]
        v1_vec = Arrow(axes.c2p(0, 0), axes.c2p(2, 1), color=COLOR_V1, buff=0)
        v2_vec = Arrow(axes.c2p(0, 0), axes.c2p(1, 2), color=COLOR_V2, buff=0)
        b_vec = Arrow(axes.c2p(0, 0), axes.c2p(4, 5), color=COLOR_B, buff=0)
        
        # Parallelogram (v1, v2)
        poly_a = Polygon(
            axes.c2p(0, 0), axes.c2p(2, 1), axes.c2p(3, 3), axes.c2p(1, 2),
            color=COLOR_AREA_A, fill_opacity=0.4, stroke_width=2
        )
        # Parallelogram (b, v2)
        poly_ab = Polygon(
            axes.c2p(0, 0), axes.c2p(4, 5), axes.c2p(5, 7), axes.c2p(1, 2),
            color=COLOR_AREA_AB, fill_opacity=0.3, stroke_width=2
        )

        v1_label = MathTex(r"\vec{v}_1", color=COLOR_V1, font_size=20)
        v2_label = MathTex(r"\vec{v}_2", color=COLOR_V2, font_size=20)
        b_label = MathTex(r"\vec{b}", color=COLOR_B, font_size=20)
        
        # Grid placement for labels relative to geometric objects
        self.place_at_grid(v1_label, 'C3', scale_factor=1.0)
        self.place_at_grid(v2_label, 'A2', scale_factor=1.0)
        self.place_at_grid(b_label, 'A4', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_MATH_1))
        self.play(Create(axes), Create(v1_vec), Create(v2_vec), Create(b_vec))
        self.play(Write(v1_label), Write(v2_label), Write(b_label))
        self.play(FadeIn(poly_a), FadeIn(poly_ab))
        
        eq1 = MathTex(r"\text{Area}(\vec{b}, \vec{v}_2) = x \cdot \text{Area}(\vec{v}_1, \vec{v}_2)", color=COLOR_MATH_1)
        self.place_in_area(eq1, 'A4', 'A6', scale_factor=0.6)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_MATH_2))
        eq2 = MathTex(r"x = \frac{\text{Area}(\vec{b}, \vec{v}_2)}{\text{Area}(\vec{v}_1, \vec{v}_2)}", color=COLOR_MATH_2)
        self.place_in_area(eq2, 'B4', 'B6', scale_factor=0.6)
        self.play(Write(eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_MATH_3))
        eq3 = MathTex(r"x = \frac{\det(A_b)}{\det(A)}", color=COLOR_MATH_3)
        self.place_in_area(eq3, 'C4', 'C6', scale_factor=0.6)
        self.play(Write(eq3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_MATH_4))
        eq4 = MathTex(
            r"x = \frac{\det\left(\begin{bmatrix} 4 & 1 \\ 5 & 2 \end{bmatrix}\right)}{\det\left(\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\right)}",
            color=COLOR_MATH_4
        )
        self.place_in_area(eq4, 'D4', 'E6', scale_factor=0.6)
        self.play(Write(eq4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_MATH_5))
        eq5 = MathTex(r"x = \frac{3}{3} = 1", color=COLOR_MATH_5)
        self.place_in_area(eq5, 'F4', 'F6', scale_factor=0.6)
        self.play(Write(eq5))
        
        # Final highlight
        res_box = SurroundingRectangle(eq5, color=WHITE, buff=0.1)
        self.play(Create(res_box))
        self.play(Indicate(eq5, color=WHITE))
        self.play(FadeOut(res_box))
        self.wait(2)
