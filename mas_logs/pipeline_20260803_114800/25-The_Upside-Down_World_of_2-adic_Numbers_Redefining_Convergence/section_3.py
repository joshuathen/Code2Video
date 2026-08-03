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
        self.setup_layout(
            "A New Measuring Stick: The 2-adic Metric",
            [
                "2-adic distance depends on shared factors of two.",
                "High divisibility means numbers are extremely close.",
                "On a standard ruler, zero and sixty-four are far.",
                "On a 2-adic ruler, they are very close.",
                "This new metric forms a fractal-like number structure."
            ]
        )

        # Define specific hex colors for consistency and visibility
        COLOR_1 = "#FFA500" # Orange
        COLOR_2 = "#FFFF00" # Yellow
        COLOR_3 = "#ADD8E6" # Light Blue
        COLOR_4 = "#90EE90" # Light Green
        COLOR_5 = "#FFB6C1" # Light Pink
        
        RULER_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_1)
        
        formula = MathTex(
            r"d_2(x, y) =", r"\frac{1}{2^{", r"v_2(x-y)", r"}}",
            color=COLOR_1
        )
        # Resolved Issue 27: Scale formula to 0.9 and place in A2-B5
        self.place_in_area(formula, 'A2', 'B5', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_2)
        v2_term = formula[2]
        
        self.play(v2_term.animate.set_color(COLOR_2))
        # Pulsing the v_2 term to emphasize divisibility
        self.play(Indicate(v2_term, color=COLOR_2, scale_factor=1.4))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_3)
        
        # Standard Ruler visual using Asset
        # Resolved Issue 20 & 28: Use Asset and place in C2-D5
        std_ruler_icon = SVGMobject(RULER_ASSET, color=WHITE, height=0.5)
        std_line = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        zero_dot_std = Dot(std_line.get_start(), color=COLOR_3)
        sixty_four_dot_std = Dot(std_line.get_end(), color=COLOR_3)
        zero_lbl_std = Text("0", font_size=18).next_to(zero_dot_std, DOWN, buff=0.1)
        sixty_four_lbl_std = Text("64", font_size=18).next_to(sixty_four_dot_std, DOWN, buff=0.1)
        std_title = Text("Standard Ruler", font_size=20, color=COLOR_3).next_to(std_line, UP, buff=0.3)
        
        standard_ruler_group = VGroup(std_line, std_ruler_icon, std_title, zero_dot_std, sixty_four_dot_std, zero_lbl_std, sixty_four_lbl_std)
        self.place_in_area(standard_ruler_group, 'C2', 'D5', scale_factor=0.8)
        
        self.play(FadeIn(standard_ruler_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_4)
        
        # 2-adic Ruler visual using Asset
        # Resolved Issue 20 & 29: Use Asset and place in E2-F5
        adic_ruler_icon = SVGMobject(RULER_ASSET, color=WHITE, height=0.5)
        adic_line = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        # In 2-adic distance, d(0, 64) = 1/64, which is tiny. 
        # Position 0 at center, 64 very close to it.
        zero_pos_adic = ORIGIN
        sixty_four_pos_adic = RIGHT * 0.08
        
        zero_dot_adic = Dot(zero_pos_adic, radius=0.06, color=COLOR_4)
        sixty_four_dot_adic = Dot(sixty_four_pos_adic, radius=0.06, color=COLOR_4)
        
        zero_lbl_adic = Text("0", font_size=16).next_to(zero_dot_adic, DOWN, buff=0.1)
        sixty_four_lbl_adic = Text("64", font_size=16).next_to(sixty_four_dot_adic, UP, buff=0.1)
        adic_title = Text("2-adic Ruler", font_size=20, color=COLOR_4).next_to(adic_line, UP, buff=0.3)

        adic_ruler_group = VGroup(adic_line, adic_ruler_icon, adic_title, zero_dot_adic, sixty_four_dot_adic, zero_lbl_adic, sixty_four_lbl_adic)
        self.place_in_area(adic_ruler_group, 'E2', 'F5', scale_factor=0.8)
        
        self.play(FadeIn(adic_ruler_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_5)
        
        # Visualize the fractal-like structure: points 2, 4, 8, 16 clustering near 0
        points = [2, 4, 8, 16]
        cluster_elements = VGroup()
        
        # We need to position these relative to the zero_dot_adic in world coordinates
        origin_center = zero_dot_adic.get_center()
        
        for val in points:
            # d_2(0, val) = 1/val. Scale factor 2.0 for visibility
            offset_x = (1.0 / val) * 2.5 
            pos = origin_center + RIGHT * offset_x
            dot = Dot(pos, radius=0.05, color=COLOR_5)
            lbl = Text(str(val), font_size=14, color=COLOR_5).next_to(dot, DOWN, buff=0.05)
            cluster_elements.add(VGroup(dot, lbl))

        self.play(
            LaggedStart(
                *[FadeIn(el) for el in cluster_elements],
                lag_ratio=0.3
            )
        )
        
        # Glow/Pulse cluster to show fractal structure
        self.play(
            *[Indicate(el[0], color=COLOR_5, scale_factor=1.5) for el in cluster_elements],
            run_time=2
        )
        
        self.wait(2)
