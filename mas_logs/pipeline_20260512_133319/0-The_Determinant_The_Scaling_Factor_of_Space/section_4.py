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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup with correct lecture lines
        self.setup_layout(
            "The Formula: ad - bc",
            [
                'We calculate this area using the formula: ad minus bc.',
                "These variables map directly to the parallelogram's geometry.",
                'Subtracting the excess pieces yields the precise area.'
            ]
        )

        # Colors
        COLOR_A = "#58C4DD" # Blue
        COLOR_D = "#58C4DD" # Blue
        COLOR_B = "#FC6255" # Red
        COLOR_C = "#FC6255" # Red
        COLOR_PARA = "#77B05D" # Green
        COLOR_BOX = GRAY_B

        # === Animation for Lecture Line 1 ===
        # Formula: det(A) = ad - bc
        f1 = Text("det(A) = ", font_size=32)
        f2 = Text("a", color=COLOR_A, font_size=32)
        f3 = Text("d", color=COLOR_D, font_size=32)
        f4 = Text(" - ", font_size=32)
        f5 = Text("b", color=COLOR_B, font_size=32)
        f6 = Text("c", color=COLOR_C, font_size=32)
        formula = VGroup(f1, f2, f3, f4, f5, f6).arrange(RIGHT, buff=0.1)
        
        # [Issue 41/43 Fix]: Position at A4, scale 0.9
        self.place_at_grid(formula, 'A4', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_A)
        
        # [Issue 30]: Asset integration
        try:
            para_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/parallelogram.svg")
            self.place_at_grid(para_icon, 'B6', scale_factor=0.4)
            self.play(FadeIn(para_icon))
        except:
            # Fallback if asset is missing in local dev environment
            para_icon = RegularPolygon(n=4, color=COLOR_PARA).scale(0.2)
            self.place_at_grid(para_icon, 'B6')

        # Geometry Setup
        # Matrix [a, c; b, d] -> vectors (a,b) and (c,d)
        a, b, c, d = 2.0, 0.8, 0.8, 2.0
        s = 0.6  # Scene scale factor
        
        # Anchor the geometry in the central grid area C3-E5
        geo_center = self.grid['D4']
        offset = np.array([-(a+c)*s/2, -(b+d)*s/2, 0])
        origin = geo_center + offset

        p0 = origin
        p1 = origin + np.array([a*s, b*s, 0])
        p2 = origin + np.array([c*s, d*s, 0])
        p3 = origin + np.array([(a+c)*s, (b+d)*s, 0])
        
        parallelogram = Polygon(p0, p1, p3, p2, color=COLOR_PARA, fill_opacity=0.3)
        
        # Labels for a, b, c, d
        label_a = Text("a", color=COLOR_A, font_size=20).next_to(origin + np.array([a*s/2, 0, 0]), DOWN, buff=0.1)
        label_b = Text("b", color=COLOR_B, font_size=20).next_to(p1, LEFT, buff=0.1)
        label_c = Text("c", color=COLOR_C, font_size=20).next_to(p2, DOWN, buff=0.1)
        label_d = Text("d", color=COLOR_D, font_size=20).next_to(origin + np.array([0, d*s/2, 0]), LEFT, buff=0.1)
        
        # Guide lines for components
        line_a = Line(origin, origin + np.array([a*s, 0, 0]), color=COLOR_A)
        line_d = Line(origin, origin + np.array([0, d*s, 0]), color=COLOR_D)
        
        self.play(
            Create(parallelogram),
            Create(line_a),
            Create(line_d),
            FadeIn(label_a), FadeIn(label_b),
            FadeIn(label_c), FadeIn(label_d)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_B)
        
        # Bounding box and excess pieces
        box_p1 = origin + np.array([(a+c)*s, 0, 0])
        box_p2 = origin + np.array([(a+c)*s, (b+d)*s, 0])
        box_p3 = origin + np.array([0, (b+d)*s, 0])
        bounding_box = DashedVMobject(Polygon(origin, box_p1, box_p2, box_p3, color=COLOR_BOX))
        
        # Excess triangles and rectangles
        t_ab1 = Polygon(origin, origin + np.array([a*s, 0, 0]), p1, color=COLOR_B, fill_opacity=0.5, stroke_width=0)
        t_ab2 = Polygon(p2, origin + np.array([c*s, (b+d)*s, 0]), p3, color=COLOR_B, fill_opacity=0.5, stroke_width=0)
        
        t_cd1 = Polygon(origin, origin + np.array([0, d*s, 0]), p2, color=COLOR_C, fill_opacity=0.5, stroke_width=0)
        t_cd2 = Polygon(p1, origin + np.array([(a+c)*s, b*s, 0]), p3, color=COLOR_C, fill_opacity=0.5, stroke_width=0)
        
        r_bc1 = Rectangle(width=c*s, height=b*s, color=WHITE, fill_opacity=0.3, stroke_width=0).move_to(origin + np.array([(a + c/2)*s, (b/2)*s, 0]))
        r_bc2 = Rectangle(width=c*s, height=b*s, color=WHITE, fill_opacity=0.3, stroke_width=0).move_to(origin + np.array([(c/2)*s, (d + b/2)*s, 0]))
        
        self.play(Create(bounding_box))
        self.play(FadeIn(t_ab1), FadeIn(t_ab2), FadeIn(t_cd1), FadeIn(t_cd2), FadeIn(r_bc1), FadeIn(r_bc2))
        self.wait(1)
        
        # Calculation text
        calc_text = Text("Area = ad - bc", font_size=24, color=WHITE)
        # [Issue 42 Fix]: Position at F4, scale 0.8
        self.place_at_grid(calc_text, 'F4', scale_factor=0.8)
        self.play(Write(calc_text))
        self.wait(2)
