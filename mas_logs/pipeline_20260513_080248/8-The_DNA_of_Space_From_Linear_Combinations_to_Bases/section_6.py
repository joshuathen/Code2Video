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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Summary and Geometric Wrap-up"
        lines = [
            "Combinations are tools; Span is the territory.",
            "Independence ensures no wasted movement instructions.",
            "Together, they form the space's unique DNA."
        ]
        self.setup_layout(title, lines)
        
        yellow_color = "#FFFF00"
        blue_color = "#00BFFF"
        green_color = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Color Line 1
        self.play(self.lecture[0].animate.set_color(yellow_color))

        # Vector addition (tools) - Left Area (A1-B3)
        v1 = Arrow(start=ORIGIN, end=[1, 0, 0], buff=0, color=yellow_color)
        v2 = Arrow(start=ORIGIN, end=[0.5, 0.8, 0], buff=0, color=ORANGE)
        v_sum = Arrow(start=ORIGIN, end=[1.5, 0.8, 0], buff=0, color=WHITE)
        v2_shifted = v2.copy().move_to(v1.get_end() + v2.get_length()*0.5*v2.get_unit_vector())
        
        combo_group = VGroup(v1, v2, v_sum, v2_shifted).scale(0.8)
        self.place_in_area(combo_group, "A1", "B3")
        
        # Span (territory) - Right Area (A4-B6)
        # Create a shaded plane represented by a parallelogram
        span_plane = Polygon(
            [0,0,0], [1.5,0,0], [2,1.2,0], [0.5,1.2,0],
            fill_opacity=0.3, fill_color=yellow_color, stroke_width=0
        )
        v_base1 = Arrow(start=ORIGIN, end=[0.6, 0, 0], buff=0, color=yellow_color)
        v_base2 = Arrow(start=ORIGIN, end=[0.2, 0.4, 0], buff=0, color=ORANGE)
        span_group = VGroup(span_plane, v_base1, v_base2)
        self.place_in_area(span_group, "A4", "B6")

        self.play(
            GrowArrow(v1), GrowArrow(v2),
            run_time=0.8
        )
        self.play(
            Create(v2_shifted),
            GrowArrow(v_sum),
            run_time=0.8
        )
        self.play(
            FadeIn(span_plane),
            GrowArrow(v_base1),
            GrowArrow(v_base2),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color Line 2
        self.play(
            self.lecture[1].animate.set_color(blue_color),
            FadeOut(combo_group),
            FadeOut(span_group)
        )

        # Three vectors in 2D (one is redundant)
        # Place in area C1-D6
        u = Arrow(start=ORIGIN, end=[1, 0.2, 0], buff=0, color=blue_color)
        v = Arrow(start=ORIGIN, end=[0.2, 1, 0], buff=0, color=blue_color)
        w = Arrow(start=ORIGIN, end=[1.2, 1.2, 0], buff=0, color=RED) # redundant
        
        # Shaded span area
        full_span = Polygon(
            [-1.5,-1.5,0], [1.5,-1.5,0], [1.5,1.5,0], [-1.5,1.5,0],
            fill_opacity=0.2, fill_color=blue_color, stroke_width=1
        )
        
        redundancy_group = VGroup(full_span, u, v, w)
        self.place_in_area(redundancy_group, "C1", "D6", scale_factor=0.7)

        self.play(
            Create(full_span),
            GrowArrow(u), GrowArrow(v), GrowArrow(w),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Redundant vector disappears, span remains
        self.play(
            FadeOut(w),
            u.animate.set_color(WHITE),
            v.animate.set_color(WHITE),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color Line 3
        self.play(self.lecture[2].animate.set_color(green_color))

        # Basis vectors expand into a grid
        # Clear middle area, focus on DNA metaphor
        grid_lines = VGroup()
        for i in range(-3, 4):
            grid_lines.add(Line([-2, i*0.5, 0], [2, i*0.5, 0], stroke_width=0.5, stroke_opacity=0.3))
            grid_lines.add(Line([i*0.5, -2, 0], [i*0.5, 2, 0], stroke_width=0.5, stroke_opacity=0.3))
        
        basis_u = Arrow(start=ORIGIN, end=[0.5, 0, 0], buff=0, color=green_color, stroke_width=4)
        basis_v = Arrow(start=ORIGIN, end=[0, 0.5, 0], buff=0, color=green_color, stroke_width=4)
        
        dna_group = VGroup(grid_lines, basis_u, basis_v)
        self.place_in_area(dna_group, "E1", "F6", scale_factor=0.8)

        # Load DNA icon
        try:
            dna_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dna.svg")
            dna_icon.set_color(green_color)
        except:
            # Fallback if asset is missing (though prompt says it MUST be used)
            dna_icon = Text("DNA", color=green_color)
            
        self.place_at_grid(dna_icon, "E3", scale_factor=0.6)
        
        dna_text = Text("The DNA of Space", font_size=20, color=green_color)
        self.place_at_grid(dna_text, "E5", scale_factor=1.0)

        self.play(
            ReplacementTransform(VGroup(u, v), VGroup(basis_u, basis_v)),
            FadeOut(full_span),
            Create(grid_lines),
            run_time=1.5
        )
        
        self.play(
            FadeIn(dna_icon, shift=UP),
            Write(dna_text),
            run_time=1
        )
        
        self.wait(2)
