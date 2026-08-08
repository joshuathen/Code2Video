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
        self.setup_layout("Visualizing the Fractal Path", [
            "Disk movements trace a beautiful, hidden fractal path.",
            "The state space forms a perfect Sierpinski Triangle.",
            "Binary counting maps a perfect Hamiltonian path through it."
        ])
        
        # Color definitions
        FRACTAL_COLOR = "#00FFFF"
        PATH_COLOR = YELLOW
        HIGHLIGHT_COLOR = "#FFD700"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(FRACTAL_COLOR)
        
        # Grid anchors for vertices (Constraints: within 1 unit)
        # Main triangle vertices
        v1 = self.grid["B3"]
        v2 = self.grid["E1"]
        v3 = self.grid["E5"]
        
        main_triangle = Polygon(v1, v2, v3, color=FRACTAL_COLOR, stroke_width=4)
        
        # State labels near vertices
        label1 = MathTex("S_0", font_size=24, color=FRACTAL_COLOR)
        label2 = MathTex("S_1", font_size=24, color=FRACTAL_COLOR)
        label3 = MathTex("S_2", font_size=24, color=FRACTAL_COLOR)
        
        self.place_at_grid(label1, "A3", scale_factor=1.0)
        self.place_at_grid(label2, "F1", scale_factor=1.0)
        self.place_at_grid(label3, "F5", scale_factor=1.0)
        
        self.play(Create(main_triangle), Write(label1), Write(label2), Write(label3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(FRACTAL_COLOR)
        
        # Subdivide the triangle into four smaller ones to begin the fractal
        m12 = (v1 + v2) / 2
        m23 = (v2 + v3) / 2
        m31 = (v3 + v1) / 2
        
        t1 = Polygon(v1, m12, m31, color=FRACTAL_COLOR, stroke_width=2)
        t2 = Polygon(v2, m12, m23, color=FRACTAL_COLOR, stroke_width=2)
        t3 = Polygon(v3, m23, m31, color=FRACTAL_COLOR, stroke_width=2)
        
        self.play(
            FadeOut(main_triangle),
            Create(t1), Create(t2), Create(t3),
            FadeOut(label1), FadeOut(label2), FadeOut(label3)
        )
        self.wait(1)

        # Second level of subdivision (Level 2 fractal)
        tris_lv2 = VGroup()
        for p in [(v1, m12, m31), (v2, m12, m23), (v3, m23, m31)]:
            sm12 = (p[0] + p[1]) / 2
            sm23 = (p[1] + p[2]) / 2
            sm31 = (p[2] + p[0]) / 2
            tris_lv2.add(Polygon(p[0], sm12, sm31, color=FRACTAL_COLOR, stroke_width=1.5))
            tris_lv2.add(Polygon(p[1], sm12, sm23, color=FRACTAL_COLOR, stroke_width=1.5))
            tris_lv2.add(Polygon(p[2], sm23, sm31, color=FRACTAL_COLOR, stroke_width=1.5))

        self.play(FadeOut(t1), FadeOut(t2), FadeOut(t3), Create(tris_lv2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(FRACTAL_COLOR)
        
        # Binary Counter with fixes from issues 26 and 27
        counter_tracker = ValueTracker(0)
        counter_label = Text("Counter:", font_size=18).set_color(WHITE)
        self.place_at_grid(counter_label, "A4", scale_factor=0.8)
        
        bits = VGroup(*[Rectangle(width=0.25, height=0.15, color=WHITE, stroke_width=1) for _ in range(6)]).arrange(RIGHT, buff=0.1)
        self.place_at_grid(bits, "A5", scale_factor=1.0)

        def update_bits(mob):
            val = int(counter_tracker.get_value())
            bin_str = bin(val)[2:].zfill(6)
            for i, bit in enumerate(bin_str):
                if bit == "1":
                    mob[i].set_fill(HIGHLIGHT_COLOR, opacity=0.8)
                else:
                    mob[i].set_fill(BLACK, opacity=0)
        
        bits.add_updater(update_bits)
        self.add(counter_label, bits)

        # Representative Hamiltonian Path points for tracing
        path_points = [
            v1, (v1+m12)/2, m12, (v2+m12)/2, v2, 
            (v2+m23)/2, m23, (v3+m23)/2, v3, 
            (v3+m31)/2, m31, (v1+m31)/2, v1
        ]
        path_segments = VGroup()
        for i in range(len(path_points)-1):
            path_segments.add(Line(path_points[i], path_points[i+1], color=PATH_COLOR, stroke_width=3))

        # Trace the path and increment counter
        self.play(
            counter_tracker.animate.set_value(63),
            Create(path_segments),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # Transition to multi-level Sierpinski Gasket
        def get_gasket(order, side, center):
            if order == 0:
                return Polygon(
                    center + UP*side, 
                    center + LEFT*side*0.866 + DOWN*side*0.5,
                    center + RIGHT*side*0.866 + DOWN*side*0.5,
                    stroke_width=0.5, color=FRACTAL_COLOR, fill_opacity=0.3
                )
            else:
                s = side/2
                return VGroup(
                    get_gasket(order-1, s, center + UP*s),
                    get_gasket(order-1, s, center + LEFT*s*0.866 + DOWN*s*0.5),
                    get_gasket(order-1, s, center + RIGHT*s*0.866 + DOWN*s*0.5)
                )

        multi_level_gasket = get_gasket(4, 2.2, self.grid["D3"])
        
        self.play(
            FadeOut(tris_lv2),
            FadeOut(path_segments),
            FadeIn(multi_level_gasket)
        )
        self.wait(2)

        # Transition fractal into Towers of Hanoi solution view
        # Fix from issue 28: Move hanoi_demo to E5-F6 to avoid obstruction
        pegs = VGroup(
            Line(DOWN, UP).shift(LEFT*0.6),
            Line(DOWN, UP),
            Line(DOWN, UP).shift(RIGHT*0.6)
        ).set_color(GREY_B)
        
        base_line = Line(LEFT, RIGHT).scale(0.8).next_to(pegs, DOWN, buff=0)
        
        disks = VGroup(
            Rectangle(width=0.6, height=0.15, color=RED, fill_opacity=1),
            Rectangle(width=0.4, height=0.15, color=GREEN, fill_opacity=1),
            Rectangle(width=0.2, height=0.15, color=BLUE, fill_opacity=1)
        ).arrange(UP, buff=0.05).move_to(pegs[0].get_bottom() + UP*0.1)
        
        hanoi_demo = VGroup(pegs, base_line, disks)
        self.place_in_area(hanoi_demo, "E5", "F6", scale_factor=0.8)
        
        self.play(
            FadeOut(multi_level_gasket),
            FadeOut(counter_label),
            FadeOut(bits),
            FadeIn(hanoi_demo)
        )
        
        # Final movement illustration
        self.play(
            disks[2].animate.move_to(pegs[2].get_bottom() + UP*0.1),
            run_time=1.5
        )
        self.wait(2)
