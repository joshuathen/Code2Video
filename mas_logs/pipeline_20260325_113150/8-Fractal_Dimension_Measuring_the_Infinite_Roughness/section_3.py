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
        # Mandatory layout setup
        self.setup_layout(
            "The Fractal Breakpoint: The Sierpinski Gasket",
            [
                "Meet the Sierpinski Gasket, a fractal triangle.",
                "Double its side length and count the copies.",
                "Instead of four triangles, we see three.",
                "The middle section is always removed.",
                "Its dimension sits between one and two."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5555"))
        
        main_tri = Triangle(color="#FF5555", fill_opacity=1).set_stroke(width=2)
        # Fix Issue 50: reduced scale factor to 1.0
        self.place_in_area(main_tri, "B2", "D4", scale_factor=1.0)
        
        self.play(DrawBorderThenFill(main_tri))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF77")
        )
        
        v = main_tri.get_vertices()
        m_01 = (v[0] + v[1]) / 2
        m_12 = (v[1] + v[2]) / 2
        m_20 = (v[2] + v[0]) / 2
        
        t_top = Polygon(v[0], m_01, m_20, color="#FF5555", fill_opacity=1).set_stroke(width=1)
        t_bl  = Polygon(m_01, v[1], m_12, color="#FF5555", fill_opacity=1).set_stroke(width=1)
        t_br  = Polygon(m_20, m_12, v[2], color="#FF5555", fill_opacity=1).set_stroke(width=1)
        t_mid = Polygon(m_01, m_12, m_20, color="#FF5555", fill_opacity=0.4).set_stroke(width=1)
        
        s_label = Text("S = 2", font_size=24, color="#FFFF77")
        # Fix Issue 51: Moved to B5
        self.place_at_grid(s_label, "B5", scale_factor=1.0)

        self.play(
            FadeOut(main_tri),
            FadeIn(t_top), FadeIn(t_bl), FadeIn(t_br), FadeIn(t_mid),
            Write(s_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#77FF77")
        )

        self.play(t_mid.animate.set_fill(opacity=0).set_stroke(opacity=0), run_time=1)
        
        n_label = Text("N = 3", font_size=24, color="#77FF77")
        # Fix Issue 51: Moved to C5
        self.place_at_grid(n_label, "C5", scale_factor=1.0)
        
        self.play(Write(n_label))
        self.play(Flash(t_top, color="#77FF77"), Flash(t_bl, color="#77FF77"), Flash(t_br, color="#77FF77"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#77FFFF")
        )
        
        def get_sub_triangles(poly):
            pts = poly.get_vertices()
            mid01 = (pts[0] + pts[1]) / 2
            mid12 = (pts[1] + pts[2]) / 2
            mid20 = (pts[2] + pts[0]) / 2
            return [
                Polygon(pts[0], mid01, mid20, color="#FF5555", fill_opacity=1).set_stroke(width=0.5),
                Polygon(mid01, pts[1], mid12, color="#FF5555", fill_opacity=1).set_stroke(width=0.5),
                Polygon(mid20, mid12, pts[2], color="#FF5555", fill_opacity=1).set_stroke(width=0.5)
            ]

        recursion_level_2 = VGroup(
            *get_sub_triangles(t_top),
            *get_sub_triangles(t_bl),
            *get_sub_triangles(t_br)
        )
        
        self.play(
            FadeOut(t_top, t_bl, t_br),
            FadeIn(recursion_level_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFFFF")
        )

        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        dim_label = Text("1 < D < 2", font_size=28, color="#FFFFFF")
        # Fix Issue 51: Moved to D5
        self.place_at_grid(dim_label, "D5", scale_factor=1.0)
        
        dim_calc = Text("D = log(3) / log(2) ≈ 1.58", font_size=22, color="#FFFFFF")
        # Fix Issue 52: Positioned in area E2-F5
        self.place_in_area(dim_calc, "E2", "F5", scale_factor=0.8)

        self.play(Write(dim_label))
        self.play(FadeIn(dim_calc, shift=UP*0.2))
        self.wait(2)
