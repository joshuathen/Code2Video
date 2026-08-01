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
        self.setup_layout("The Master Map: The Mandelbrot Set", [
            "The Mandelbrot set maps all possible c values.",
            "Black regions represent c values with connected Julia sets.",
            "It serves as a dictionary for all complex behaviors."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Mandelbrot set from SVG [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg]
        m_set = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        m_set.set_fill("#222222", opacity=1.0).set_stroke(WHITE, width=2)
        
        # Positioning according to Issue 27: Resize and move to reduce crowding
        self.place_in_area(m_set, 'B2', 'E5', scale_factor=1.7)
        
        # Animate creation
        self.play(Create(m_set), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Inside point 'c'
        c_point = Dot(color=GREEN).scale(1.2)
        # Position c inside the main cardioid area relative to the SVG center
        c_pos_in = m_set.get_center() + np.array([0.2, 0.1, 0])
        c_point.move_to(c_pos_in)
        c_label = Text("c", color=WHITE, font_size=20, font="Serif", slant=ITALIC).next_to(c_point, UP, buff=0.1)
        
        # Julia Set Preview frame [Issue 26: Use larger area and better scale]
        julia_bg = RoundedRectangle(corner_radius=0.1, height=1.6, width=1.6, color=WHITE, fill_opacity=0.1)
        self.place_in_area(julia_bg, 'A5', 'B6', scale_factor=0.8)
        
        # Connected Julia representation (stylized blob)
        julia_conn = Circle(radius=0.35, color=GREEN, fill_opacity=0.4).set_stroke(GREEN, 2)
        julia_conn.move_to(julia_bg.get_center())
        # Internal detail
        swirl = AnnularSector(inner_radius=0.08, outer_radius=0.3, angle=PI, start_angle=0, color=GREEN, fill_opacity=0.3).move_to(julia_conn)
        julia_conn_group = VGroup(julia_conn, swirl)
        
        julia_title = Text("Connected Julia Set", font_size=16, color=GREEN).next_to(julia_bg, UP, buff=0.1)

        # Pulse inside point and show preview
        self.play(FadeIn(c_point), FadeIn(c_label))
        self.play(c_point.animate.scale(1.5), run_time=0.4)
        self.play(c_point.animate.scale(1/1.5), run_time=0.4)
        
        self.play(FadeIn(julia_bg), FadeIn(julia_conn_group), FadeIn(julia_title))
        self.wait(1)
        
        # Move c outside the set
        c_pos_out = m_set.get_center() + np.array([1.2, 1.0, 0])
        
        # Disconnected Julia representation (Dust)
        np.random.seed(42)
        julia_dust = VGroup(*[
            Dot(radius=0.03, color=RED).move_to(
                julia_bg.get_center() + np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0])
            ) for _ in range(30)
        ])
        julia_title_dust = Text("Julia 'Dust'", font_size=16, color=RED).next_to(julia_bg, UP, buff=0.1)

        self.play(
            c_point.animate.move_to(c_pos_out).set_color(RED),
            c_label.animate.next_to(c_pos_out, UP, buff=0.1),
            ReplacementTransform(julia_conn_group, julia_dust),
            ReplacementTransform(julia_title, julia_title_dust),
            run_time=1.5
        )
        
        # Pulse outside
        self.play(c_point.animate.scale(1.5), run_time=0.4)
        self.play(c_point.animate.scale(1/1.5), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        
        # Load dictionary asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg]
        dict_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg")
        self.place_at_grid(dict_icon, 'F6', scale_factor=0.6)
        
        # Dictionary concept visual: flash various points on the Mandelbrot set
        sample_points = [
            m_set.get_center() + np.array([-0.15, 0.08, 0]),
            m_set.get_center() + np.array([0.1, -0.2, 0]),
            m_set.get_center() + np.array([0.3, 0.3, 0]),
            m_set.get_center() + np.array([-0.5, 0, 0])
        ]
        
        flashes = [Flash(p, color=BLUE, flash_radius=0.2, line_length=0.1) for p in sample_points]
        
        self.play(FadeIn(dict_icon))
        self.play(LaggedStart(*flashes, lag_ratio=0.4), run_time=2)
        self.wait(2)
