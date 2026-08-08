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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Euler’s Characteristic Formula", [
            "Connected planar graphs follow Euler’s Characteristic formula.",
            "Vertices minus edges plus faces equals two.",
            "$V - E + F = 2$ acts as a structural invariant.",
            "Flatten a cube to see this in action.",
            "Eight vertices, twelve edges, six faces, totals two."
        ])
        
        # Formula V - E + F = 2
        formula = MathTex(r"V - E + F = 2", font_size=40, color=WHITE)
        self.place_in_area(formula, "A2", "A5", scale_factor=0.8)
        
        # Cube asset
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", fill_opacity=0.3, stroke_width=2)
        self.place_in_area(cube, "D3", "F5", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), Write(formula))

        # === Animation for Lecture Line 2 ===
        label_v = Text("V", color="#FFA500", font_size=24)
        label_e = Text("E", color="#FFA500", font_size=24)
        label_f = Text("F", color="#FFA500", font_size=24)
        self.place_at_grid(label_v, "B2", scale_factor=0.9)
        self.place_at_grid(label_e, "B3", scale_factor=0.9)
        self.place_at_grid(label_f, "B4", scale_factor=0.9)
        self.play(self.lecture[1].animate.set_color("#FFA500"), FadeIn(label_v), FadeIn(label_e), FadeIn(label_f))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFA500"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"), FadeIn(cube))

        # === Animation for Lecture Line 5 ===
        # Markers distributed across unique grid cells as per B035
        v_markers = VGroup(*[Dot(color="#00FF00", radius=0.08) for _ in range(8)])
        e_markers = VGroup(*[Dot(color="#0000FF", radius=0.06) for _ in range(12)])
        f_markers = VGroup(*[Dot(color="#FF0000", radius=0.1) for _ in range(6)])
        
        # Using arbitrary placement for markers around the cube area
        for i, m in enumerate(v_markers):
            m.move_to(cube.get_center() + RIGHT * (i%4 - 1.5) * 0.2 + UP * (i//4 - 0.5) * 0.2)
        for i, m in enumerate(e_markers):
            m.move_to(cube.get_center() + RIGHT * (i%6 - 2.5) * 0.15 + DOWN * (i//6 + 0.5) * 0.2)
        for i, m in enumerate(f_markers):
            m.move_to(cube.get_center() + UP * (i%3 - 1) * 0.3)
        
        self.play(self.lecture[4].animate.set_color("#FFFFFF"), FadeIn(v_markers), FadeIn(e_markers), FadeIn(f_markers))
        self.wait(2)
