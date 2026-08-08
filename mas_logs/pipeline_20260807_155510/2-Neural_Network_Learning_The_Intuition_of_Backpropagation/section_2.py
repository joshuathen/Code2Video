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
        self.setup_layout("Prerequisite: The Loss Landscape", [
            "Error values form a complex topographical landscape.",
            "We aim to find the lowest valley here.",
            "Each weight acts like a coordinate on this map."
        ])
        
        self.place_at_grid(self.title, 'A3', scale_factor=1.0)
        self.place_in_area(self.lecture, 'B1', 'E1', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.lecture_texts[0].set_color("#FFFFFF")
        # Creating a 3D-like plot (using Surface for landscape)
        landscape = Surface(
            lambda u, v: np.array([
                u,
                v,
                0.5 * (np.sin(u) + np.cos(v))
            ]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(15, 15)
        ).set_style(fill_opacity=0.8, stroke_color=BLUE, stroke_width=0.5)
        self.place_in_area(landscape, 'C2', 'F5', scale_factor=0.9)
        self.play(Create(landscape))

        # === Animation for Lecture Line 2 ===
        self.lecture_texts[1].set_color("#FFFF00")
        min_point = Dot(color="#FFFF00")
        min_point.move_to(landscape.get_center())
        self.play(Create(min_point))

        # === Animation for Lecture Line 3 ===
        self.lecture_texts[2].set_color("#00FFFF")
        # Load asset
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg", color="#00FFFF")
        ball.move_to(landscape.get_center() + OUT * 0.5)
        self.play(FadeIn(ball))
        # Path movement (simple linear animation for the weight state)
        path = Line(start=ball.get_center(), end=landscape.get_center() + UP*0.5, color="#00FFFF")
        self.play(MoveAlongPath(ball, path))
        self.wait(2)
