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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup
        title_text = "The Geometry of Velocity: Mapping to a Circle"
        lecture_lines = [
            'We map these velocities onto a geometric plane.',
            'Rescaling the axes transforms our equations into a circle.',
            "Energy conservation keeps the system on this circle's edge.",
            'Each block-to-block collision is a jump along the circle.',
            'Wall collisions reflect the point across the vertical axis.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets
        asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/block.svg"
        block_small = SVGMobject(asset_path, color=BLUE, fill_opacity=1)
        block_large = SVGMobject(asset_path, color=RED, fill_opacity=1)
        
        # Grid Center for Geometry
        grid_center = (self.grid["A1"] + self.grid["F6"]) / 2

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#888888")
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=4,
            axis_config={"color": "#888888"},
            tips=False
        )
        self.place_in_area(axes, "A1", "F6")
        
        label_v1 = Text("v1", font_size=20, color="#888888")
        label_v2 = Text("v2", font_size=20, color="#888888")
        self.place_at_grid(label_v1, "E6")
        self.place_at_grid(label_v2, "A4")

        self.play(Create(axes), Write(label_v1), Write(label_v2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        label_x = Text("x = sqrt(m)v1", font_size=18, color=WHITE)
        label_y = Text("y = sqrt(M)v2", font_size=18, color=WHITE)
        self.place_at_grid(label_x, "E6")
        self.place_at_grid(label_y, "A4")

        self.play(
            FadeOut(label_v1), FadeOut(label_v2),
            FadeIn(label_x), FadeIn(label_y),
            axes.animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        circle = Circle(radius=1.8, color="#00FFFF")
        circle.move_to(axes.c2p(0, 0))
        
        self.play(Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        
        # Initial point on circle
        angle = 0.3
        state_point = Dot(axes.c2p(1.8 * np.cos(angle), 1.8 * np.sin(angle)), color=YELLOW)
        
        # Blocks for representation
        self.place_at_grid(block_small, "E2", scale_factor=0.3)
        self.place_at_grid(block_large, "E5", scale_factor=0.6)
        
        self.play(FadeIn(state_point), FadeIn(block_small), FadeIn(block_large))
        
        # Jump animation (simulate collision)
        new_angle = 1.2
        target_pos = axes.c2p(1.8 * np.cos(new_angle), 1.8 * np.sin(new_angle))
        
        self.play(
            block_small.animate.shift(RIGHT * 1),
            block_large.animate.shift(LEFT * 1),
            run_time=0.5
        )
        self.play(
            state_point.animate.move_to(target_pos),
            block_small.animate.shift(LEFT * 0.5),
            block_large.animate.shift(RIGHT * 0.5),
            run_time=0.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        
        # Wall reflection
        wall = Line(self.grid["B1"], self.grid["E1"], color=WHITE)
        reflect_angle = PI - new_angle
        reflect_pos = axes.c2p(1.8 * np.cos(reflect_angle), 1.8 * np.sin(reflect_angle))
        
        self.play(Create(wall))
        self.play(
            block_small.animate.move_to(self.grid["E1"] + RIGHT * 0.3),
            run_time=0.5
        )
        self.play(
            state_point.animate.move_to(reflect_pos),
            block_small.animate.move_to(self.grid["E2"]),
            run_time=0.5
        )
        self.wait(2)
