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
        # Setup the layout
        lecture_lines = [
            'Light intensity fades as distance from the source increases.',
            'Doubling the distance spreads light over four times the area.',
            'This is the inverse square law: brightness equals 1/d².'
        ]
        self.setup_layout("Prerequisite 1: The Inverse Square Law", lecture_lines)

        # Colors
        light_color = "#FFFF00"  # Yellow
        area_color = "#00FFFF"   # Cyan
        ray_color = "#FFFFFF"    # White

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(light_color)
        
        # Central light source [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/lightbulb.svg] at D1
        source = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/lightbulb.svg")
        source.set_color(light_color)
        self.place_at_grid(source, "D1", scale_factor=0.6)
        
        source_glow = Dot(color=light_color, radius=0.3, fill_opacity=0.2)
        source_glow.move_to(source.get_center())
        
        self.play(DrawBorderThenFill(source), FadeIn(source_glow))

        # Circular waves emitting outwards
        waves = VGroup()
        for r in range(1, 4):
            wave = Circle(radius=0.1, color=light_color, stroke_width=2, stroke_opacity=0.8)
            wave.move_to(source.get_center())
            waves.add(wave)

        self.play(
            Succession(
                *[
                    wave.animate(run_time=1.5, rate_func=linear).scale(30).set_stroke(opacity=0)
                    for wave in waves
                ],
                lag_ratio=0.4
            )
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(light_color)

        source_pos = source.get_center()

        # First square at distance d
        square_d = Square(side_length=0.8, color=area_color, fill_opacity=0.2)
        self.place_at_grid(square_d, "D3")
        
        # Second square at distance 2d (composed of 4 smaller squares)
        square_2d_group = VGroup(*[
            Square(side_length=0.8, color=area_color, fill_opacity=0.2)
            for _ in range(4)
        ]).arrange_in_grid(2, 2, buff=0)
        self.place_at_grid(square_2d_group, "D5")

        # Rays from source to the corners of square_2d
        c1 = square_2d_group.get_corner(UL)
        c2 = square_2d_group.get_corner(UR)
        c3 = square_2d_group.get_corner(DL)
        c4 = square_2d_group.get_corner(DR)
        
        rays = VGroup(
            Line(source_pos, c1, color=ray_color, stroke_width=1, stroke_opacity=0.4),
            Line(source_pos, c2, color=ray_color, stroke_width=1, stroke_opacity=0.4),
            Line(source_pos, c3, color=ray_color, stroke_width=1, stroke_opacity=0.4),
            Line(source_pos, c4, color=ray_color, stroke_width=1, stroke_opacity=0.4),
        )

        # Labels for distances - Using Text
        d_label = Text("d", color=WHITE, font_size=24)
        d2_label = Text("2d", color=WHITE, font_size=24)
        self.place_at_grid(d_label, "E3", scale_factor=1.0)
        self.place_at_grid(d2_label, "E5", scale_factor=1.0)

        self.play(Create(rays), FadeIn(square_d), FadeIn(d_label))
        self.wait(0.5)
        self.play(FadeIn(square_2d_group), FadeIn(d2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(light_color)

        # Intensity labels
        intensity_d = Text("I", color=light_color, font_size=32)
        intensity_2d = Text("I/4", color=light_color, font_size=32)
        
        self.place_at_grid(intensity_d, "C3", scale_factor=1.0)
        self.place_at_grid(intensity_2d, "C5", scale_factor=1.0)
        
        # Law formula near the top right
        law_formula = Text("I ∝ 1/d²", color=light_color, font_size=36)
        self.place_in_area(law_formula, "A4", "B6", scale_factor=1.0)

        self.play(Write(intensity_d))
        self.play(Write(intensity_2d))
        self.play(Write(law_formula))
        
        self.wait(3)
