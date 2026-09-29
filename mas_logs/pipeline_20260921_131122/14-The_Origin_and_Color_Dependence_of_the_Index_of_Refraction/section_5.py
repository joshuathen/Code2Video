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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Chromatic Aberration in Cameras", [
            "Simple lenses cannot focus all colors together.",
            "This results in color fringing in images.",
            "Achromatic doublets solve this chromatic aberration."
        ])
        
        # Assets
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        
        # Initial placements
        self.place_at_grid(lens, 'D3', scale_factor=0.7)
        
        beam = Line(LEFT*1.5, RIGHT*1.5, color=WHITE, stroke_width=4)
        self.place_in_area(beam, 'D1', 'F6', scale_factor=0.6)
        
        animation_group = VGroup(lens, beam)
        self.place_in_area(animation_group, 'C3', 'E6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(lens), Create(beam))
        self.play(beam.animate.shift(RIGHT*1))
        
        # Spectral colors splitting
        self.place_at_grid(prism, 'E4', scale_factor=0.5)
        self.play(FadeIn(prism))
        
        red_ray = Line(ORIGIN, RIGHT*1.5, color=RED).next_to(prism, RIGHT, buff=0)
        blue_ray = Line(ORIGIN, RIGHT*1.5, color=BLUE).next_to(prism, RIGHT, buff=0)
        red_ray.rotate(0.2, about_point=prism.get_center())
        blue_ray.rotate(-0.2, about_point=prism.get_center())
        
        self.play(Create(red_ray), Create(blue_ray))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        self.place_at_grid(sensor, 'D6', scale_factor=0.5)
        self.play(FadeIn(sensor))
        
        # Blur on sensor
        dot_r = Dot(color=RED).move_to(sensor.get_center() + UP*0.1)
        dot_b = Dot(color=BLUE).move_to(sensor.get_center() + DOWN*0.1)
        self.play(FadeIn(dot_r), FadeIn(dot_b))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Achromatic doublet simulation
        doublet = VGroup(
            Circle(radius=0.5, color=BLUE, stroke_width=3).set_fill(BLUE, opacity=0.2),
            Circle(radius=0.5, color=RED, stroke_width=3).set_fill(RED, opacity=0.2)
        ).arrange(RIGHT, buff=-0.2)
        
        self.play(ReplacementTransform(lens, doublet))
        self.play(
            dot_r.animate.move_to(sensor.get_center()),
            dot_b.animate.move_to(sensor.get_center())
        )
