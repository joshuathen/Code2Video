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
        lecture_lines = [
            "Lenses focus colors at differences.",
            "Chromatic aberration causes blur.",
            "Achromatic doublets correct color fringes."
        ]
        self.setup_layout("Real-World Application: Chromatic Aberration", lecture_lines)
        
        # Setup visual elements
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg")
        self.place_at_grid(lens, 'B3', scale_factor=0.6)
        
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        self.place_at_grid(sensor, 'B5', scale_factor=0.6)
        
        # Rays
        ray_blue = Line(start=[-3, 1, 0], end=lens.get_center(), color='#0000FF')
        ray_red = Line(start=[-3, -1, 0], end=lens.get_center(), color='#FF0000')
        light_rays = VGroup(ray_blue, ray_red)
        
        # Focus points
        focus_blue = Dot(color='#0000FF').move_to(self.grid['B4'])
        focus_red = Dot(color='#FF0000').move_to(self.grid['B5'])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color('#FFFF00')
        self.play(Create(lens), Create(light_rays))
        self.play(
            FadeIn(focus_blue),
            FadeIn(focus_red)
        )

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color('#FFFF00')
        self.play(FadeIn(sensor))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color('#FFFF00')
        # Simulate correction
        self.play(
            focus_blue.animate.move_to(focus_red.get_center()),
            sensor.animate.move_to(focus_red.get_center())
        )
        self.wait(2)
