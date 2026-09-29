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
        self.setup_layout("Visualizing Newton Fractals", [
            "Boundaries between basins are not simple straight lines.",
            "They form infinitely complex, self-similar fractal patterns.",
            "Zooming reveals a tangled, endless web."
        ])
        
        # Elements
        complex_plane = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        fractal_region = Rectangle(width=2, height=2, color="#00FF00", fill_opacity=0.3)
        point = Dot(color="#0000FF", radius=0.05)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(complex_plane, 'D2', 'F5', scale_factor=0.5)
        self.play(Create(complex_plane))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.place_in_area(fractal_region, 'D3', 'E4', scale_factor=0.4)
        self.play(FadeIn(fractal_region))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#0000FF")
        self.place_at_grid(point, 'E3', scale_factor=0.6)
        self.play(FadeIn(point))
        
        # Simulate zoom-in logic as per storyboard
        self.play(
            point.animate.scale(2).set_color(WHITE),
            fractal_region.animate.scale(2).set_opacity(0.1),
            run_time=2
        )
        self.wait(1)
