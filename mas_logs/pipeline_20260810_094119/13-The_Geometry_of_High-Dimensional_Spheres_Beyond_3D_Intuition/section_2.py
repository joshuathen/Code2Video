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
        lecture_lines = [
            "As dimensions increase, volume behaves strangely.",
            "Total volume concentrates near the sphere's surface.",
            "Almost all volume resides in the peel.",
            "The center becomes practically empty.",
            "Counter-intuitive compared to 3D."
        ]
        self.setup_layout("The Counter-Intuitive Volume Collapse", lecture_lines)
        
        # Colors for lecture lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Elements
        line1d = Line(LEFT*0.5, RIGHT*0.5, color=BLUE)
        square2d = Square(side_length=1.5, color=GREEN)
        
        # Using assets
        cube3d = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color=YELLOW)
        n_dim_text = Text("N-Dimensions", font_size=24, color=ORANGE)
        
        # Sphere contrast using asset
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=RED)
        sphere_contrast = VGroup(
            sphere_asset,
            Square(side_length=1.5, color=WHITE, stroke_opacity=0.3)
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        self.play(Create(self.place_at_grid(line1d, 'B3', scale_factor=0.8)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        self.play(Create(self.place_at_grid(square2d, 'C3', scale_factor=0.6)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        self.play(Create(self.place_at_grid(cube3d, 'D3', scale_factor=0.8)))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        self.play(Write(self.place_at_grid(n_dim_text, 'E3', scale_factor=0.8)))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.play(Create(self.place_in_area(sphere_contrast, 'B4', 'E6', scale_factor=0.5)))
        self.wait(2)
