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
        lecture_lines = ["The Mandelbrot set maps all Julia sets.", "It acts as a parameter space dictionary.", "Changing the parameter 'c' alters dynamics."]
        self.setup_layout("The Mandelbrot Set: The Map of All Maps", lecture_lines)
        
        # Load SVG Assets
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        globe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")

        # === Animation for Lecture Line 1 ===
        # Display the Mandelbrot formula z_{n+1} = z_n^2 + c. Force tex_template to ensure rendering.
        formula = MathTex(r"z_{n+1} = z_n^2 + c", color="#FFFFFF", tex_template=TexTemplate())
        self.place_at_grid(formula, "B4", scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Map parameter c values to the set's membership. Color #FFFF00.
        mandel_shape = Circle(color="#FFFF00", fill_opacity=0.5)
        self.place_in_area(mandel_shape, "C4", "E6", scale_factor=0.7)
        self.place_at_grid(map_icon, "C3", scale_factor=0.3)
        self.play(Create(mandel_shape), FadeIn(map_icon), run_time=2)
        self.lecture[1].set_color("#FFFF00")
        
        # Show the set growing as c varies. Color #FF00FF.
        mandel_grow = Circle(color="#FF00FF", fill_opacity=0.3)
        self.place_in_area(mandel_grow, "C4", "E6", scale_factor=1.0)
        self.play(ReplacementTransform(mandel_shape, mandel_grow))
        self.lecture[2].set_color("#FF00FF")

        # Highlight the cardioid and circular bulbs. Color #00FFFF.
        highlight = Dot(self.grid["D4"], color="#00FFFF")
        self.play(FadeIn(highlight))

        # Zoom into a small region revealing infinite complexity. Color #FFFFFF.
        self.place_at_grid(globe_icon, "E2", scale_factor=0.4)
        self.play(FadeIn(globe_icon), highlight.animate.scale(5), run_time=1.5)
        self.wait(1)
