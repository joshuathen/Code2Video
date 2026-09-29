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
        lecture_lines = [
            "The Mandelbrot set maps all dynamic possibilities.",
            "It determines if orbits remain bounded or diverge.",
            "It showcases stunning self-similarity at every scale.",
            "Mini-Mandelbrots appear throughout the main structure.",
            "This set acts as the ultimate parameter map."
        ]
        self.setup_layout("The Mandelbrot Set: The Map of All Dynamics", lecture_lines)
        
        # Elements
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        compass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        c_plane_label = Text("Parameter c-plane", font_size=24, color=BLUE)
        formula = MathTex(r"f_c(z) = z^2 + c", color=YELLOW)
        mandelbrot_shape = VMobject().set_points_as_corners([
            [-1, 1, 0], [0, 1.5, 0], [1, 1, 0], [1.2, 0, 0], [0.8, -0.5, 0], [0, -1, 0], [-1, -0.5, 0], [-1, 1, 0]
        ]).set_fill(RED, opacity=0.5).set_stroke(WHITE, width=2)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(c_plane_label, 'A3', scale_factor=1.0)
        self.place_at_grid(map_icon, 'A2', scale_factor=0.5)
        self.play(Write(c_plane_label), FadeIn(map_icon))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(formula, 'B3', scale_factor=0.9)
        self.play(Create(formula))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.place_in_area(mandelbrot_shape, 'B4', 'E5', scale_factor=0.6)
        self.play(FadeIn(mandelbrot_shape))

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]))
        self.place_at_grid(compass_icon, 'F6', scale_factor=0.4)
        self.play(mandelbrot_shape.animate.set_color(PURPLE), FadeIn(compass_icon), run_time=1)

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]))
        self.play(
            self.lecture[0].animate.set_color(BLUE),
            self.lecture[1].animate.set_color(YELLOW),
            self.lecture[2].animate.set_color(RED),
            self.lecture[3].animate.set_color(PURPLE),
            self.lecture[4].animate.set_color(GREEN)
        )
        self.wait(2)
