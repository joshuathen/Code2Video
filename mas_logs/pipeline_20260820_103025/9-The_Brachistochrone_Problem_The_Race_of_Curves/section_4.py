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
        self.setup_layout("Conceptual Proof: Snell’s Law Analogy", 
                          ["Light follows the path of least time.", 
                           "Gravity acts like a variable medium.", 
                           "Particle speed matches light in refraction."])
        
        # Setup Refraction Layers
        layers = VGroup(*[Rectangle(width=6, height=0.5, fill_opacity=0.3, fill_color=interpolate_color(BLUE, GREY, i/3), stroke_width=0) for i in range(4)])
        layers.arrange(DOWN, buff=0)
        self.place_in_area(layers, 'D2', 'F6', scale_factor=0.6)
        
        # Load Assets
        flashlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        
        self.place_at_grid(flashlight, "C4", scale_factor=0.7)
        self.place_at_grid(prism, "C5", scale_factor=0.7)
        self.place_at_grid(mirror, "C6", scale_factor=0.7)
        
        light_path = VMobject(color="#FFFF00", stroke_width=4)
        light_path.set_points_smoothly([self.grid["C4"], self.grid["D5"], self.grid["E6"]])

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(flashlight), Create(light_path), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8C00"))
        self.play(FadeIn(prism), layers.animate.set_fill(opacity=0.6), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(mirror))
        # Add a particle on the path
        particle = Dot(color=WHITE)
        particle.move_to(light_path.get_start())
        self.play(MoveAlongPath(particle, light_path), run_time=3, rate_func=linear)
        self.play(FadeOut(particle))
