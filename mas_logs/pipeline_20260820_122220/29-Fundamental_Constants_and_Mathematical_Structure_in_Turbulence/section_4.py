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
        self.setup_layout("Dissipation Scales: Where Constants Meet Reality", [
            "Viscosity dominates at the Kolmogorov scale.",
            "Kinetic energy converts into internal heat.",
            "The dissipation range is fluid specific."
        ])
        
        # Elements
        eta_label = MathTex(r"\\eta", color=BLUE)
        swirl = Circle(radius=0.5, color=BLUE)
        viscous_waves = VGroup(*[Line(start=ORIGIN, end=UP*0.2, color=YELLOW) for _ in range(8)]).arrange(RIGHT)
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg", color=WHITE)
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg", color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(swirl, "B5")
        self.place_at_grid(eta_label, "B6")
        self.place_at_grid(microscope, "B4")
        self.play(Create(swirl), Write(eta_label), FadeIn(microscope))
        self.place_in_area(viscous_waves, "C4", "D6", scale_factor=0.9)
        self.play(Create(viscous_waves), swirl.animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        energy_box = Square(side_length=1.5, color=WHITE)
        heat_particles = VGroup(*[Dot(color=RED) for _ in range(20)])
        self.place_in_area(energy_box, "E4", "F6", scale_factor=0.8)
        self.place_at_grid(heat_particles, "E5", scale_factor=0.5)
        self.place_at_grid(thermometer, "F5")
        self.play(Create(energy_box), FadeIn(heat_particles), FadeIn(thermometer))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        fluid_text = Text("Fluid Dependent", font_size=20, color=GREEN)
        self.place_at_grid(fluid_text, "B4", scale_factor=0.7)
        self.play(Write(fluid_text))
        self.wait(2)
