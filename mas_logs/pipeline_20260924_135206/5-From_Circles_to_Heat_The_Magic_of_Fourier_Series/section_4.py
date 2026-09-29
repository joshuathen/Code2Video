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
        self.setup_layout("Visual Synthesis and Application", [
            "Decompose a jagged temperature spike into waves.",
            "Each individual wave component decays over time.",
            "Summing these waves reconstructs smooth heat curves."
        ])
        
        # Colors for lines
        colors = [YELLOW, BLUE, GREEN]
        
        # Setup Axes for the heat plot
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 2, 0.5], x_length=4, y_length=3)
        self.place_in_area(axes, "B4", "E6", scale_factor=0.55)
        self.add(axes)

        # Load SVG assets
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        
        # Show initial jagged spike
        spike = axes.plot(lambda x: 1.5 if 1.5 < x < 2.5 else 0.5, x_range=[0, 4])
        spike.set_color(WHITE)
        spike_label = Text("Spike", font_size=16).next_to(spike, UP)
        
        therm = thermometer.copy()
        self.place_at_grid(therm, "B5", scale_factor=0.3)
        
        self.play(Create(spike), Write(spike_label), FadeIn(therm))
        
        # Decompose into waves
        waves = VGroup(*[axes.plot(lambda x, n=i: 0.3 * np.sin(n * PI * x), x_range=[0, 4]) for i in range(1, 4)])
        waves.set_color(colors[0])
        self.play(Transform(spike, waves), FadeOut(spike_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        
        # Waves decay
        self.play(*[wave.animate.set_stroke(opacity=0.3) for wave in waves])
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        
        # Summing to smooth curve
        final_curve = axes.plot(lambda x: 1.0 + 0.3 * np.sin(2 * PI * x), x_range=[0, 4])
        final_curve.set_color(colors[2])
        
        self.play(ReplacementTransform(waves, final_curve))
        self.play(therm.animate.move_to(axes.c2p(3, 1.3)), run_time=1)
        
        # Pulse highlight
        self.play(final_curve.animate.scale(1.1), run_time=0.5)
        self.play(final_curve.animate.scale(1/1.1), run_time=0.5)
        self.wait(1)
