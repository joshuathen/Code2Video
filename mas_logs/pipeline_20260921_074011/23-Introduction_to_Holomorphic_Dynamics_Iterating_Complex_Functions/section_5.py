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
            "Dynamics evolve static functions into systems.",
            "Applications range from signals to engineering.",
            "We study stability for system control."
        ]
        self.setup_layout("Summary and Real-World Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # List key concepts
        concepts = VGroup(
            Text("Static Function", font_size=24, color=BLUE),
            Text("Iterative Map", font_size=24, color=GREEN),
            Text("System Evolution", font_size=24, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(concepts, 'B2', scale_factor=1.0)
        
        self.play(Write(concepts))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show nature fractal examples/signals
        signal_wave = FunctionGraph(lambda x: 0.5 * np.sin(4*x), x_range=[-2, 2], color=PURPLE)
        fractal_tree = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fern.svg").set_color(GREEN)
        
        self.place_at_grid(signal_wave, 'E3', scale_factor=0.7)
        self.place_at_grid(fractal_tree, 'F5', scale_factor=0.6)
        
        self.play(Create(signal_wave), FadeIn(fractal_tree))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Stability regions
        stability_area = Circle(radius=1.0, color=RED, fill_opacity=0.3)
        control_dot = Dot(color=WHITE)
        
        self.place_at_grid(stability_area, 'C5', scale_factor=1.0)
        self.place_at_grid(control_dot, 'C6', scale_factor=0.5)
        
        self.play(FadeIn(stability_area), GrowFromCenter(control_dot))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
