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
            "This formula defines modern signal processing.",
            "Complex exponentials represent oscillating waves perfectly.",
            "Technology converts signals into digital data using this logic."
        ]
        self.setup_layout("Real-World Application: Signal Processing", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show a sine wave input signal originating from a smartphone
        smartphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")
        self.place_at_grid(smartphone, "A2", scale_factor=0.3)
        wave = FunctionGraph(lambda t: np.sin(4 * t), x_range=[-1.5, 1.5], color="#00FFFF")
        self.place_in_area(wave, "A4", "B6", scale_factor=0.6)
        
        self.play(FadeIn(smartphone), Create(wave))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Represent signal as a rotating vector in complex plane
        circle = Circle(radius=1.0, color="#FF00FF")
        vector = Arrow(start=ORIGIN, end=RIGHT, color="#FF00FF")
        self.place_at_grid(circle, "C3", scale_factor=0.5)
        self.place_at_grid(vector, "C3", scale_factor=0.5)
        
        self.play(Create(circle), GrowArrow(vector))
        self.play(Rotate(vector, angle=2*PI, about_point=vector.get_start()), run_time=2)
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        # Display frequency spectrum output of signal processed by antenna
        bars = VGroup(*[Rectangle(height=np.random.rand()*1.5, width=0.3, color="#00FF00", fill_opacity=0.8) for _ in range(5)])
        bars.arrange(RIGHT, buff=0.1)
        self.place_in_area(bars, "D3", "F5", scale_factor=0.5)
        antenna = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/antenna.svg")
        self.place_at_grid(antenna, "F6", scale_factor=0.3)
        
        self.play(FadeIn(bars), FadeIn(antenna))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
